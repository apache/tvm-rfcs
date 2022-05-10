- Feature Name: introducing-decl-buffer
- Author: Wuwei Lin (@vinx13), Eric Lunderberg (@Lunderberg)
- Start Date: 2022-05-04
- RFC PR: [apache/tvm-rfcs#0000](https://github.com/apache/tvm-rfcs/pull/70)
- GitHub Issue: TBD

# Summary
[summary]: #summary

This is a follow-up of https://github.com/apache/tvm/pull/9727 and
[RFC#63](https://github.com/apache/tvm-rfcs/pull/63). Currently buffer can be implicitly
declared and then used. The implicit behavior can be error prone and makes analysis more difficult.
This RFC introduces `DeclBuffer`, a new IR construct as an explicit statement for buffer declaration.

# Motivation
[motivation]: #motivation

Currently a Buffer object can be created and then referenced in TIR, without explicit declaration
or allocation. For example, in TVM script, one can use `T.buffer_decl` to create a new buffer and
then use it in the rest of the program.
```
@T.prim_func
def buffer_alias(A: T.Buffer[(16,), "float"]):
    A_vector = T.buffer_decl([4], "float32x4", data=A.data)
    T.evaluate(A_vector[0])  # read from buffer alias
```
However, `T.buffer_decl` doesn’t translate to a node in AST. The AST will be
```
PrimFunc {
  buffer_map: {A: Buffer[(16,), "float"},
  body: Evaluate {
    BufferLoad {
      buffer: Buffer(data = A.data, [4], "float32x4")  # implicit creation of new buffer
      index: [0]
    }
  }
}
```
In this example, `BufferLoad` loads from an implicitly-created new buffer which aliases another
buffer. This example shows that a data variable can be used to create a buffer in arbitrary ways.
There are no guarantee that the created buffer and the underlying data variable have consistent
physical memory. This makes analysis in TIR difficult and error-prone as one should always check
whether a buffer in TIR is an implicitly-created one. 

By introducing explicit `DeclBuffer` statement, we can require that a buffer must always be declared
before any usages. This makes the creation and the usage of buffer better-managed within TIR.
Developers (e.g pass writers) can collect buffer information such as allocation, aliasing by
visiting `DeclBuffer` nodes.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

`DeclBuffer` will be defined as 
```
class DeclBuffer : public Stmt {
    Buffer buffer;  // the buffer declared
    Stmt body;  // the scope of the buffer
};
```

In TVM script, `T.buffer_decl` will be renamed to `T.decl_buffer` to make the name a verb phase that
is consistent with the existing ones such as `T.alloc_buffer`, `T.match_buffer`. `T.decl_buffer`
will be translated to a `DeclBuffer` object in TIR. This only changes the way parser handles
`T.decl_buffer`, the user API of `T.decl_buffer` in TVM script will stay the same.

In TIR, `DeclBuffer` will be handled in `StmtFunctor`. Visitors or mutators of `DeclBuffer` can be
override to handle `DeclBuffer` in TIR passes.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

## Allocation of intermediate buffer
The intermediate buffer inside `PrimFunc` can be declared and allocated in the following way:

```
Allocate {
  data: A_data(Var(name=...))
  extent: ...
  body: DeclBuffer {
    buffer: A(data=A_data, dtype=..., shape=...),
    body: {
      ...
    }
  }
}
```

Alternatively, a buffer can be declared first and then allocated. In this case, any usage of the
buffer before it is allocated is not allowed.

## Declaration of buffer alias
Buffer declared in `DeclBuffer` can reuse data variable from another buffer. This creates a buffer
alias.

```
DeclBuffer {
  buffer: A(data=Var(name=...), dtype=..., shape=...),
  body: {
    DeclBuffer {
      buffer: A_alias(data=A.data, ...)
      body: ...
    }
  }
}
```

## Replace `preflattened_buffer_map` with buffer alias

Currently, `PrimFunc` has two maps, `preflattened_buffer_map` and `buffer_map`, to specify the input
buffer shapes. Before the flattening passes (`FlattenBuffer` and `StorageFlatten`),
`preflattened_buffer_map` is empty and `buffer_map` contains the logical shapes of the buffers.
After flattening, the logical shapes are moved to `preflattened_buffer_map`, and `buffer_map` will
store the physical shapes of the buffers. The change of the information stored in `buffer_map` can
be confusing. These two maps can be unified into a single `buffer_map` that defines the logical
shapes of the input buffers. The buffer access in physical shape, which is an internal behavior of
`PrimFunc` after flattening, can be achieved by using `DeclBuffer` to create buffer aliases in
physical shapes.

This is illustrated in the example below.

Before flattening:
```
@T.prim_func
def elemwise(A: T.Buffer[(16, 16), "float32"], C: T.Buffer[(16, 16), "float32"]):
    for i, j in T.grid(16, 16):
        C[i, j] = A[i, j]
```

After flattening:
```
@T.prim_func
def elemwise(A: T.Buffer[(16, 16), "float32"], C: T.Buffer[(16, 16), "float32"]):
    A_flattened = T.decl_buffer(A.data, (256,), "float32")
    C_flattened = T.decl_buffer(C.data, (256,), "float32")
    for i, j in T.grid(16, 16):
        C_flattened[i * 16 + j] = A[i * 16 + j]
```

## TVM script updates
* `T.allocate` will return data variable instead of a buffer. If the subsequent program need to access
the data variable as a buffer, it should use `T.decl_buffer` to declare the buffer.
* `T.buffer_decl` will be renamed to `T.decl_buffer`.

## Engineering plan
This RFC introduces a TIR change that may require significant refactor to the existing codebase.
It can be decomposed into three parts to reduce a pull request size.

- Part 1: Introduce `DeclBuffer` data structure, add corresponding visitors in IR functors.
- Part 2: Refactor existing passes and test cases to use `DeclBuffer`.
- Part 3: Enforce the usage of `DeclBuffer`. No implicit buffer declarations are allowed.

# Rationale and alternatives
In S-TIR, there is an alternative to define buffer declarations inside the block, similar to the
existing alloc_buffers, match_buffers:

```
class Block : public Stmt {
  /*! \brief The buffer allocated in the block. */
  Array<Buffer> alloc_buffers;
  /*! \brief The match buffer regions. */
  Array<MatchBufferRegion> match_buffers;
  /*! \brief The buffer declared in the block. */
  Array<Buffer> decl_buffers;
};
```
This unifies the scope of `DeclBuffer` with the block scope. In low-level TIR, a `DeclBuffer`
statement is still needed because Block is not available in low-level TIR. This is similar to the
current status that `block->alloc_buffers` is lowered to Allocate. For now since there are no needs
of `DeclBuffer` during TIR scheduling, we would like to avoid introducing `block->decl_buffers` to
keep it simple. It can be an incremental work upon this when future needs come up.

Another option would be to separate the concepts of memory allocation and buffer access.
A memory allocation would represent the allocation of some number of bytes, and would always use
physical shape. Each buffer would have a backing allocation, and would represent access into some
tensor, and would use logical/transformed shape. Overall, it would be the difference between having
one "real" buffer and multiple aliases, as opposed to having several buffers, and a memory
allocation backing them, emphasizing that there’s nothing special about the first buffer. We decided
this isn’t necessary, because it would add way more boilerplate for the most common case of one
buffer, and would encourage people to make buffer aliases when not necessary.

# Drawbacks
The scope of the buffer in `DeclBuffer` is declared as `body` field. It adds level of recursion in
TIR visitors. Since the number of buffers declared inside a `PrimFunc` is usually small, this is
unlikely a concern.

# Prior art
[prior-art]: #prior-art

Buffer declaration is implicitly supported prior to this RFC. In TVM script, `T.buffer_decl` is used
to declare a buffer, which can be in other TIR expressions and/or statements. This RFC is intended
to formalize this process by using explicit `DeclBuffer` statement.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

Should low-level code generators handle buffer aliases?  One option would be to remove them in a
lowering pass.  Another option would be to use them to represent explicit type casts, rather than
having any implicit typecasts.

When `DeclBuffer` creates a buffer alias, what are the requirements (`shape`, `dtype`,
`elem_offset`, etc.) of the aliasing buffer? The current behavior of the implicit buffer aliasing
is to assume the aliasing buffer is valid, and rely on codegen to handle buffer aliases.

# Future possibilities
[future-possibilities]: #future-possibilities

With explicit `DeclBuffer` statement in TIR, we can introduce analysis passes for buffer aliasing.
This will help the existing TIR passes to explicitly examine whether their assumption on buffer
aliasing are satisfied.

