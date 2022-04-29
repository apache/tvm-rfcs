- Feature Name: Clarifying Buffer Declaration and Access
- Author: Wuwei Lin (@vinx13), Eric Lunderberg (@Lunderberg)
- Start Date: 2022-03-18
- RFC PR: [apache/tvm-rfcs#63](https://github.com/apache/tvm-rfcs/pull/63)
- GitHub Issue: [apache/tvm#10505](https://github.com/apache/tvm/issues/10505)

# Summary 
[summary]: #summary

In https://github.com/apache/tvm/pull/9727 and
[RFC#39](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0039-buffer-physical-layout.md), we
deprecated `Load` and `Store` to use `BufferLoad` and `BufferStore` instead in order to support
generalized multi-dimensional physical buffer access. Here we document necessary clarifications,
implications about the new buffer convention, as well as the post-hoc pass checklist.

# Motivation
[motivation]: #motivation

The main goal of this RFC is to summarize the existing buffer convention and the IR changes in
https://github.com/apache/tvm/pull/9727 which have a broader impact. There are no new semantics
proposed in this RFC.

# Reference - level explanation
[reference-level-explanation]: #reference-level-explanation

**What’s a buffer?**

Buffer is a compile-time representation of contiguous block of memory. Since a Buffer is typically
used as backing storage for a `TensorType`, it includes relevant information from that `TensorType`
which can be sufficiently generalized to an array, such as data type and shape information.
A Buffer needs to be declared and allocated before it can be used.

**Declaration of buffer**

Buffer can be declared in the following ways:

- Inside the `buffer_map` of `PrimFunc`. TIR's type system does not accommodate rich array types,
instead representing them as `T.handle` (typically emitted as `void*`). The `buffer_map` specifies
how to interpret such `T.handle` when using it as a basis for array accesses.
- `T.alloc_buffer` is used `S-TIR` to create and allocate a buffer.
- `T.buffer_decl` can be used to create a buffer alias by specifying the underlying data variable to
reuse the data from another buffer. It can also be used to reinterpret the data type of the buffer.
`T.buffer_decl` can also be used to create a buffer alias with a different `elem_offset`.
`elem_offset` should be handled during the lowering process.

Examples of `T.buffer_decl` is shown below.
```
@T.prim_func
def buffer_alias(A: T.Buffer[(16,), "float"]):
    A_vector = T.buffer_decl([4], "float32x4", data=A.data)

@T.prim_func
def buffer_alloc():
    A = T.buffer_decl([4, 4], "float32")
    Allocate(A.data, [16], "float32")
```

In the future, we will consider renaming `T.buffer_decl` to `T.decl_buffer` to make it name a verb
phase that is consistent with the existing ones like `T.alloc_buffer`, `T.match_buffer`. 

**Allocation of buffer**

In low-level TIR, `tir::Allocate` is used to allocate a data variable with given shapes. `tir::Allocate`
returns a data variable of type `T.handle` (since TIR's type system does not accommodate rich arrays), which may be
reinterpreted with a different shape or data type using `T.buffer_decl`.

**Explicit `DeclBuffer` IR construct**

`T.buffer_decl` doesn't correspond to a TIR node. Instead, `T.buffer_decl` returns either:
- A Buffer node whose data member points to the aliased Buffer.
- A Buffer node whose data member is a new pointer-type Var (the var is expected to be initialized
via tir::Allocate elsewhere)"

The current behavior of `TVMScriptPrinter` is to implicitly print a `T.buffer_decl` at the beginning
of `PrimFunc` for any undefined buffers. The implicit behavior can be error-prone. In light of the
migration, we should consider an explicit `DeclBuffer` as part of the IR. This will be further
discussed in a separate RFC.

**Buffer Aliasing**

`T.buffer_decl` creates a buffer alias if the underlying data variable (`.data` field) overlaps with
another buffer. Buffer created via `T.alloc_buffer` always do not alias. Buffer aliases do not need
`Allocate` to create the data variable -- they may simply reuse the data variable from the Buffer
being aliased. If a transformation would produce multiple allocations of the same buffer var
(e.g. unrolling a loop that contains an allocation), the transform should update the allocations to
be unique using `tvm::tir::ConvertSSA`.

Buffers should not alias each other unless necessary, because aliased buffers increase complexity
for TIR transformations. Passes that rewrite buffers should clearly indicate how aliased buffers
are handled. For example, when changing the underlying layout of stored elements in a buffer, all
buffer aliases must also be updated. Currently, we don't have analysis for buffer aliasing.
This is a future developement task if buffer aliasing is used broadly. Therefore, while buffer
aliasing is typically free at runtime, this imposes a cost for buffer aliasing both to compile times
and development complexity. 

**Discussion: When it is safe to transform a buffer**

We would like to discuss some examples of when it is safe to transform a buffer w.r.t. aliasing rules:
1. reshape
2. layout transform (e.g. swap indices)
3. compact.

(1) is fine under aliasing as long as the low level memory is shared. This is because buffer alias
here is used to reinterpret a buffer, which only changes the way we access the buffer. As long as
there are no other buffer transformations or analysis applied to this buffer, it is safe to use the
alias.

On the other hand, any transformations or analysis applied on a buffer should be clear how to handle
buffer aliases correctly. (2) and (3) are such examples, they would need more
cares. (2) requires all the aliases be changed together. (3) requires to compute the compact buffer
shape and then rewrite the buffer shape. This need us to take all alias into consideration and then
rewrite their shapes together.

**Generalizing buffer accesses**

Previously we used `Load` and `Store` to represent low-level buffer accesses. `Load` and `Store`
consist of data variable, data type and index, which can be directly translated to pointer cast and
accesses in runtime. Note that data type given to `Load` / `Store` can be different from the
Buffer's data variable type. For example,

```python
A = T.buffer_decl(shape=(16,), dtype='float')
T.load("float4", A.data, T.ramp(4, 1, 4))
```

can be translated to

```cpp
*((float4*)(A + 4))
```

in C codegen. 

However, `BufferLoad` and `BufferStore` themselves can not reinterpret a buffer to a different shape
or data type. They always return the data type specified on underlying buffer object. This is the
fundamental difference between `Load/Store` and `BufferLoad/BufferStore` that we need to deal with
carefully.

Vectorized access is achieved by using `Ramp` as index in `Load/Store`. Vectorized buffer access
via `BufferLoad`/`BufferStore` can be achieved either by using a scalar index to access a buffer
that has a vectorized type, or by using `Ramp` as an index into a buffer that has a scalar type.
For N-D buffer indices, it is possible that `Ramp` being used in multiple dimensions
(e.g. `A[Ramp(...), ..., Ramp(...)]` ). In this case the number of lanes of the data type of such
value is the product of each `Ramp`. We limit `Ramp` to only the last dimension as multiple `Ramp`
creates additional complexity.

Different combinations of buffer type and index type (scalar vs. vector) are clarified in

[RFC#39](https://github.com/Lunderberg/tvm-rfcs/blob/data_layout/rfcs/0039-buffer-physical-layout.md#rationale-and-alternatives),
excerpts are the following:

```python
@T.prim_func
def scalar_load_from_scalar_buffer(A: T.Buffer[(64,), "float32"]):
    assert A[0].dtype == "float32"

@T.prim_func
def vector_load_from_vector_buffer(A: T.Buffer[(16,), "float32x4"]):
    assert A[0].dtype == "float32x4"

@T.prim_func
def vector_load_from_vector_buffer(A: T.Buffer[(16,), "float32x4"]):
    A_vector_2 = T.buffer_decl([32], "float32x2", data=A.data)
    assert A[0].dtype == "float32x4"
    assert A_vector_2[0].dtype == "float32x2"

@T.prim_func
def vector_load_from_scalar_buffer_option1(A: T.Buffer[(64,), "float32"]):
    assert A[T.ramp(0, 1, 4)].dtype == "float32x4"

@T.prim_func
def vector_load_from_scalar_buffer_option2(A: T.Buffer[(64,), "float32"]):
    A_vector = T.buffer_decl([16], "float32x4", data=A.data)
    assert A_vector[0].dtype == "float32x4"

@T.prim_func
def scalar_load_from_vector_buffer(A: T.Buffer[(16,), "float32x4"]):
    A_scalar = T.buffer_decl([64], "float32", data=A.data)
    assert A_scalar[0].dtype == "float32"

#multiple dimensional buffer accesses
@T.prim_func
def nd_scalar_load_from_scalar_buffer(A: T.Buffer[(64, 64,), "float32"]):
    assert A[0, 0].dtype == "float32"

@T.prim_func
def nd_vector_load_from_scalar_buffer(A: T.Buffer[(64,64), "float32"]):
    assert A[0, T.ramp(0, 1, 4)].dtype == "float32x4"
```

In rare cases, vector index can be used to access a vector buffer. We leave this usage as
undefined until we have a clear use case.

**VectorBufferRewrite**

In some backend like SPIR-V where runtime pointer casts are not available, even between types that
differ only in the number of lanes (e.g. `float16` and `float16x4.`), `VectorTypeRewriter` will be
used to rewrite the buffer to a vector type. (VectorBufferRewrite rewrites the buffer from
`vector_load_from_scalar_buffer` into `scalar_load_from_vector_buffer` in the above example).

**Removing pre-flattened buffer**

Buffer information before flattening are necessary during compilation. They specify the calling
convention of `PrimFunc` and are translated to assertions of buffer shapes, strides, etc. in
runtime. `preflattened_buffer_map` was introduced in https://github.com/apache/tvm/pull/9727 to
save these information after buffer flattening.

During the lowering process, although buffer accesses inside `PrimFunc` are flattened to match
physical buffer dimensions, the calling convention of the `PrimFunc` are kept unchanged - It still
expect the parameter to have multi-dimensional logical buffer shape. Therefore, we would like to
unify `preflattened_buffer_map` and `buffer_map`. `buffer_map` should be kept unchanged during
buffer flattening. Instead, we declare an aliasing buffer as the flattened buffer after flattening.
For example, after flattening, the TIR will look like

```python
def fn(X: T.Buffer([2, 3], "float32"):
  X_flattened = T.buffer_decl(X.data, [6], "float32")
  for i in grid(6):
    X_flattened[i] = ....
```

# Pass Checklist

Here are a list of TIR passes that can be impacted significantly when migrating from `Load/Store` to
`BufferLoad/BufferStore`.

- `StorageFlatten` / `FlattenBuffer`: These passes flatten buffer to physical dimensions. As
discussed above, they should create flattened buffer via `T.buffer_decl` while keeping `buffer_map`
unchanged (see the discussion in *Removing pre-flattened buffer* section). Any subsequent passes
that rewrite buffer, such as, `InjectDoubleBuffer`, `InjectVirtualThread` , should operate on
physical buffers and should not changing the number of buffer dimensions. `Allocate` after
flattening will reflect physical buffer dimensions. Alternatively, these passes could be made
simpler by moving them to occur before the buffer is flattened.  For example, implementing
`InjectDoubleBuffer` by changing the shape to `[2, *old_shape]`, and accessing using `[i%2,
*old_indices]`.  That would limit the size/stride handling to occur only during  buffer flattening.
- VectorizeLoop: This pass should rewrite buffer indices to `Ramp` for vectorized accesses, should
consider limiting vector index as the last dimension.
- `StorageRewrite`: This pass should be extended to handle N-D physical buffer.
- `VectorTypeRewriter` should also consider limiting vector index as the last dimension.
- `MakePackedAPI`: This pass adds additional parameters (variables) to `PrimFunc` according to the
FFI calling convention. These variables can no longer be used in `Load` directly. Buffer should be
declared and then `BufferLoad` should be used  to access values of these parameters.
- `LowerThreadAllreduce`: This pass is involved with a few buffer rewriting. Need to check buffer
declarations / accesses follow the new convention here.

# Conclusion and Key takeaways
- `T.buffer_decl` creates buffer alias, it is important to consider implications and use
`T.buffer_decl` properly. Passes that transform buffers should consider how to buffer alias.
Therefore we should be able to have a unified method called `T.buffer_decl` in both TIR and
TVMScript.
- There are several way for buffer definition, `T.buffer_decl, T.match_buffer, T.alloc_buffer`.
- `BufferLoad/BufferStore` can be generalized to allow `Ramp` as part of the index.
- `T.buffer_decl` is going to be used to declare flattened Buffer aliases, and
`preflattened_buffer_map` will be removed.
