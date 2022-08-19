- Feature Name: tvmscript-metaprogramming
- Start Date: 2022-06-16
- RFC PR: [apache/tvm-rfcs#79](https://github.com/apache/tvm-rfcs/pull/79)
- GitHub Issue: [apache/tvm#12442](https://github.com/apache/tvm/issues/12442)
- Co-Authors: Yaxing Cai ([**@cyx-6**](https://github.com/cyx-6), main implementation), Lite Ye
  ([**@yelite**](https://github.com/yelite)), Yong Wu
  ([**@yongwww**](https://github.com/yongwww)), Yuchen Jin
  ([**@YuchenJin**](https://github.com/YuchenJin)), Eric Lunderberg
  ([**@Lunderberg**](https://github.com/Lunderberg)), Masahiro Masuda
  ([**@masahi**](https://github.com/masahi)), Junru Shao
  ([**@junrushao1994**](https://github.com/junrushao1994), main designer)

# Summary
[summary]: #summary

This RFC proposes a new TVMScript parser infrastructure, supporting extensive
metaprogramming and syntactic sugars. The new infrastructure is IR-agnostic,
treating TIR just as one of dialects. Additionally, the new infrastructure will
provide better tooling around Python ecosystem (pylint, mypy, etc.).

# Motivation
[motivation]: #motivation

**What is TVMScript**. 
Check [Blitz Course to TensorIR](https://tvm.apache.org/docs/tutorial/tensor_ir_blitz_course.html) and
[TVMScript Unified Printer RFC](https://github.com/apache/tvm-rfcs/pull/74/files#diff-6965a40ad8df7618ae68e11c88f924542a506c74a931cc3011ae9f99989b5f51R20-R26)
for an introduction into TVMScript.

**What is metaprogramming.** In the context of TVMScript, metaprogramming means
a programmable way to control IR generation. For example, in
https://github.com/apache/tvm/pull/11097, a metaprogramming feature was added
to the TVMScript parser, allows users to programmably control the shapes of the
input buffers of a `PrimFunc`.

### Limitation of current design

The current parser lacks capability on generic metaprogramming that allows user
to have more control on IR construction. This makes it challenging to support
operators like NMS (non-maximum suppression, which is crucial to object
detection model). There is an implementation of NMS at
[python/tvm/topi/cuda/nms.py#L367-L386](https://github.com/apache/tvm/blob/d0650bad66d0ff89a01347537021bc442a98c223/python/tvm/topi/cuda/nms.py#L367-L386).
The implementation of NMS-like operators requires rank-polymorphism and the
ability to interleave host program with TVMScript, which is difficult to be
implemented under the current design.

TVMScript also needs reasonable support on Python tooling. Currently it doesn’t
play nicely with pylint and mypy. For example,
[test_meta_schedule_postproc_rewrite_tensorize.py](https://github.com/apache/tvm/blob/d0650bad66d0ff89a01347537021bc442a98c223/tests/python/unittest/test_meta_schedule_postproc_rewrite_tensorize.py)
has 100+ warnings from pylint within only 500 hundred lines of code. This
creates confusion to the user and leaves an impression that TVMScript isn’t a
mature product and not production-ready. Even though it’s something that can be
incrementally improved under the current design, we believe it’s easier to get
an ideal result if we have a design with the tooling support in mind.

The current design also lacks of unified approach for different IRs. At
[https://github.com/tlc-pack/relax/tree/relax/python/tvm/script/relax](https://github.com/tlc-pack/relax/tree/relax/python/tvm/script/relax),
a mature implementation of TVMScript parser is maintained for Relax. But it’s
hard to extend if we want to support more IRs for TVM unity.

To conclude, with this RFC, we want to:

1. Add more metaprogramming features to TVMScript, making it easier for TVM
   developers to write complicated operators.
2. Improve tooling and documentation of TVMScript, reducing the friction for an
   average machine learning practitioner to use TVMScript.
3. Modularize and infrastructuralize the TVMScript parser, lowering the cost to
   implement parser for new IR.


# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

## Metaprogramming features to support

### (F1) Template Metaprogramming

Users should be able to use variables from outer scope in the TVMScript
function/class. The parsed result should be identical to function/class with
the variable replaced by its value. For instance,

```python
@T.prim_func
def matmul(
  A: T.Buffer[(128, 128)],
) -> None:
  ...

def gen_matmul(n, m) -> None:
  @T.prim_func
  def f(A: T.Buffer[(n, m)]):
    ...
  return f

f = gen_matmul(n=128, m=128) # `f` should be identical to `matmul`
```

This is already partially supported by https://github.com/apache/tvm/pull/11097
for using `PrimExpr` captured by outer function. With the new parser, we want
to support this feature in more places and with more variable types.

### (F2) Rank-polymorphism

Users should be able to write a single function to handle different ranks of
input buffers (different numbers of dimensions). For example, user should be
able to write a generic function to do broadcast add,

```python
def broadcast_add(a, b, c):
  @T.prim_func
  def f(
    A: T.BufferFrom(a),
    B: T.BufferFrom(b),
    C: T.BufferFrom(c),
  ) -> None:
    for i, i_a, i_b in T.some_broadcast_method(A.shape, B.shape):
      with T.block():
        C[*i] = A[*i_a] + B[*i_b]

broadcast_add(
  a = Buffer((128, 1), "float32"),
  b = Buffer((1, 128), "float32"),
  c = Buffer((128, 128), "float32"),
)
```

### (F3) Sugar: TE Compute in TIR

Users should be able to replace boilerplate code with a function call, which’s
expanded to large chunk of code during parsing. For example, we may want to use
TE’s compute-like syntax to replace nested loop,

```python
@T.prim_func
def te_compute_sugar(
  A: T.Buffer[(128, 128)],
  B: T.Buffer[(128, 128)],
) -> None:
  ...
  C = T.compute((128, 128), lambda i, j: A[i, j] + B[i, j])
  ...

## expands to ====>

@T.prim_func
def te_compute_expanded(
  A: T.Buffer[(128, 128)],
  B: T.Buffer[(128, 128)],
) -> None:
  ...
  for i in range(128):
    for j in range(128):
      with T.block("..."):
        C[i, j] = A[i, j] + B[i, j]
  ...
```

### (F4) Interleave host program and TVMScript program to customize metaprogramming

As an escape hatch from writing code to be parsed by the TVMScript
parser, users should be able to write imperative code to construct IR nodes
directly and embed it inside regular TVMScript. Those code will be evaluated
by the Python interpreter when parsing. This gives users the ultimate tool when
TVMScript isn’t expressible enough for their use cases. For example, at
[python/tvm/topi/vision/nms.py#L380-L431](https://github.com/apache/tvm/blob/3cb4597ed48360e3f3d80161d1c03f833072d28e/python/tvm/topi/vision/nms.py#L380-L431),
there are blocks of repetitive code on computing the coordinates of the four
corners of bounding box. This can be simplified as:

```python
# Before, without IRBuilder interleaving
@T.prim_func
def nms(...):
  ...
  for i in range(batch_size):
    ...
    a_l = min(
      output[batch_idx, box_a_idx, box_start_idx],
      output[batch_idx, box_a_idx, box_start_idx + 2],
    )
    a_t = min(
      output[batch_idx, box_a_idx, box_start_idx + 1],
      output[batch_idx, box_a_idx, box_start_idx + 3],
    )
    a_r = max(
      output[batch_idx, box_a_idx, box_start_idx],
      output[batch_idx, box_a_idx, box_start_idx + 2],
    )
    a_b = max(
      output[batch_idx, box_a_idx, box_start_idx + 1],
      output[batch_idx, box_a_idx, box_start_idx + 3],
    )
		...
    for k in range(j):
      check_iou = ...
			...
      if check_iou > 0:
        # b_l: left, b_t: top, b_r: right, b_b: bottom
        b_l = min(
          output[batch_idx, box_b_idx, box_start_idx],
          output[batch_idx, box_b_idx, box_start_idx + 2],
        )
        b_t = min(
          output[batch_idx, box_b_idx, box_start_idx + 1],
          output[batch_idx, box_b_idx, box_start_idx + 3],
        )
        b_r = max(
          output[batch_idx, box_b_idx, box_start_idx],
          output[batch_idx, box_b_idx, box_start_idx + 2],
        )
        b_b = max(
          output[batch_idx, box_b_idx, box_start_idx + 1],
          output[batch_idx, box_b_idx, box_start_idx + 3],
        )
        ...

# With IRBuilder interleaving:

from tvm.script import tir as T

def get_box_coordinates(output, batch_idx, box_idx, box_start_idx):
  """a method executed by python interpreter"""
  box_l = T.min(
    output[batch_idx, box_idx, box_start_idx],
    output[batch_idx, box_idx, box_start_idx + 2],
	) # type(box_l) is PrimExpr
  ... # Repeat for other coordinates
  return box_l, box_t, box_r, box_b

@T.prim_func(capture=[get_box_coordinates])
def nms(...):
  ...
  for i in range(batch_size):
    ...
    a_l, a_t, a_r, a_b = get_box_coordinates(output, batch_idx, box_a_idx, box_start_idx)
    ...
    for k in range(j):
      check_iou = ...
      ...
      if check_iou > 0:
        b_l, b_t, b_r, b_b = get_box_coordinates(output, batch_idx, box_b_idx, box_start_idx)
        ...
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

## IRBuilder as Core

As the foundation of IR construction, we will provide a set of APIs called
IRBuilder to let user construct IR imperatively. IRBuilder will be used by the
parser, as well as by users directly as described in the feature F4. IRBuilder
allows user to write code in a style that’s similar to TVMScript, while it’s
being executed as host program. For example, 

```python
from tvm.script.builder import Builder, def_, def_many
from tvm.script import tir as T

with Builder() as b:
  with T.prim_func():
    T.func_name("main")
    buffer_a = T.Buffer((128, 128, 128), "float32")
    buffer_b = T.Buffer((128, 128, 128), "float32")
    arg_a = T.arg("A", buffer_a)
    arg_b = T.arg("B", buffer_b)
    with T.grid(128, 128, 128) as (i, j, k):
      def_many(["i", "j", "k"], [i, j, k])
      with T.block(name="block"):
        vi = def_("vi", T.axis.spatial(128, i))
        vj = def_("vj", T.axis.spatial(128, j))
        vk = def_("vk", T.axis.reduce(128, k))

f = b.get() # f is a PrimFunc
```

produces equivalent result as 

```python
@T.prim_func
def main(
  A: T.Buffer(shape=(128, 128, 128), dtype="float32"),
  B: T.Buffer(shape=(128, 128, 128), dtype="float32"),
) -> None:
  for i, j, k in T.grid(128, 128, 128):
      with T.block("block"):
          vi = T.axis.S(128, i)
          vj = T.axis.S(128, j)
          vk = T.axis.R(128, k)
```

As shown in the example above, user doesn't need to pass the builder `b` to
subsequent calls to IRBuilder API. The current builder state is maintained in a
threadlocal store to improve the ergonomics of IRBuilder API by avoiding
passing the builder state explicitly.

The implementation of IRBuilder will be in C++ so that it can be used in an
environment without Python. Python binding will be created to expose IRBuilder
to TVMScript parser.

With the separation between IRBuilder and parser, most implementation and
documentation can be reused between TVMScript and IR definition. For example,
most of operators are simply imported into the IRBuilder package, like
```python
from tvm.tir import sin, cos
```
so the documentation and type signatures only need to be written once, and the
APIs are guaranteed to be consistent.

## Parse-time evaluation

TVMScript Parser can be considered as a thin layer built above the IRBuilder
API. The parser transforms input AST into a sequence of calls to IRBuilder API,
by evaluating small fragments of code as it visits AST. The IRBuilder is
responsible for building the actual IR graph. All metaprogramming features we
discussed above (F1 through F4) can be implemented through this parse-time
evaluation in a consistent manner. Using the same `gen_matmul` example from F1,

```python
def gen_matmul(n, m) -> None:
  @T.prim_func
  def f(A: T.Buffer[(n, m)]):
    ...
  return f
```

What parser does here is to:

1. Collect the environment inside `gen_matmul`, getting a dictionary
    1. All primitive types (`int`, `str`, `float` and `None`) will be captured
       automatically, while advanced types (like function) needs to be
       explicitly declared in the decorator to be captured (for example,
       `@T.prim_func(capture=[get_box_coordinates])`)
2. Call the corresponding function from IRBuilder, as the parser visits the AST
   of function `f`.
    1. When visiting the function argument, call `eval` on its type annotation, 
       with the environment captured in the first step.
    2. `T.Buffer[(n, m)]` gets evaluated to a value with type `tir.Buffer`.
    3. Call the IRBuilder API `T.arg("A", buffer)` to add an arg to the function
       that’s being constructed

Another example,

```python
for *i in T.grid(*A.shape):
  ...
```

The parser will:

1. Evaluate the expression `T.grid(*A.shape)` by the step described above.
   `T.grid` returns a value that is nearly equivalent to `List[Var]`. 
2. Call `exec` on a specially constructed statement `*i = __tvm_rhs_var__`,
   with `locals` that maps `__tvm_rhs_var__` to the value evaluated in step 1.
3. Collect the value of `i` from the `locals` dictionary
4. Call the IRBuilder API `def_many(["i"], [i])` 

As mentioned above, all metaprogramming features (F1 through F4) can be
implemented through this parse-time evaluation. It's straightforward to see how
F1 and F2 are implemented by parse-time evaluation, but it might be harder to
grasp the idea behind F3 and F4. 

For F3 (TE Compute in TIR),
```python
C = T.compute((128, 128), lambda i, j: A[i, j] + B[i, j])
# build a similar graph as
for i in range(128):
  for j in range(128):
    with T.block("..."):
      C[i, j] = A[i, j] + B[i, j]
```
`T.compute` is provided in IRBuilder API and will construct all IR nodes (For,
BlockRealize, Block and BufferStore). `T.compute` calls the lambda function to
get the rhs of BufferStore. Then `T.compute` returns a `Buffer` node that
represents `C`. The parser handles the assignment by assigning `"C"` to the
`name_hint` of the returned buffer (by calling IRBuilder API `def_("C", C)`),
and put it into the internal variable table (which is used to resolve variable
when evaluating subsequent statements and expressions).

For F4 (Interleave host program and TVMScript program),
```python
a_l, a_t, a_r, a_b = get_box_coordinates(output, batch_idx, box_a_idx, box_start_idx)
```
The call to the `get_box_coordinates` function is evaluated when parser is visiting the 
assign statement. The parser calls IRBuilder `def_many(["a_l", "a_t", "a_r", "a_b"], <returned_tuple>)`
and put them into the internal variable table.

Note that we will not place extra restriction on the signature of user-provided
function (`get_box_coordinates` in this example). More precisely, the function
argument types can be anything because parser is able to capture outter
variables thus bringing variables with arbitrary type into the scope. The restriction on
returned type depends on the language spec of the target IR. For example, in
TIR user can write `A[i] = ...` to represent `BufferStore`, where the rhs is a
`PrimExpr`, then the user can provide a custom function
`compute_magic_number(index: PrimExpr) -> PrimExpr` to use on the rhs as `A[i]
= compute_magic_number(i)`.

By running `eval` and `exec` provided by the Python interpreter, we can implement
language features which are difficult to implement manually, and also make sure
TVMScript has the same semantics on expression compared to regular Python code.

## Parser Registration

The logic of how to process a particular Python syntax node is registered
through decorator. For example,

```python
@dispatch.register(token="tir", type_name="For")
def visit_for(self: Parser, node: doc.For) -> None:
    ...
```

handles the transformation of Python `For` node. The `token="tir"` in the
decorator means that the handler is for TIR. `self: Parser` has all the
infrastructural API and maintains a stack of `token` to determine which
function to dispatch to. This makes embedding different IR possible (for
example, embedding TIR in Relax). The folder structure will look like

```
python/tvm/script/
└── parser
    ├── core
    │   └── ... # Parser infrastructure
    ├── tir # TIR dialect
    │   └── ...
    └── relax # Hypothetical Relax dialect (not part of our RFC)
        └── ...
```

## Usage of `eval` and `exec`

The parser uses `eval` and `exec` in the following places:
- It calls `eval` to evaluate fragment of expressions
- It calls `exec` to evaluate different kinds of assignment statement, like `first, *rest, last = T.grid(...)`

The usage of `eval` and `exec` is necessary to our implementation. TVMScript
allows users to construct IR graph in TVM declaratively as if they were writing
Python code, lowering the barrier of using TVM to do low-level customization.
However it is still restrictive and does not allow the usage of many Python
features to build abstractions for user's code. All features proposed in this
RFC can be seen as an effort to narrow this gap, thus they are designed to
follow the Python semantics. On the implementation side, the most robust
approach is to leverage the Python interpreter itself to facilitate those
features, rather than write our own version of restricted Python interpreter.
And `eval` and `exec` are the most suitable choices to achieve this. Other
mechanism, like multiprocess + IPC and subinterpreter, either lacks an easy
path to exchange Python objects, or requires dependency on the C API of CPython.

In our use cases, `eval` and `exec` do not create additional
security risk for users. All inputs to `eval` come from user's code
directly, without modification. Our usage of `exec` only executes a
specific form of code, 
```python
<lhs> = __tvm_rhs_var__
```
where `<lhs>` is the tokens from the left hand side of assign statement,
directly from user's code. The situation is very different from cases that make
`eval` and `exec` infamous, where they are used to evaluate untrusted
input from end users, typically in a web server. Furthermore, we require users
to explicitly capture variables. Therefore, The evaluation will only involve
objects from `tvm`, Python builtins and objects explicitly captured by users.
This rules out the possibility that an external function is called by parser
without user's acknowledgement. If the malicious actor wants to exploit the
system through the `eval` or `exec` in TVMScript parser, they must first get
another RCE (remote code execution) vulnerability in the Python runtime to
modify the code in runtime, which makes such exploit useless (one needs to
first have an RCE to exploit another RCE with the same exposure). 


# Drawbacks
[drawbacks]: #drawbacks

N/A

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

N/A

# Prior art
[prior-art]: #prior-art

### Prior works in TVM
- Hybrid Script: [https://tvm.apache.org/docs/reference/langref/hybrid_script.html](https://tvm.apache.org/docs/reference/langref/hybrid_script.html)
- RFC for TVMScript: [https://discuss.tvm.apache.org/t/rfc-hybrid-script-support-for-tir/7516](https://discuss.tvm.apache.org/t/rfc-hybrid-script-support-for-tir/7516)

### Other libraries
#### Taichi 
[https://www.taichi-lang.org](https://www.taichi-lang.org/)

Taichi has a very similar metaprogramming model ([link to doc](https://docs.taichi.graphics/docs/master/meta)) as we presented in this RFC.
The biggest difference is that Taichi requires `ti.static` to be wrapped around everything that needs to be evaluated in compile time. It also 
has advanced features like loop unrolling, compile time branching and compile-time recursion. 

In TVMScript parser, it does not need special marker to denote compile-time
evaluation. Expressions are consistently evaluated in compile time (evaluated
to IR node like PrimExpr, rather than the concrete value like a float matrix.), 
thanks to the separation of IRBuilder and parser. Features like loop
unrolling can be implemented in the IRBuilder layer per target IR. This keeps
the core parser as minimal as possible.


#### Triton 
[http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)

Triton uses the meta-parameters to generalize kernels (for example, 
[tutorials/03-matrix-multiplication.html](https://triton-lang.org/master/getting-started/tutorials/03-matrix-multiplication.html)).
Meta parameters are placed together with real parameters, but with type
annotation `tl.constexpr` to differentiate. This method slightly deviates from
regular Python semantics, as users will intuitively expect to pass them
together with real parameters when calling the kernel.

In TVMScript, one of the design principle is to narrow the gap between
TVMScript and regular Python code. TVMScript should not surprise users with
syntax or language features that deviate from Python. All features proposed in
this RFC are designed to strictly follow the semantics of Python and aimed to
be intuitive to Python users.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

N/A

# Future possibilities
[future-possibilities]: #future-possibilities

N/A
