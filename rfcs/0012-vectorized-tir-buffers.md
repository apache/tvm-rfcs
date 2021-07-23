- Feature Name: vectorized-tir-buffers
- Start Date: 2021-07-22
- RFC PR: [apache/tvm-rfcs#0012](https://github.com/apache/tvm-rfcs/pull/0012)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary
[summary]: #summary

All pointers and buffer allocations in TIR are strongly typed, and may
point to either scalar or vectorized elements (e.g. either `float32*` or
`float32x2*`).  Accessing a pointer or buffer with a single-lane index
results in a value with the same element type as the pointer/buffer.
Accessing a pointer or buffer with a multi-lane index results in a value
of type `element_type.with_lanes(element_type.lanes() * index.lanes()`.

Casts between pointer types with the same base type (e.g. from `float32*`
to `float32x2*`) or between different base types (e.g. from `int32*` to
`float16*`) may be present in the TIR graph.  Optimization passes that can
introduce such casts, such as `StorageRewrite`, should only do so if the
target supports these casts.  Codegen should not introduce additional
pointer type casts beyond those specified in the TIR graph.

Vectorized loads/stores should be specified in the TIR graph as access into
a pointer/buffer whose element type is vectorized.  Codegen should assume
that any multi-lane indices have already been vectorized by the TIR
optimization passes if possible, and should not apply vectorization beyond
what is specified in the TIR graph.

# Motivation
[motivation]: #motivation

Following [TVM PR#8528](https://github.com/apache/tvm/pull/8528), which
resolved an issue with array access in the Vulkan runtime, led to and/or
exposed some inconsistencies in the TIR semantics for buffer indices.
Vulkan/SPIR-V require all arrays to be typed, and doesn't allow type casts
that would be permissible in C code, such as casting between `float32*` and
`float32x2*`.  As a result, any vectorized load/store operations must act
on an array whose elements are vectorized types.  However, this is
inconsistent with how stores/loads are expressed in TIR passed to other
codegens.

Currently, many places in IR
(e.g. [`Load::Load`](https://github.com/apache/tvm/blob/07243a89/src/tir/ir/expr.cc#L621)
checking the number of output lanes) and codegen (e.g. [`CodeGenC`'s
`LoadNode`](https://github.com/apache/tvm/blob/07243a89/src/target/source/codegen_c.cc#L719)
loop over output lanes rather than element/index lanes) implicitly assume
that all array elements have `lanes == 1`.  This is inconsistent with the
type-checking requirements of SPIR-V, which requires vectorized load/store
to occur on arrays with multi-lane element type.  The TIR semantics should
be expanded to cover both use cases, and then be interpreted uniformly
across all runtimes.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

- A "vectorized load" or "vectorized store" refers to memory copies that
  copy many objects with a single underlying instruction.

- A "scalar type" is any TVM Datatype with `lanes == 1`.  These represent a
  single value.

- A "vectorized type" is any TVM Datatype with `lanes > 1`.  These
  represent several values that can be acted on simultaneously.

By having all vectorized types be explicitly specified in the TIR graph,
the logic to identify vectorized access can be moved to an optimization
pass, and does not need to be repeated across all runtimes.  This will
simplify implementation of new codegen targets, as there is less logic that
needs to be included in them.

This also allows for additional type-checking to be performed during he
codegen.  Prior to this RFC, if the datatype associated with a store/load
doesn't match the type stored, the codegen is allowed to add a pointer
cast.  After this RFC, all pointer casts must be explicitly specified, and
any type mismatch is an error.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

The following items would need to be implemented for this RFC.

- Support for `CastNode` to indicate a pointer cast.  The dtype of
  `Cast(dtype, ptr_value)` should be `kHandle`, with the type annotation
  of `PointerType(PrimType(dtype))`.

- New optimization pass to identify use of `RampNode` with stride of 1, and
  to rewrite as a pointer-cast followed by access with a scalar index.

- Updates to `StoreNode`/`LoadNode` visitors in all C-based codegens

  - Removal of checks for a `RampNode` index.  If any exist after
    optimization pass, assume that it is deliberate for non-vectorized
    access.
    
  - Fallback explicit loop should be a loop over the lanes of the index and
    the lanes of the element type.
    
- Checks in TIR `Store::Store` and `Load::Load` include identifying the
  number of lanes in the array elements, asserting that `value_lanes =
  element_lanes*index_lanes`.

# Drawbacks
[drawbacks]: #drawbacks

This is an explicit change to the semantics of the TIR graph, which may
result in unexpected breakage.  Previously, all buffers were assumed to
have a scalar elements, and vectorization was done during the codegen step.
Allowing buffers to have vectorized elements may 

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

Possible options for how buffer store/loads should be specified in the TIR
graph, what semantics they mean, and how the codegen should interpret it
are listed below.
  
1. Buffer/pointer types

   a. Buffers are untyped, and have no type until/unless cast to
      an appropriate type (`memset` semantics).
   
   b. Buffers are typed, and the element type must be scalar
      (`int arr[size];` semantics).
   
   c. Buffers are typed, and the element type may be either scalar or
      vectorized (`int arr[lanes][size];` semantics).
   
2. Casting of pointer types

   a. The codegen must cast all pointer types to the type specified by
      `StoreNode` and `LoadNode` (i.e. `store_node->value.dtype()` and
      `load_node.dtype()`), regardless of the type of the buffer.  This
      includes casting to a vectorized type with `lanes > 1`.
      
   b. The codegen must cast all pointer types to type specified by
      `StoreNode` and `LoadNode`, but with the number of lanes set
      to 1. (i.e. `store_node->value.dtype().element_of()` and
      `load_node.dtype().element_of()`), regardless of the type of the
      buffer.
      
   c. Pointer types may be cast from one type to another, but it must be
      explicitly specified in the TIR graph.  The dtype `Cast(dtype,
      ptr_value)` should be `kHandle`, with the type annotation set to
      `PointerType(PrimType(dtype))`.  Optimization passes that may
      introduce pointer casts should only do so if the target supports
      them.
      
   d. Pointer types may not be cast from one type to another.  `CastNode`
      applies only to value types, and not to pointer types.
   
3. Index values

    a. Indices/offsets are specified as an integer number of bytes.

    b. Indices/offsets are specified as an integer number of array
       elements.
   
4. Index lanes, result type

   a. Indices must always have exactly one lane.  The type of the value
      accessed is the same as the buffer's element type.
      
   b. Indices may have more than one lane.  The type of the value accessed
      is the buffer's element type, but with the same number of lanes as
      the index. (e.g. When accessing a buffer of type `float16x4*` with an
      index of type `int32x4*`, the result is type `float16x4`.)  This is
      the current behavior when using a `RampNode` with stride of 1 as an
      index.
      
   c. Indices may have more than one lane.  The type of the value accessed
      is the buffer's element type, but with the number of lanes equal to
      the product of the number of lanes in the index and the number of
      lanes in the buffer's element type.  (e.g. When accessing a buffer of
      type `float16x4*` with an index of `int32x4*`, the result is type
      `float16x16`)

Prior to this RFC, `CodeGenC` and subclasses assume that buffers have a
scalar element type (option 1b), indices are specified in array elements
(option 2b), that indices may have more than one lane (option 3b), and that
the codegen may cast pointer types as needed to produce the requested
output type (option 4b).  To minimize the amount of code change needed,
these options should remain the same unless there is a reason to change
them.

## Typing

Vectorized stores/loads in SPIR-V requires stores/loads to occur on an
array with vectorized types.  Therefore, option 1c is preferred.

## Pointer casting

The stronger typing required by SPIR-V shaders prevents use of option 4b.
A pointer can only be dereferenced to its exact element type, and cannot
even have a cast between scalar and vectorized.  However, forbidding
pointer casts altogether would prevent possible optimizations in runtimes
that support them, so option 4d shouldn't be used either.  Option 4c,
allowing pointer type casts that are explicitly specified TIR, would allow
runtimes that support pointer casts to take advantage of them, while
avoiding them in runtimes that don't.  This would also add a single
optimization pass where vectorized stores/loads are enabled, rather than
repeating similar logic checking for ramp nodes in each codegen.

## Index values

A byte-offset (option 3a) would make sense for compatibility with option 1a
(untyped arrays), but that is not the preferred case.  Since it is neither
the current convention, nor is there an obvious reason to change it,
indices will continue to be specified in terms of array elements (option
3b).

## Index lanes, result type

Prior to this RFC, codegen that supports vectorized load/store have special
handling of RampNodes as indices
(e.g. [`CodeGenC`](https://github.com/apache/tvm/blob/07243a89/src/target/source/codegen_c.cc#L712),
[CodeGenSPIRV](https://github.com/apache/tvm/blob/07243a89/src/target/spirv/codegen_spirv.cc#L446)).
In the case of `CodeGenC`, this can apply on arrays with either scalar
elements (1-lane elements, `N`-lane indices, result has `N` lanes) or
vectorized elements (`M`-lane elements, `N`-lane indices, result has `N`
lanes), by applying pointer casts.  In the case of`CodeGenSPIRV`, these can
only apply on arrays with vectorized elements.  The proposed type of buffer
access are summarized in the table below.


| Current CUDA behavior       | Scalar Index      | Vector Index (Ramp with stride=1, `N` lanes) | Vector Index (other, `N` lanes) |
|---                          |---                |---                                           |---                              |
| Scalar Elements             | Scalar            | Vector, `N` lanes                            | Vector, `N` lanes               |
| Vector Elements (`M` lanes) | Not supported     | Vector, `N` lanes                            | Not supported                   |


| Proposed semantics          | Scalar Index      | Vector Index (Ramp with stride=1, `N` lanes) | Vector Index (other, `N` lanes) |
|---                          |---                |---                                           |---                              |
| Scalar Elements             | Scalar            | Vector, `N` lanes                            | Vector, `N` lanes               |
| Vector Elements (`M` lanes) | Vector, `M` lanes | Vector, `N*M` lanes                          | Vector, `N*M lanes`             |

This gives consistent semantics, that all access of `M`-lane with and
`N`-lane index yields a value with `N*M` lanes (option 3c).  This would be
coupled with an optimization pass to rewrite `RampNode` with stride=1 into a
pointer cast followed by access with a scalar index, so the overall
behavior of the `RampNode` remains the same.

# Prior art
[prior-art]: #prior-art

Unknown, suggestions would be appreciated.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

- Is this new set of semantics internally consistent?

- Are there incompatibilities between these semantics and other operators?

- Are there issues that would arise from exposing functionality currently
  in the codegen to the TIR graph?
  
- Where else in TIR and codegen are likely to break as a result of allowing
  vectorized elements in an array?

# Future possibilities
[future-possibilities]: #future-possibilities

Having explicit pointer casts would also simplify the handling of boolean
arrays.  Prior to this RFC, several locations identify buffers that point
to boolean tensors, and convert to an `int8` backing array
(e.g. [`Buffer::vload`](https://github.com/apache/tvm/blob/07243a89/src/tir/ir/buffer.cc#L299),
[`Buffer::vstore`](https://github.com/apache/tvm/blob/07243a89/src/tir/ir/buffer.cc#L314),
[`CodeGenSPIRV`](https://github.com/apache/tvm/blob/07243a89/src/target/spirv/codegen_spirv.cc#L60)).
As some runtimes do not support the use of `int8`, pulling this logic into
an optimization pass will simplify this future change.
