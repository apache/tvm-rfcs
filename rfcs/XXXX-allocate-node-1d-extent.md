- Feature Name: AllocateNode, 1-d extent
- Start Date: 2021-10-06
- RFC PR: [apache/tvm-rfcs#0000](https://github.com/apache/tvm-rfcs/pull/0000)
- GitHub PR: [apache/tvm#9194](https://github.com/apache/tvm/pull/9194)
- Related RFCs: [apache/tvm-rfcs#0039](https://github.com/apache/tvm-rfcs/pull/0039)

# Summary
[summary]: #summary

Replace `Array<PrimExpr> AllocateNode::extents`, which gives the
extent of each logical dimension of a tensor, with `PrimExpr
AllocateNode::extent`, which gives the 1-d size of a flat memory
buffer.

# Motivation
[motivation]: #motivation

This RFC is a subset of
[RFC#0039](https://github.com/apache/tvm-rfcs/pull/0039), creating a
separation between the physical layout of a buffer as it exists in
memory, and the logical layout of a buffer as it exists in a tensor's
compute definition.  This change removes information from
`AllocateNode` that describes the logical layout, leaving only
information relevant to the physical layout.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

`AllocateNode::extent` gives the size of a flat memory buffer to be
allocated at runtime.  This is used either by the target-specific code
generator for allocations internal to a compute kernel, or to generate
calls to `DeviceAPI::AllocWorkspace` for allocations initiated in the
host and later passed to a compute kernel.  In either case,
`AllocateNode::extent` is used to specify the size of the array, in
terms of the underlying datatype.



# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation


`AllocateNode::extent` is defined based on the realized extents of a
buffer, as stored in `BufferRealizeNode::bounds`.  When
`BufferRealizeNode` is lowered to `AllocateNode`, which occurs either
in `tir.transform.StorageFlatten` for schedules defined in TE or in
`tir.transform.FlattenBuffer` for schedules defined in TensorIR, the
product of the extents along each dimension of the logical shape is
the total number of elements in the buffer stored in
`AllocateNode::extent`.

Previously, `AllocateNode` contained an `Array<PrimExpr> extents`,
specifying the realized extent of each dimension of the tensor backed
by the allocated buffer.  This required optimization passes that act
on an allocated buffer (e.g. `tir.transform.InjectDoubleBuffer`) to be
aware of the tensor shape, even though it only needs information about
the physical size of a buffer.  In addition, any code that allocated
memory needed to re-compute the total array size from the extent of
each dimension, leading to unnecessary code duplication.

# Drawbacks
[drawbacks]: #drawbacks

- Optimization passes that require knowing the dimensionality of a
  buffer or the size along each dimension cannot occur after the
  `StorageFlatten` or `FlattenBuffer` has removed this information.
  These passes must be placed prior to the generation of `AllocateNode`,
  and must instead act on `BufferRealizeNode`, `BufferStoreNode`, and
  `BufferLoadNode`.

- This is a breaking change in the TIR semantics, and would break some
  user-defined schedules.


# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

- This is intended to be followed by another change impacting
  `AllocateNode`, replacing the 1-d `PrimExpr AllocateNode::extent`
  with the N-d `Array<PrimExpr> AllocateNode::shape`, and could
  instead be done as a single change.
  
  `AllocateNode::shape` will represent a N-d physical layout, where
  `AllocateNode::extents` represented a N-d logical layout.  Use of
  `PrimExpr AllocateNode::extent` a a transition between the two
  should help avoid any bugs resulting from accidental use of these
  two types.  Since both the previous `extents` and the intended
  `shape` are of type `Array<PrimExpr>`, making a single change from
  `extents` to `shape` could appear as a simple rename, rather than as
  a change in semantics.

# Prior art
[prior-art]: #prior-art

No relevant prior art known.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

- What level of impact do we expect to have on user-defined schedules?

  This would impact schedules that explicitly define IR to compute a
  value, where that IR contains `AllocateNode`.  The level of impact
  would depend on the frequency of this use case.

# Future possibilities
[future-possibilities]: #future-possibilities

This implements a portion of
[apache/tvm-rfcs#0039](https://github.com/apache/tvm-rfcs/pull/0039),
making a distinction between physical and logical layout.
