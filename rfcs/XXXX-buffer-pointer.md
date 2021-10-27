- Feature Name: (fill me in with a unique identifier, `my_awesome_feature`)
- Start Date: (fill me in with today's date, YYYY-MM-DD)
- RFC PR: [apache/tvm-rfcs#0000](https://github.com/apache/tvm-rfcs/pull/0000)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary
[summary]: #summary

A location being access in a buffer is represented as a
`BufferPointer` object, which holds a reference to the buffer being
accessed, and an array of indices to specify the location.
`BufferLoad` and `BufferStore` objects each hold a `BufferPointer`
object to specify where they operate.

# Motivation
[motivation]: #motivation

`BufferLoad` and `BufferStore` both act on a location within a buffer,
and must know where read/write their respective values.  However, many
transformations are unconcerned with whether an access is a read or a
write.  (e.g. `tir.transform.LowerMatchBuffer`, which rewrites access
of a matched buffers to instead be direct access of the backing
buffer.)  By having a `BufferPointer` object to represent a pointer
into a buffer's memory, these transformations can be done without
repeated code.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

The `BufferPointer` object contains a reference to the buffer that it
acts upon, and the indices at which the access is being performed.

`BufferPointer` also provides a utility function
`BufferPointerNode::value_dtype()`, which returns the expected
datatype at the specified location.  This will typically be the same
as the buffer's datatype, but may have a different number of lanes.
For example, a `BufferPointer` whose buffer's datatype is `float16`,
and whose index is `Ramp(pos, stride=1, lanes=4)` for vectorized
access will return `float16*4` for `value_dtype()`.

Previously, `BufferLoad` and `BufferStore` held references to the
buffer and indices directly.  To migrate these codes, replace
references to `BufferX::buffer` with `BufferX::pointer::buffer` and
replace references to `BufferX::indices` with
`BufferX::pointer::indices`.


# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

This is the technical portion of the RFC. Explain the design in sufficient detail that:

- Its interaction with other features is clear.
- It is reasonably clear how the feature would be implemented.
- Corner cases are dissected by example.

The section should return to the examples given in the previous section, 
and explain more fully how the detailed proposal makes those examples work.

# Drawbacks
[drawbacks]: #drawbacks

Requires an additional indirection to access the buffer and pointer,
so transformations that must distinguish between reads and writes
become more verbose.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

The `BufferLoad::pointer` and `BufferStore::pointer` could be generic
`PrimExpr`, instead of being `BufferPointer` objects.  This would
require the datatype to be a handle, with an additional parameter to
indicate what is being stored.  However, this 

Currently, the `BufferLoad::pointer` and `BufferStore::pointer`
objects are visited by `ExprMutator` and `StmtMutator`, but are
required to be `BufferPointer` objects.  This is implemented in
type-checking in `ExprMutator::VisitExpr_(BufferLoad*)` and
`StmtMutator::VisitStmt_(BufferStore*)`, but it isn't apparent at the
callsite that the returned `PrimExpr` must be a `BufferPointer`.

Prior to this RFC's implementation, all transformations that modify a
buffer must have near-equivalent mutators for both the `BufferLoad`
and `BufferStore` nodes.

# Prior art
[prior-art]: #prior-art

This follows a C-style convention.  Given an array `int x[100]`, the
location `int* ptr = (x+50)` represents the location of element 50 in
array `x`.  A buffer load is then represented as `*ptr`, and buffer
store is represented as `*ptr = val;`.

This also maps onto Vulkan semantics, where a `OpAccessChain`
instruction is used to generate a pointer into an array, which can
then be used with either `OpLoad` or `OpStore`.

# Unresolved questions
[unresolved-questions]: #unresolved-questions



# Future possibilities
[future-possibilities]: #future-possibilities

The `BufferPointer` could represent a pointer to a specific element in
the generated C code.  This can be used for generating pointers to
pass into hardware-specific intrinsics, rather than using
`BufferNode::elem_offset` or the built-in `tvm_access_ptr`.
