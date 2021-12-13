- Feature Name: Buffer Physical Layout
- Authors: Eric Lunderberg (@Lunderberg), Wuwei Lin (@vinx13)
- Start Date: 2021-10-05
- RFC PR: [apache/tvm-rfcs#0039](https://github.com/apache/tvm-rfcs/pull/0039)
- GitHub Issue: Not Yet Written

# Summary
[summary]: #summary

This RFC introduces layout transformations that can be applied to a
buffer during the lowering process.  These transformations will be
part of the schedule, allowing the same compute definition to be used
across multiple different layouts.  These transformations can produce
either flat memory buffers or multi-dimensional memory buffers to be
exposed to the low-level code generators.

# Motivation
[motivation]: #motivation

Currently, TVM assumes that all buffers can be treated as flat memory.
That is, while a rank-N tensor requires N values to describe its shape
and N indices to identify a particular value within it, the underlying
buffer allocated by the low-level codegen has a single value defining
the size, and access into that buffer is done using a single index.
This assumptions holds for most cases, such as a CPU accessing RAM,
but doesn't hold in all cases.  For example, texture memory on a GPU
requires two indices to access.  These are currently handled on a
case-by-case basis, such as using `tvm::tir::builtin::texture2d_store`
in a `CallNode`.

In addition, computations that are semantically identical (e.g. 2-d
convolution) require independent compute definitions and schedules
(e.g. `conv2d_nchw` and `conv2d_hwcn`) based on the format of the data
accepted as input.

This RFC introduces a mechanism to specify transformations to be
applied to the layout of buffers in memory, along with a unified
method of presenting multiple indices to the low-level code
generators.  This will allow for target-specific handling of non-flat
memory, and will allow for code re-use across compute definitions that
differ only in memory layout.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

A buffer is represented by a `tvm::tir::Buffer` object, and has some
shape associated with it.  This shape is initially defined from the
buffer's shape in the compute definition.  Buffers can either be
allocated within a `tvm::tir::PrimFunc` using a `tvm::tir::Allocate`
node, or can be passed in as parameters to a `PrimFunc`.  Buffer
access is done using `tvm::tir::BufferLoad` and
`tvm::tir::BufferStore` for reads and writes, respectively.

When a TIR graph is passed into the low-level code generator
`tvm::codegen::Build`, the rank of each buffer must be supported by
the target code generator.  Typically, this will mean generating a
single index representing access into flat memory.  Some code
generators may attach alternative semantics for `rank>1`
buffers (e.g. rank-2 buffers to represent texture memory on OpenCL).
A low-level code generator should check the rank of the buffers it is
acting on, and give a diagnostic error for unsupported rank.

To define the layout transformation in a TE schedule, use the
`transform_layout` method of a schedule, as shown below.  The
arguments to `transform_layout` is a function that accepts a list of
`tvm.tir.Var` representing a logical index, and outputs a list of
`tvm.tir.PrimExpr` giving a corresponding physical index.  If
`transform_layout` isn't called, then no additional layout
transformations are applied.

For example, below defines the reordering from NHWC logical layout to
NCHWc physical layout.  Similar to `cache_read` and `cache_write`, the
`transform_layout` method introduces a new stage in the schedule.

```python
# Compute definition, written in terms of NHWC logical axes
B = te.compute(A.shape, lambda n,h,w,c: A[n,h,w,c])
s = te.create_schedule(B.op)

def nhwc_to_nchwc(logical_axes):
    n,h,w,c = logical_axes
    return [n, c//4, h, w, c%4]

B_nchwc = s[B].transform_layout(nhwc_to_nchwc)

# Compute definition that would produce an equivalent physical layout
B_equivalent = te.compute(
    [A.shape[0], A.shape[3]//4, A.shape[1], A.shape[2], 4],
    lambda n, c_outer, h, w, c_inner: A[n, h, w, 4*c_outer+c_inner],
)
```

By default, after all explicitly specified layout transformations are
applied, all axes are flattened to a single axis by following a
row-major traversal.  This produces a 1-d buffer, which corresponds to
flat memory.  To produce `rank>1` buffers in the physical layout,
insert `te.AXIS_SEPARATOR` into the axis list return by the physical
layout function.  These define groups of axes, where each group is
combined into a single physical axis.

```python
B = te.compute(shape=(M,N,P,Q), ...)
s = te.create_schedule(B.op)

# Default, produces a 1-d allocation with shape (M*N*P*Q,)
s[B].transform_layout(lambda m,n,p,q: [m,n,p,q])

# One separator, produces a 2-d allocation with shape (M*N, P*Q).
s[B].transform_layout(lambda m,n,p,q: [m, n, te.AXIS_SEPARATOR, p, q])

# Two separators, produces a 3-d allocation with shape (M, N*P, Q).
s[B].transform_layout(lambda m,n,p,q: [m, te.AXIS_SEPARATOR, n, p, te.AXIS_SEPARATOR, q])

# Can be used along with reorders and splits.
s[B].transform_layout(lambda m,n,p,q: [m, q//4, n, te.AXIS_SEPARATOR, p, q%4])
```


The `te.AXIS_SEPARATOR` object exists only within the API interface,
and is not part of the representation of the layout transformation
within the generated TIR graph.  Instead, the TIR graph will contain
an integer list of axis separators, to be used when flattening buffers
to device-supported rank in the `StorageFlatten` or `FlattenBuffer`
passes.


# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

Transformation of a buffer is represented by the attribute
`"buffer_layout_transformations"` in the `PrimFunc` attributes.  This
is a map whose keys are buffer var to be reshaped, and whose values
are the transformations to be applied.  Many of the utilities
needed for this transformation already exist in `iter_affine_map.h`,
and are used in the implementation.

A buffer may be allocated with `AllocateNode`, and may be interacted
with using `BufferLoadNode` and `BufferStoreNode`.
`BufferRealizeNode` should only appear in TE-based schedules, and
should be converted to `AllocateNode`.  `LoadNode` and `StoreNode`
are deprecated.

## Impacted TIR Nodes

- BufferNode
  - Describes a N-d buffer.  The layout of the buffer may be

- BufferRealizeNode
  - Realization of a buffer, in logical layout.
  - For external buffers, serves as an optional annotation.  For
    internal buffers, results in allocation of memory.


- BufferLoadNode/BufferStoreNode
  - Read/write of a buffer.

  - Change from previous behavior: Will exist throughout the lowering
    process, and will be passed to the low-level code generators.
    Transformations that previously created `Load` and `Store` nodes
    will instead create `BufferLoad` and `BufferStore` nodes with 1-d
    indices.



- AllocateNode
  - Allocation of a buffer, in physical layout.

  - Declares an allocation of a buffer.

  - Change from previous behavior: Previously, `AllocateNode` held the
    `buffer_var`, datatype, and buffer extents directly. After
    implementation of this RFC, `AllocateNode` will instead hold the
    `Buffer` that is to be allocated.


- LoadNode/StoreNode
  - Read/write of a 1-d buffer, given a `Var` pointer to the start of
    the buffer and a single index.

  - Deprecated, should instead use `BufferLoad` and `BufferStore` with
    a 1-d index.


## Impacted tir Transformations

- `ApplyBufferTransforms`
  - A new pass that takes as input a TIR graph that may have buffer
    transformations stored in the `PrimFunc` attributes.  Returns
    a TIR graph with all buffer transforms applied as specified.

  - Rewrite `indices` in BufferStore/BufferLoad nodes based on the
    specified transformation.

  - The transformations are stored as a `Map<Var, Array<IndexMap>>` in
    the `"buffer_layout_transformations"` attribute of a primfunc.
    All buffers whose `BufferNode::data` is a key in this map should
    have their physical layout rewritten.  If the array contains
    multiple transformations, they are applied sequentially.

    A possible structure for the `IndexMap` node is shown
    below.

    ```
    class IndexMapNode : public Object {
    public:
      /*! \brief Variables representing the indices prior to remapping.
       *
       * If initial_index is empty, then final_index should also be
       * empty, and no mapping is applied.
       */
      Array<Var> initial_index;

      /*!
       * \brief Expressions defining the indices after remapping.
       *
       * These expressions should only be in terms of the initial_index,
       * and must be expressible as a `tvm::arith::IterSumExpr`.  The
       * mapping from `initial_index` to `final_index` must be injective.
       *
       * If final_index is empty, then initial_index should also be
       * empty, and the map is an identity function.
       */
      Array<PrimExpr> final_index;
    };
    ```

  - After applying the transformations, the
    `"buffer_layout_transformations"` attribute should be removed.
    This ensures that additional application of
    `ApplyBufferTransforms` has no effect.

- FlattenBuffer/StorageFlatten

  - Existing passes that convert from logical layout to physical
    layout for TE schedules (StorageFlatten) or TensorIR schedules
    (FlattenBuffer).

  - The transformations are stored as a `Map<Var, Array<IntImm>>` in
    the `"buffer_axis_separators"` attribute of a primfunc.  All
    buffers whose `BufferNode::data` is a key in this map should be
    flattened to an output buffer of rank
    `separators[buf->data].size()+1`.  All other buffers should be
    flattened to a 1-d output buffer.

  - After flattening a buffer to an N-d output, the corresponding
    value in the `"buffer_axis_separators"` attribute should be set to
    `range(N-1)`.  This ensures that repeated application of the
    flattening passes have no additional effect.  (The attribute
    shouldn't be deleted entirely, as that would cause a flattened
    rank-`N` buffer and an unflattened rank-`N` buffer to have
    identical representations.)


## Examples

The following are intended as pseudo-code, and exclude details not
relevant to this RFC (e.g. dtype).  These do not correspond with the
final version of TensorIR that implements this RFC.  Numeric values
are shown unsimplified to indicate where they come from.

The first example shows a 2-d buffer with no layout transformations
explicitly specified.  The generated `PrimFunc` has no
`"buffer_layout_transformations"` attribute, and so the default
behavior is used, applying a row-major traversal to generate a flat
1-d buffer.

```python
# In TE schedule, no call to transform_layout.

# Initial TIR graph
x = Buffer(name="x", shape=[2,3])
with Allocate(x):
    val = BufferLoad(x, [10, 15])
    BufferStore(x, 7, [20, 23])

# After flattening to 1-d
x = Var(name="x")
with Allocate(x, shape=[2*3]):
    val = BufferLoad(x, [10*3 + 15])
    BufferStore(x, 7, [20*3 + 23])
```

This next example shows a 2-d logical buffer, which is lowered to a
1-d physical buffer.  `transform_layout` has been used to define a
physical layout whose fastest changing dimension corresponds to the
first index in the logical layout.

```python
# In TE schedule
# s[x].transform_layout(lambda i,j: [j,i])

# Initial TIR graph
attrs["buffer_layout_transformations"][x] = lambda i,j: [j,i]
x = Buffer(name="x", shape=[2,3])
with Allocate(x):
    val = BufferLoad(x, [10, 15])
    BufferStore(x, 7, [20, 23])

# After applying the explicit reordering
x = Buffer(name="x", shape=[3,2])
with Allocate(x):
    val = BufferLoad(x, [15, 10])
    BufferStore(x, 7, [23, 20])

# After flattening to 1-d
x = Var(name="x")
with Allocate(x, shape=[3*2]):
    val = BufferLoad(x, [15*2 + 10])
    BufferStore(x, 7, [23*2 + 20])
```

The next example shows a remapping from NHWC logical layout to NCHWc
physical layout.  The 4 logical axes are expanded to 5 logical axes
during the `ApplyBufferTransforms` pass, then flattened into 1 physical
axis during StorageFlatten/FlattenBuffer.

```python
# In TE schedule
# s[x].transform_layout(lambda n,h,w,c: [n, c//4, h, w, c%4])

# Initial TIR graph
attrs["buffer_layout_transformations"][x] = lambda n,h,w,c: [n, c//4, h, w, c%4]
x = Buffer(name="x", shape=[16,64,64,128], reorder_splits=nhwc_to_nchwc, axis_separators=[])
with Allocate(x):
    val = BufferLoad(x, [11, 37, 23, 101])

# After applying the explicit reordering
x = Buffer(name="x", shape=[16, 128/4, 64, 64, 4], reorder_splits=[], axis_separators=[])
with Allocate(x):
    val = BufferLoad(x, index=[11, floor(101/4), 37, 23, 101%4])

# After flattening to 1-d
x = Var(name="x")
with Allocate(x, shape=[16 * (128/4) * 64 * 64 * 4]):
    val = BufferLoad(x, index=[(128/4)*64*64*4*11 + 64*64*4*floor(101/4) + 64*4*37 + 4*23 + 101%4])
```

Lastly, an example of remapping from `NHWC` logical layout to `NCHWc`
physical layout, packed into a 2-d physical layout with `NCH` in the
first physical axis and `Wc` in the second physical axis.  This is the
definition used by the current `"global.texture"` definition used for
texture memory.  The change applied during SplitReorderIndices is
identical to the previous example, but StorageFlatten produces a 2-d
physical index.  The interpretation of this 2-d index depends on the
target-specific codegen.

```python
# In TE schedule
# s[x].transform_layout(lambda n,h,w,c: [n, c//4, h, te.AXIS_SEPARATOR, w, c%4])

# Initial TIR graph
attrs["buffer_layout_transformations"][x] = lambda n,h,w,c: [n, c//4, h, w, c%4]
attrs["buffer_axis_separators"][x] = [2]
x = Buffer(name="x", shape=[16,64,64,128])
with Allocate(x):
    val = BufferLoad(x, [11, 37, 23, 101])

# After applying the explicit reordering.
attrs["buffer_axis_separators"][x] = [2]
x = Buffer(name="x", shape=[16, 128/4, 64, 64, 4])
with Allocate(x):
    val = BufferLoad(x, index=[11, floor(101/4), 37, 23, 101%4])

# After applying StorageFlatten or FlattenBuffer.  The final result is
# 2-d, due to the te.AXIS_SEPARATOR used in the `.transform_layout`.
# The `"buffer_axis_separators"` attribute is set to [0], to
# distinguish this 2-d flattened buffer from a 2-d unflattened buffer.
attrs["buffer_axis_separators"][x] = [0]
x = Var(name="x")
with Allocate(x, shape=[16 * (128/4) * 64, 64 * 4]):
    val = BufferLoad(x, index=[(128/4)*64*11 + 64*floor(101/4) + 37, 4*23 + 101%4])
```



# Drawbacks
[drawbacks]: #drawbacks

This change may make it more difficult to reason about the memory
layout when writing the `te.compute` definition.  When the physical
layout differs from the logical layout, it isn't guaranteed that
`A[i]` and `A[i+1]` will be adjacent.  For example, a tensor with
compute definition defined in `NHWC` layout and with layout
transformation to `NCHWc` defined by `[n, c//4, h, w, c%4]`, locations
`(0,0,0,3)` and `(0,0,0,4)` in the compute definition will not be
adjacent.


# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

- Can these design goals be met with existing features?

  The `te.compute` function can be used to define an updated layout.
  However, this introduces a new tensor that must be inlined to avoid
  additional memory allocation, and cannot be used for input
  parameters.

  This design applies equally to tensors defined as a result of a
  computation and to input tensors.  In both cases, the
  `transform_layout` causes all reads/writes to that buffer to obey
  the specified layout.  In the case of input tensors, it states that
  the tensors passed in will be in the specified format.


- Should buffer transformations be a node within a TIR graph, or an
  attribute?

  Option 1 is preferred.

  - Option 1: The transformations are stored in attributes of
    `PrimFunc`.

    This makes it clear that the transformations apply to all uses of
    the buffer within the graph, and are not scoped to some region of
    the TIR graph.

  - Option 2: The transformations are stored in node that inherits
    from `tir::Stmt`.

    This would be easier for other passes to visit using
    `StmtVisitor`, if the layout transformations require modification.
    However, it would add confusion if a `Stmt` impacts buffers far
    outside its own scope.



- When should the `tir::transform::ApplyBufferTransforms` pass be
  applied?

  Applying it at the end of phase-2 in `driver_api.cc::CreatePassList`
  satisfies these conditions.

  - To ensure that host and device have the same definition for buffer
    layout, it should occur before the host/device split in
    `MakePackedAPI`.

  - Since other transformations can make use of buffer
    transformations, it should otherwise be as late as possible in the
    lowering flow.  (e.g. `InjectDoubleBuffer` mapping to a new buffer
    shape)



- Should buffer transformations re-use functionality of other nodes?

  Option 1 is preferred.

  - Option 1: Add buffer transformations as an attribute to the
    `PrimFunc`.

  - Option 2: In TE-based schedules, `AttrStmtNode` could give the
    buffer to be transformed, along with the transformation to be
    applied, similar to how `buffer_bind_scope` is currently handled.

    The `BufferTransform` must also contain multiple objects that are
    not derived from `PrimExpr`, the buffer to be transformed and the
    mapping to be applied, while `AttrStmtNode` only allows a single
    `ObjectRef` node and a `PrimExpr` value.

  - Option 3: In TensorIR-based schedules, `MatchBufferRegion` could
    be extended to also include a transformation while performing the
    buffer replacement.

    However, this could make it more difficult to reason about which
    locations in the buffer region are being accessed.

  - Option 4: The `BufferNode` object could contain an array of
    transformations that should be applied to it during the lowering
    process.  This would be convenient and allow for arbitrarily many
    transformations.

    Wouldn't follow the TVM convention of having annotations external
    to the node itself.


- Where should transformations to be applied to the function inputs be
  specified?

  Option 1 is preferred.

  - Option 1: Any `BufferTransform` that describes a buffer in the
    `PrimFuncNode::buffer_map` gets applied to that buffer.

    Would require two traversals, the first to locate all buffer
    transforms, and the second to apply them.

  - Option 2: `BufferTransform` nodes listed in the `PrimFunc::attrs`
    under a `"buffer_argument_transforms"` key apply to the function arguments.

    Would only need a single traversal to apply.

    Would require other passes to be aware of where a buffer was first
    defined, in order to add it to the appropriate location.


- What arguments should the function passed to `transform_layout` accept?

  In these examples, the tensor is rank `N` prior to the
  transformation.

  Option 3 is preferred.

  - Option 1: Accept a list of length `N`.  Each element of the list
    is a variable corresponding to a coordinate in the input tensor.

    This would be the simplest python implementation, but would
    require additional configuration to have named variables in the
    mapping.

  - Option 2: Accept `N` named positional arguments (`func(i,j,k)`),
    where each argument is a variable corresponding to a coordinate in
    the input tensor.

    This follows the usual method of defining the `fcompute` function
    passed to `te.compute`.  This also allows the named variables to
    be used as the names in TIR, improving readability.

    However, this wouldn't allow utility functions that define
    transformations that apply to an arbitrary number of indices, such
    as a layout transformation that changes the last index, while
    leaving the other `N-1` indices untouched.

  - Option 3: Accept either `N` named positional arguments
    (`func(i,j,k)`), or a variable number of arguments
    (`func(*indices)`).

    This follows the same convention as the `fcompute` function passed
    to `te.compute`.  This would allow either an explicit listing of
    all indices as named arguments, or an arbitrary number of indices.


# Prior art
[prior-art]: #prior-art

- CuDNN has an [explicit enumeration of allowed input
  formats](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnTensorFormat_t),
  which are specific to image formatting.

- The reorder/split/flatten sequences is equivalent in numpy to using
  `np.reshape` to split the logical axes, then `np.transpose` to
  reorder them, then `np.reshape` to merge multiple axes into the N-d
  physical axes.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

- Should the `te.AXIS_SEPARATOR` appear in the TIR graph?

  Option 1 is preferred.

  - Option 1: The `te.AXIS_SEPARATOR` is a TE-specific concept, and
    does not appear in the generated TIR graph.  Instead, it changes
    the `BufferTransform` node that represent the flattening of
    buffers to a device-supported number of indices.

    This would be a unified way to represent all layout
    transformations in the TIR graph, which may or may not change the
    rank of the buffer.  The flattening of buffers to a
    device-supported rank would be handled identically to any other
    layout transformation, rather than having an implicit row-major
    traversal.

  - Option 2: The `te.AXIS_SEPARATOR` is represented in the TIR graph,
    and alters the behavior of the `StorageFlatten` pass.  There is no
    `BufferTransform` node that represents the flattening of

    In a TIR graph without any other modifications, this would
    maintain the current behavior of the `StorageFlatten` pass, which
    reduces the N-d buffer to a 1-d buffer by a row-major traversal.
    In a TIR graph with some additional annotation to represent the
    `M` axis separators, the N-d buffer could instead be reduced to a
    `M+1`-d buffer.

- What is appropriate terminology for size/shape/extent of physical
  and logical buffers?

  If Allocate/BufferStore/BufferLoad each hold a reference to the
  buffer they act upon, then this becomes a somewhat irrelevant
  question, as there is only one `BufferNode::shape`.

  - I am partial to using "shape" both for the N-d parameters, and
    have attempted to use it consistently through this RFC.
  - "size" implies a 1-d buffer, which wouldn't be appropriate for
    an N-d parameter.
  - "extent" would be a reasonable name, but is currently used by
    `tvm::RangeNode` to indicate a range of values that may start at
    a non-zero value.  Since the indices for logical and physical
    buffers both start at zero, using "extents" for the maximum
    index would imply some offset.



- How should loops over an array be handled when re-writing the shape?

  To avoid memory latency issues, loops should iterate over an array
  sequentially when possible.  Iteration that is defined in terms of
  the logical layout may be inappropriate for the physical layout.

  Option 3 is preferred.

  - Option 1: Do nothing, and always keep the same iteration order,
    using the same iteration axes as defined in the compute
    definition.

    This would produce valid code, but not necessarily performant
    code.  This can be a default behavior during development, to be
    improved upon.

  - Option 2: Automatically detect loops that are over the full extent
    of an array in sequential order of the logical layout, and rewrite
    to be in sequential order of the physical layout.

    This would reduce the memory latency issues, but raises some
    implementation questions.

    - If a loop body references multiple tensors with different
      physical layouts, which should define the loop iteration order?

    - If a series of nested loops contains a `cache_read` or
      `cache_write` stage, can these be recognized and reordered?

  - Option 3: Expose the transformed axes to be used as part of a
    schedule definition.  In TE, the return value from `AA =
    s[A].transform_layout(...)` would be a tensor, and the transformed
    axes `AA.op.axis` can then be used for the remainder of the
    schedule.

    This would allow the greatest flexibility, but would make the
    schedule dependent on the transformed layout, beyond the one
    definition.

# Future possibilities
[future-possibilities]: #future-possibilities

- Could be used to simplify many of the `topi` schedules for image
  processing.
- Could introduce variation of physical layout during `cache_read` and
  `cache_write` steps, as a potential source of optimization.
