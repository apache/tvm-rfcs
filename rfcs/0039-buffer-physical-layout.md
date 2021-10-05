- Feature Name: Buffer Physical Layout
- Start Date: 2021-10-05
- RFC PR: [apache/tvm-rfcs#0039](https://github.com/apache/tvm-rfcs/pull/0039)
- GitHub Issue: TODO

# Summary
[summary]: #summary

This RFC introduces a hard boundary between the “logical layout” of a
mathematical tensor and the “physical layout” of a buffer in memory,
along with a specification for defining the conversion between the
two.

# Motivation
[motivation]: #motivation

Currently, TVM assumes that all buffers can be treated as flat memory.
That is, while a tensor may have N dimensions, the underlying buffer
allocated by the low-level codegen has a single value defining the
size, and access into that buffer is done using a single index.  This
assumptions holds for most cases, such as a CPU accessing RAM, but
doesn't hold in all cases.  For example, texture memory on a GPU
requires two indices to access.  In addition, computations that are
semantically identical (e.g. 2-d convolution) require independent
compute definitions and schedules (e.g. `conv2d_nchw` and
`conv2d_hwcn`) based on the format of the data accepted as input.

This RFC introduces a mechanism to specify and vary the physical
layout of buffers in memory.  This will allow for target-specific
handling of non-flat memory, and will allow for code re-use across
compute definitions that differ only in memory layout.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

“Logical layout” refers to the layout of a tensor as it exists in the
tensor.  All indices refer to the location of elements within an N-d
tensor, which may or may not correspond to the layout as it exists in
either host or device memory.  For example, compute defintions for
image processing may be written on tensors in the NHWC format.

“Physical layout” refers to the layout of memory as it exists within
physical memory, either on the host or on the device.  For example,
the physical layout of that same image processing data may a row-major
traversal of a NCHWc layout (e.g. [cudnn's `CUDNN_TENSOR_NCHW_VECT_C`
format](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnTensorFormat_t)),
where the C axis has been split into vectorizable chunks, and
reordered.

For logical layouts, any number of dimensions are allowed.  However,
for physical layouts, the dimensionality must be supported by the
specific runtime.  1-d physical layouts correspond to flat memory, and
should be supported by all runtimes.  N-d physical layouts have
runtime-specific interpretation, and may not be supported by all
runtimes.  For example, for OpenCL devices that support texture
memory, a 2-d physical layout may be used to represent access into a
2-d memory space.

To define the physical layout in a TE schedule, use the
`set_physical_layout` method of a schedule, as shown below.  The
arguments to `set_physical_layout` are either tuples of
`(logical_axis, factor)`, to indicate that the logical axis should be
split by the factor given, or as `logical_axis` to indicate that the
logical axis should be used with no additional splitting.  The order
of arguments defines any reordering that may occur when generating the
physical layout.  If `set_physical_layout` isn't called, then no
splits or reorderings are applied.

For example, below defines the reordering from NHWC logical layout to
NCHWc physical layout.

```python
# Compute definition, written in terms of NHWC logical axes
B = te.compute(A.shape, lambda n,h,w,c: A[n,h,w,c])
s = te.create_schedule(B.op)

# Option 1: Numeric axes
s[B].set_physical_layout(0, 3, 1, 2, (3,4))

# Option 2: Equivalent, named axes
n,h,w,c = B.op.axis
s[B].set_physical_layout(n, c, h, w, (c,4))

# Compute definition that would produce an equivalent physical layout
B_equivalent = te.compute(
    [A.shape[0], A.shape[3]//4, A.shape[1], A.shape[2], 4],
    lambda n, c_outer, h, w, c_inner: A[n, h, w, 4*c_outer+c_inner],
)
```

By default, after the splits and reorders are applied, all axes are
flattened to a single physical axis by following a row-major
traversal.  This produces a 1-d physical layout, which corresponds to
flat memory.  To add additional dimensions in the physical layout,
insert `te.PHYSICAL_AXIS_SEPARATOR` into the axis list in
`set_physical_layout`.  These define groups of axes, where each group
is combined into a single physical axis.

```python
B = te.compute(shape=(M,N,P,Q), ...)
m, n, p, q = B.op.axis
s = te.create_schedule(B.op)

# Default, produces a 1-d allocation with shape (M*N*P*Q,)
s[B].set_physical_layout(m, n, p, q)

# One separator, produces a 2-d allocation with shape (M*N, P*Q).
s[B].set_physical_layout(m, n, te.PHYSICAL_AXIS_SEPARATOR, p, q)

# Two separators, produces a 3-d allocation with shape (M, N*P, Q).
s[B].set_physical_layout(m, te.PHYSICAL_AXIS_SEPARATOR, n, p, te.PHYSICAL_AXIS_SEPARATOR, q)

# Can be used along with reorders and splits.
s[B].set_physical_layout(m, q, n, te.PHYSICAL_AXIS_SEPARATOR, p, (q, 4))
```


# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

A buffer in logical layout may be allocated with `BufferRealizeNode`,
and may be interacted with using `BufferLoadNode` and
`BufferStoreNode`.  A buffer in physical layout may be allocated with
`AllocateNode`, and may be interacted with using `LoadNode` and
`StoreNode`.  Lowering from logical to physical layout will occur at
the StorageFlatten pass for TE-based schedules, and at the
FlattenBuffer pass for TensorIR-based schedules.

A graph may only interact with a buffer in a single type of layout.
That is, a buffer that is allocated with BufferRealize (logical
layout) may not be accessed with Load (physical layout), and must
instead be accessed with BufferLoad (logical layout).  Logical layout
and physical layout may co-exist within the same graph, so long as
they refer to different buffers.

## Impacted TIR Nodes

- BufferNode
  - Describes a buffer, in logical layout.
  
  - Change: Add an `reorder_split` member variable, to describe
    reorderings and axis splits that generate the physical layout from
    the logical layout.  These will default to a physical layout that
    assigns the first logical index to the slowest-changing dimension,
    and the last logical index to the fastest-changing dimension, with
    no reordering.  This default behavior reproduces the previous
    behavior.
    
  - Change: Define which axes are to be merged by specifying
    `axis_separators`.  Groups of logical axes, where each group
    consists of all logical axes that do not have a separator between
    them, are to be merged into a single physical index.  This will
    default to an empty list, collapsing all logical axes/indices into
    a single physical axis/index, reproducing the previous behavior.


- BufferRealizeNode
  - Realization of a buffer, in logical layout.
  - For external buffers, serves as an optional annotation.  For
    internal buffers, results in allocation of memory.


- BufferLoadNode/BufferStoreNode
  - Read/write of a buffer, in logical layout.


- AllocateNode
  - Allocation of a buffer, in physical layout.
  
  - Gives the N-d shape of the physical buffer
    
  - Change from previous behavior: Previously, all allocations were
    done as a 1-d size of the physical layout, but `Array<PrimExpr>
    AllocateNode::extents` held the shape of the logical layout used
    to generate the `AllocateNode`.  This is replaced with Replace N-d
    “extents” (logical layout) with N-d “shape” (physical layout).
    Because these are both of type `Array<PrimExpr>`, but have
    different semantics, this change is made in two steps rather
    than a single find/replace.
    
    - Step 1: Replace N-d `Array<PrimExpr> extents` with 1-d `PrimExpr
      extent`.  Any optimization passes that require knowledge of the
      logical layout should be moved prior to the
      StorageFlatten/FlattenBuffer pass and updated to act on the
      logical layout.
      
    - Step 2: Replace 1-d `PrimExpr extent` N-d `Array<PrimExpr>
      shape`.  Any access that assumes flat memory should verify that
      `shape.size()==1`.


- LoadNode/StoreNode
  - Read/write of a buffer, in physical layout.
  - Change from previous behavior: Replace 1-d `PrimExpr index` with
    N-d `Array<PrimExpr> index`.


## Impacted tir Transformations

- SplitReorderIndices
  - A new pass that takes as input a TIR graph with buffers in logical
    layout.  The return from SplitReorderIndices
    `buffer.reorder_splits.size()==0` for all buffers in the
    graph, and represents the same computation as the input.
  
  - Replace the `Buffer` object in `BufferRealize`, `BufferLoad`, and
    `BufferStore` nodes with updated `Buffer` objects whose shape has
    all axis splits and reordering applied, and whose `reorder_splits`
    is empty.
    
  - Rewrite `index` in BufferStore/BufferLoad nodes to follow the
    updated layout.
  
- FlattenBuffer/StorageFlatten
  - Existing passes that convert from logical layout to physical
    layout for TE schedules (StorageFlatten) or TensorIR schedules
    (FlattenBuffer).
    
  - Use the `N-1` axis separators specified in BufferNode to convert to
    an N-d physical layout. The default of having 0 axis separators
    will correspond to the previous behavior of flattening to a 1-d
    physical layout.


## Examples

The following are intended as pseudo-code, and exclude details not
relevant to this RFC (e.g. dtype).  These do not correspond with the
final version of TensorIR that implements this RFC.  Numeric values
are shown unsimplified to indicate where they come from.

This first example shows a 2-d logical buffer, which is lowered to a
1-d physical buffer.  `set_physical_layout` has been used to define a
physical layout whose fastest changing dimension corresponds to the
first index in the logical layout.

```python
# Initial graph, in logical layout
x = Buffer(name="x", shape=[2,3], reorder_splits=[1,0], axis_separators=[])
with BufferRealize(x):
    val = BufferLoad(x, [10, 15])
    BufferStore(x, 7, [20, 23])

# After SplitReorderIndices has been applied.
x = Buffer(name="x", shape=[3,2], reorder_splits=[], axis_separators=[])
with BufferRealize(x):
    val = BufferLoad(x, index=[15, 10])
    BufferStore(x, 7, index=[23, 20])

# After StorageFlatten/FlattenBuffer has been applied
x = Var(name="x")
with Allocate(x, shape=[3*2]):
    val = Load(x, index=[15*2 + 10])
    Store(x, 7, index=[23*2 + 10])
```

The next example shows a remapping from NHWC logical layout to NCHWc
physical layout.  The 4 logical axes are expanded to 5 logical axes
during the SplitReorderIndices pass, then flattened into 1 physical
axis during StorageFlatten/FlattenBuffer.

```python
# Initial graph, in logical layout
x = Buffer(name="x", shape=[16,64,64,128], reorder_splits=[0,3,1,2,(3,4)], axis_separators=[])
with BufferRealize(x):
    val = BufferLoad(x, [11, 37, 23, 101])

# After SplitReorderIndices has been applied.
x = Buffer(name="x", shape=[16, 128/4, 64, 64, 4], reorder_splits=[], axis_separators=[])
with BufferRealize(x):
    val = BufferLoad(x, index=[11, floor(101/4), 37, 23, 101%4])

# After StorageFlatten/FlattenBuffer has been applied
x = Var(name="x")
with Allocate(x, shape=[16 * (128/4) * 64 * 64 * 4]):
    val = Load(x, index=[(128/4)*64*64*4*11 + 64*64*4*floor(101/4) + 64*4*37 + 4*23 + 101%4])
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
# Initial graph, in logical layout
x = Buffer(name="x", shape=[16,64,64,128], reorder_splits=[0,3,1,2,(3,4)], axis_separators=[3])
with BufferRealize(x):
    val = BufferLoad(x, [11, 37, 23, 101])

# After SplitReorderIndices has been applied.
x = Buffer(name="x", shape=[16, 128/4, 64, 64, 4], reorder_splits=[], axis_separators=[3])
with BufferRealize(x):
    val = BufferLoad(x, index=[11, floor(101/4), 37, 23, 101%4])

# After StorageFlatten/FlattenBuffer has been applied
x = Var(name="x")
with Allocate(x, shape=[16 * (128/4) * 64, 64 * 4]):
    val = Load(x, index=[(128/4)*64*11 + 64*floor(101/4) + 37, 4*23 + 101%4])
```


# Drawbacks
[drawbacks]: #drawbacks

This change may make it more difficult to reason about the memory
layout when writing the `te.compute` definition.  When the physical
layout differs from the logical layout, it isn't guaranteed that
`A[i]` and `A[i+1]` will be adjacent.  For example, a tensor with
`NHWC` logical layout and a `NCHWc` physical layout defined by
`set_physical_layout(n,c,h,w,(c,4))`, logical indices `(0,0,0,3)` and
`(0,0,0,4)` will not be adjacent.


# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

This design applies equally to tensors defined as a result of a
computation and to input tensors.  In both cases, the
`set_physical_layout` causes all reads/writes to that buffer to obey
the specified layout.  In the case of input tensors, it states that
the tensors passed in will be in the specified format.

The `te.compute` function can be used to define an updated layout.
However, this introduces a new tensor that must be inlined to avoid
additional memory allocation, and cannot be used for input tensors.


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

- What is appropriate terminology for size/shape/extent of physical
  and logical buffers?
  - I am partial to using "shape" both for the N-d logical layout
    and the N-d physical layout, and have attempted to use it
    consistently through this RFC.
  - "size" implies a 1-d buffer, which wouldn't be appropriate for
    an N-d parameter.
  - "extent" would be a reasonable name, but is currently used by
    `tvm::RangeNode` to indicate a range of values that may start at
    a non-zero value.  Since the indices for logical and physical
    buffers both start at zero, using "extents" for the maximum
    index would imply some offset.
    
- How should loops over an array be handled when re-writing the shape?

  To avoid memory latency issues, loops should iterate over an array
  sequentially sequentially when possible.  Iteration that is defined
  in terms of the logical layout may be inappropriate for the physical
  layout.
  
  - Option: Do nothing, and always keep the same iteration order, as
    defined in terms of the logical axes.
    
    This would produce valid code, but not necessarily performant
    code.  This can be a default behavior during development, to be
    improved upon.
  
  - Option: Automatically detect loops that are over the full extent
    of an array in sequential order of the logical layout, and rewrite
    to be in sequential order of the physical layout.
    
    This would reduce the memory latency issues, but raises some
    implementation questions.

    - If a loop body references multiple tensors with different
      physical layouts, which should define the loop iteration order?
      
    - If a series of nested loops contains a `cache_read` or
      `cache_write` stage, can these be recognized and reordered?

  - Option: Expose the `reorder_split` definition to be used as part
    of a schedule definition.
  
    This would allow the greatest flexibility, but would make the
    schedule dependent on the physical layout.

# Future possibilities
[future-possibilities]: #future-possibilities

- Could be used to simplify many of the `topi` schedules for image
  processing.
- Could introduce variation of physical layout during `cache_read` and
  `cache_write` steps, as a potential source of optimization.
