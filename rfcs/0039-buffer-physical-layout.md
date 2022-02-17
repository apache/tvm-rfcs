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
NCHWc physical layout.

```python
# Compute definition, written in terms of NHWC logical axes
B = te.compute(A.shape, lambda n,h,w,c: A[n,h,w,c])
s = te.create_schedule(B.op)

def nhwc_to_nchwc(n, h, w, c):
    return [n, c//4, h, w, c%4]

transformed_nchwc_axes = s[B].transform_layout(nhwc_to_nchwc)

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

If the tensor whose layout is being transformed is the result of
`te.compute`, then the loop iteration order over that tensor will be
rewritten to be along the updated memory layout.  If the loop
iteration order is modified, these new loop iteration variables will
be returned from `transform_layout()`.

```python
A = te.placeholder(shape=[16,64,128])
B = te.compute(A.shape, lambda i,j,k: 2*A[i,j,k])

s = te.create_schedule(B.op)

# A is an input placeholder, and doesn't have nested loops that
# generate it.  Therefore, while the layout of A is rewritten along
# with any reads/writes into A, there are no loop iterators to be
# rewritten and no loop iterators are returned.
s[A].transform_layout(lambda i,j,k: [i*64 + j, k//4, k%4])

# B is a computed tensor, and is computed inside a sequence of nested
# loops.  Therefore, when B's layout is rewritten, those nested loops
# are also rewritten, and the corresponding loop iterators are
# returned.
i_outer, jk_merged, i_inner = s[B].transform_layout(lambda i,j,k: [i//4, 128*j + k, i%4])

# The loop iterators returned by transform_layout() can be used later
# in the schedule, if the iteration order should be different from the
# layout order of the output tensor.
s[B].reorder(i_outer, i_inner, jk_merged)
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

For schedules written in either TE or TIR, the axis separators are stored
in `BufferNode::axis_separators`.  For TIR-based schedules, the
re-indexing of a buffer is performed on demand.  For TE-based schedules,
the mapping used to re-index a buffer is stored in the
`"layout_transform_map"` attribute of the `PrimFunc`, and is applied as
part of lowering.  This attribute is a map whose keys are buffer var to
be reshaped, and whose values are the transformations to be applied.

Many of the utilities needed for this transformation already exist in
`iter_affine_map.h`, and are used in the implementation.  For TIR-based
schedules, the transformation primitive is appleid immediately.

A buffer may be allocated with `AllocateNode`, and may be interacted
with using `BufferLoadNode` and `BufferStoreNode`.
`BufferRealizeNode` should only appear in TE-based schedules, and
should be converted to `AllocateNode`.  `LoadNode` and `StoreNode`
are deprecated.

## Impacted TIR Nodes

- BufferNode
  - Describes a N-d buffer.  This may directly represent a tensor (N-d
    buffer produced by TE), a flat memory array (1-d buffer as input
    to the low-level codegen), or intermediates between them.

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
    the `"layout_transform_map"` attribute of a primfunc.
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
    `"layout_transform_map"` attribute should be removed.
    This ensures that additional application of
    `ApplyBufferTransforms` has no effect.

- FlattenBuffer/StorageFlatten

  - Existing passes that convert from logical layout to physical
    layout for TE schedules (StorageFlatten) or TensorIR schedules
    (FlattenBuffer).

  - The transformations are stored in the `Buffer` object as the
    `BufferNode::axis_separators`.  All buffers that share the same
    `BufferNode::data` should be flattened to an
    output buffer of rank `axis_separators.size()+1`.  All other
    buffers should be flattened to a 1-d output buffer.

  - After flattening a buffer to an N-d output, the corresponding
    value in the `axis_separators` should be set to `range(N-1)`.
    This ensures that repeated application of the flattening passes
    have no additional effect.  (The list shouldn't be deleted
    entirely, as that would cause a flattened rank-`N` buffer and an
    unflattened rank-`N` buffer to have identical representations.)


## Examples

The following are intended as pseudo-code, and exclude details not
relevant to this RFC (e.g. dtype).  These do not correspond with the
final version of TensorIR that implements this RFC.  Numeric values
are shown unsimplified to indicate where they come from.

The first example shows a 2-d buffer with no layout transformations
explicitly specified.  The generated `PrimFunc` has no
`"layout_transform_map"` attribute, and so the default
behavior is used, applying a row-major traversal to generate a flat
1-d buffer.

```python
# In TE schedule, no call to transform_layout.

# Initial TIR graph
x = Buffer(name="x", shape=[64,128])
with Allocate(x):
    val = BufferLoad(x, [10, 15])
    BufferStore(x, 7, [20, 23])

# After flattening to 1-d
x = Var(name="x")
with Allocate(x, shape=[64*128]):
    val = BufferLoad(x, [10*128 + 15])
    BufferStore(x, 7, [20*128 + 23])
```

This next example shows a 2-d logical buffer, which is lowered to a
1-d physical buffer.  `transform_layout` has been used to define a
physical layout whose fastest changing dimension corresponds to the
first index in the logical layout.

```python
# In TE schedule
# s[x].transform_layout(lambda i,j: [j,i])

# Initial TIR graph
attrs["layout_transform_map"][x] = lambda i,j: [j,i]
x = Buffer(name="x", shape=[64,128])
with Allocate(x):
    val = BufferLoad(x, [10, 15])
    BufferStore(x, 7, [20, 23])

# After applying the explicit reordering
x = Buffer(name="x", shape=[128,64])
with Allocate(x):
    val = BufferLoad(x, [15, 10])
    BufferStore(x, 7, [23, 20])

# After flattening to 1-d
x = Var(name="x")
with Allocate(x, shape=[128*64]):
    val = BufferLoad(x, [15*64 + 10])
    BufferStore(x, 7, [23*64 + 20])
```

The next example shows a remapping from NHWC logical layout to NCHWc
physical layout.  The 4 logical axes are expanded to 5 logical axes
during the `ApplyBufferTransforms` pass, then flattened into 1 physical
axis during StorageFlatten/FlattenBuffer.

```python
# In TE schedule
# s[x].transform_layout(lambda n,h,w,c: [n, c//4, h, w, c%4])

# Initial TIR graph
attrs["layout_transform_map"][x] = lambda n,h,w,c: [n, c//4, h, w, c%4]
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
attrs["layout_transform_map"][x] = lambda n,h,w,c: [n, c//4, h, w, c%4]
x = Buffer(name="x", shape=[16,64,64,128], axis_separators=[2])
with Allocate(x):
    val = BufferLoad(x, [11, 37, 23, 101])

# After applying the explicit reordering.
x = Buffer(name="x", shape=[16, 128/4, 64, 64, 4], axis_separators=[2])
with Allocate(x):
    val = BufferLoad(x, index=[11, floor(101/4), 37, 23, 101%4])

# After applying StorageFlatten or FlattenBuffer.  The final result is
# 2-d, due to the te.AXIS_SEPARATOR used in the `.transform_layout`.
# The `axis_separators` are set to [0], to distinguish this 2-d flattened
# buffer from a 2-d unflattened buffer.

x = Buffer(name="x", shape=[16 * (128/4) * 64, 64*4], axis_separators=[0])
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


- What convention should be used for buffer indexing?

  Previously, the interpretation of an index into a buffer depended on
  whether the buffer was being accessed with
  `BufferStore`/`BufferLoad` (pre-flattening) or with `Store`/`Load`
  (post-flattening).  Since the same data structures will be used at
  all lowering stages, the indexing should have consistent semantics.

  Option 1 is preferred.

  - Option 1: When accessing a buffer, the type and offset are based on
    `buffer->dtype`.

    The offset of an element is given by `index *
    sizeof(buffer->dtype)`.  The type of the element being accessed is
    `buffer->dtype.with_lanes(index.lanes() * buffer->dtype.lanes())`.

    This is the convention used by user-defined schedules in TE, and
    in BufferLoad/BufferStore objects.  In this convention, scalar
    loads and vectorized loads can be expressed for scalar buffers and
    vectorized buffers.  Accessing a buffer to return a different
    datatype requires declaring an aliasing buffer that shares the
    same backing array.

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
    ```

    - Pro: The return type of `buf[0]` is always `buf.dtype`, even
      when `buf.dtype` is a vectorized type.

    - Pro: No changes needed on the user-defined schedules.

    - Con: Requires updates to code generators to follow this new
      convention.  However, the code generators will already require
      updates to support BufferLoad/BufferStore.

  - Option 2: When accessing a buffer, the type and offset are based on
    `buffer->dtype.element_of()`.

    The offset of an element is given by `index *
    sizeof(buffer->dtype.element_of())`.  The type of the element
    being accessed is `buffer->dtype.with_lanes(index.lanes())`.

    Prior to this RFC, this is the convention used by Load/Store
    nodes.  In this convention, scalar loads and vectorized loads can
    be expressed for scalar buffers and vectorized buffers.  Accessing
    a buffer to return a vectorized datatype requires using a
    vectorized index, even if the buffer holds a vectorized datatype.

    ```python
    @T.prim_func
    def scalar_load_from_scalar_buffer(A: T.Buffer[(64,), "float32"]):
        assert A[0].dtype == "float32"


    @T.prim_func
    def vector_load_from_vector_buffer(A: T.Buffer[(16,), "float32x4"]):
        assert A[T.ramp(0, 1, 4)].dtype == "float32x4"


    @T.prim_func
    def scalar_load_from_vector_buffer(A: T.Buffer[(16,), "float32x4"]):
        assert A[0].dtype == "float32"


    @T.prim_func
    def vector_load_from_scalar_buffer(A: T.Buffer[(64,), "float32"]):
        assert A[T.ramp(0, 1, 4)].dtype == "float32x4"
    ```

    - Pro: The number of lanes of output can be determined solely from
      the index used to access the buffer.  That is, `A[0]` is
      guaranteed to have one lane of output, and `A[Ramp(0, stride=1,
      lanes=4)]` is guaranteed to have four lanes of output.

    - Con: Access of a buffer with scalar index does not always have
      the same datatype as the buffer.  If the buffer has a vectorized
      datatype, then `buf[0].dtype != buf.dtype`.

    - Con: Need explicit check for vectorized types at the codegen
      level.

    - Con: Requires updates to user-defined schedules.


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
