- Feature Name: Layout Transformation Padding Roadmap
- Authors: [Eric Lunderberg](https://github.com/Lunderberg/),
           [Chris Sullivan](https://github.com/csullivan),
           [Wuwei Lin](https://github.com/vinx13/),
           [Junru Shao](https://github.com/junrushao1994)
- Start Date: 2022-06-06
- RFC PR: [apache/tvm-rfcs#0077](https://github.com/apache/tvm-rfcs/pull/0077)
- GitHub Issue: [apache/tvm#12261](https://github.com/apache/tvm/issues/12261)

# Table of contents
- [Table of contents](#table-of-contents)
- [Summary](#summary)
- [Motivation](#motivation)
- [Guide-level explanation](#guide-level-explanation)
  - [Padded Transformations](#padded-transformations)
  - [Defining Padded Values](#defining-padded-values)
  - [Overcompute vs Branching](#overcompute-vs-branching)
- [Reference-level explanation](#reference-level-explanation)
  - [TIR Changes](#tir-changes)
    - [New TIR Op, `tir::builtin::assume`](#new-tir-op-tirbuiltinassume)
    - [New TIR Op, `tir::builtin::undef`](#new-tir-op-tirbuiltinundef)
    - [Transformations/Metaschedule Primitives](#transformationsmetaschedule-primitives)
    - [Enhancement - `cache_read`, `cache_write`](#enhancement---cache_read-cache_write)
    - [Enhancement - transform_layout](#enhancement---transform_layout)
    - [New Utility - Reorder Loops According to Buffer](#new-utility---reorder-loops-according-to-buffer)
    - [Enhancement - Predicate for DomainTouched](#enhancement---predicate-for-domaintouched)
    - [Enhancement - Remove No Op](#enhancement---remove-no-op)
    - [Enhancement - Simplify](#enhancement---simplify)
    - [New Transform - Hoist Expression](#new-transform---hoist-expression)
    - [New Transform - Reduce Loop Extents](#new-transform---reduce-loop-extents)
    - [Utility - Merge Adjacent Loops](#utility---merge-adjacent-loops)
    - [New Primitive - Remove Branching Through Overcompute](#new-primitive---remove-branching-through-overcompute)
    - [New Primitive - Remove Overcompute Through Branching](#new-primitive---remove-overcompute-through-branching)
    - [New Lowering Transform - Remove T.assume](#new-lowering-transform---remove-tassume)
    - [New Lowering Transform - Remove T.undef](#new-lowering-transform---remove-tundef)
  - [Implementation options](#implementation-options)
    - [Never write to transformation padding](#never-write-to-transformation-padding)
    - [Never read from transformation padding](#never-read-from-transformation-padding)
    - [Allocate internal buffer containing transformation padding](#allocate-internal-buffer-containing-transformation-padding)
    - [Explicitly write next operator's desired default at end of function](#explicitly-write-next-operators-desired-default-at-end-of-function)
    - [Implicitly write default value of next operator](#implicitly-write-default-value-of-next-operator)
    - [Apply operator element-wise over the transformation padding](#apply-operator-element-wise-over-the-transformation-padding)
    - [Multiple Buffer Semantics](#multiple-buffer-semantics)
  - [Points of Communication](#points-of-communication)
- [Drawbacks](#drawbacks)
- [Rationale and alternatives](#rationale-and-alternatives)
- [Prior art](#prior-art)
- [Unresolved questions](#unresolved-questions)
- [Future possibilities](#future-possibilities)

# Summary
[summary]: #summary

Buffer layout transformations can require padding in the transformed
buffer.  The efficiency of an operator depends on the semantics used
for loads and stores to values in the required padding.  The choice of
buffer semantics can reduce branch divergence and avoid repeated
setting of default values, but also imposes constraints between the
producer and consumer of a buffer.

This RFC discusses a general plan for specifying buffer semantics to
be used, and the constraints imposed.  Subsequent RFCs will follow
describing the design for support of each of the semantics proposed in
this roadmap.

# Motivation
[motivation]: #motivation

Suppose a buffer of shape `[14]` is transformed such that each index
`i` is mapped to `[i//4, i%4]`.  The first index can range from 0
(`0//4`) to 3 (`14//4`), and the second index can range from 0 (`0%4`)
to 3 (`3%4`).  Therefore, the transformed shape is `[4,4]`.  However,
this has 16 elements, because the transformed coordinates `(3,2)` and `(3,3)` do
not have a corresponding index on the workload range `0 <= i < 14`.  The final
result in these locations is not determined by the compute definition,
so we have flexibility in what to store in the padding that is
introduced by the transformation, and what assumptions can be made
when reading from those locations.

For example, an element-wise function may be most efficiently written
using vectorized instructions over all values, regardless of whether
they exist in the compute definition.  Or a maxpool may be most
efficiently written if input tensors have `-INF` stored in the
transformation padding.  Satisfying both of these at the same time may
not be possible.  While the compute definition doesn't impose
constraints on the values in the transformation padding, there are
still constraints imposed by the usage of those values by different
operators.


```
 ┌─Logical-index-space───────────────────┐
 │                                       │
┌▼─┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬─▼┌──┬──┐
│00│01│02│03│04│05│06│07│08│09│10│11│12│13│14│15│
└▲─┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴─▲┘
 │                                             │
 └─Physical-index-space────────────────────────┘

 ┌─Transformed-index-space─┐
 │                         │
 │      ┌────┬────┬────┬───▼┐
 │      │ 00 │ 01 │ 02 │ 03 │
 │      ├────┼────┼────┼────┤
 │      │ 04 │ 05 │ 06 │ 07 │
 │      ├────┼────┼────┼────┤
 │      │ 08 │ 09 │ 10 │ 11 │
 │      ├────┼────┼────┼────┤
 └──────► 12 │ 13 │ 14 │ 15 │
        └────┴────┴────┴────┘
```

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

## Padded Transformations

In general, a transformation will introduce the minimum amount of
padding such that all values in the original buffer can be stored in
the layout specified.  As a result, whether a transformation
introduces padding depends on the transformation being applied and the
buffer shape on which it is being applied.  For example, consider a
schedule that contains tensor `A` with shape `[16]` and tensor `B` with shape
`[14]`.

```python
# This transformation does not introduce padding.  The original shape
# of [16] produces the transformed shape [2,8], which contains the
# original 16 values no additional padding.
sched[A].transform_layout(lambda i: [i//8, i%8])

# This transform introduces padding.  The original shape of [14] also
# produces the transformed shape [2,8], which contains the original 14
# values and an additional 2 values of padding.  These are located at
# transformed indices [1,6] and [1,7].
sched[B].transform_layout(lambda i: [i//8, i%8])
```

The above example introduces padding at the end of a buffer.  By
including an offset in the layout transformation, we can instead place
the padding at the beginning of a buffer.

```python
# This transform introduces padding.  For 0 <= i < 14, the transformed
# index (i+2)//8 can have values of 0 or 1, so the transformed shape
# is [2,8].  There are no valid values of i that would produce [0,0]
# or [0,1], so these transformed indices contain padding.
sched[B].transform_layout(lambda i: [(i+2)//8, (i+2)%8])
```

In addition to moving the location of the padded indices, use of an
offset in a layout transformation can introduce additional padding.

```python
# This transformation introduces padding.  For 0 <= i < 16, the
# transformed index (i+2)//8 can have values of 0, 1, or 2, so the
# transformed shape is [3,8].  Padding is introduced from [0,0] to
# [0,1], and from [2,2] to [2,7].
sched[A].transform_layout(lambda i: [(i+2)//8, (i+2)%8])
```


## Defining Padded Values

When a buffer is transformed, the majority of values in the
transformed buffer are constrained to have the corresponding value in
the original buffer.  However, when a buffer is padded to meet some
alignment criteria, these additional padded values have no such
constraint.

To specify the values stored in the padding, the `transform_layout`
function takes an optional argument `pad_value` that
specifies the value that should be present in the padding.  This
should be a function that maps from transformed indices to an
`Optional[PrimExpr]`.

```python
# B.shape is [14]
transform = lambda i: [i//4, i%4]

# Three equivalent calls to perform the same layout transformation.
# Padding is introduced, but access of the padding is forbidden.
sched[B].transform_layout(transform)
sched[B].transform_layout(transform, pad_value=None)
sched[B].transform_layout(transform, pad_value=lambda io,ii: None)

# Padding is introduced, and contains zeros.
sched[B].transform_layout(transform, pad_value=0.0)
sched[B].transform_layout(transform, pad_value=lambda io,ii: 0.0)

# Padding is introduced, and contains undefined values.
sched[B].transform_layout(transform, pad_value=tir.undef(dtype="float32"))
sched[B].transform_layout(transform, pad_value=lambda io,ii: tir.undef(dtype="float32"))

# Padding is introduced, and wraps to the beginning of the array.
sched[B].transform_layout(transform, pad_value=lambda io,ii: B[0, (io-14)%4])
```

The `Buffer` object stores a predicate to identify which indices
contain padding, along with the expression given in `pad_value`.  This
expression may only contain constants and the transformed buffer
itself, and may not introduce dependencies on another buffer.

For a producer of the transformed buffer, if `pad_value` is defined,
the padding value must be written to the padding prior to the
completion of the operator.  Effectively, the producer must have a
postlude as follows:

```python
for transformed_indices in T.grid(*transformed_shape):
    if padding_predicate(*transformed_indices):
        B[transformed_indices] = pad_value(*transformed_indices)
```

For a consumer of the transformed buffer, these padding values are
initially unused, but may be used in later simplifications.

## Overcompute vs Branching

Depending on the computation being performed and the value stored in
the padding, there can be trade-offs between branching and
overcompute.  For example, consider the following `PrimFunc`, which
computes the sum over each row of the input data.

```python
@T.prim_func
def row_summation(a: T.handle, b: T.handle):
    A = T.match_buffer(shape=(16, 14), dtype="float32")
    B = T.match_buffer(shape=(16,), dtype="float32")
    for i in T.serial(16):
        B[i] = 0.0
        for j in T.serial(14):
            B[i] = B[i] + A[i, j]
```

We'd like to transform the layout of buffer `A` from `[i, j]` to `[i,
j//4, j%4]`, along with the loop iteration.  By default, after using
the `transform_layout` and `split` metaschedule primitives, we have
the following function.

```python
@T.prim_func
def row_summation(a: T.handle, b: T.handle):
    A = T.match_buffer(shape=(16, 4, 4), dtype="float32")
    B = T.match_buffer(shape=(16,), dtype="float32")
    for i in T.serial(16):
        B[i] = 0.0
        for j_outer, j_inner in T.grid(4, 4):
            if 4*j_outer + j_inner < 14:
                B[i] = B[i] + A[i, j_outer, j_inner]
```

If the conditional can be removed, this function would be much more
amenable for later vectorization, or to reduce branch divergence when
bound to a thread index.  If the padding in `A` is pre-filled with
zero, then `B[i] = B[i] + 0.0` is a no-op, and can be performed
without changing the final computation.

```python
@T.prim_func
def row_summation(a: T.handle, b: T.handle):
    A = T.match_buffer(shape=(16, 4, 4), dtype="float32")
    B = T.match_buffer(shape=(16,), dtype="float32")
    for i in T.serial(16):
        B[i] = 0.0
        for j_outer, j_inner in T.grid(4, 4):
            B[i] = B[i] + A[i, j_outer, j_inner]
```

By annotating the layout transformation with the value stored in the
padding, this condition can be proven, allowing this conditional to
automatically be removed.  Since the tradeoff between branching and
overcompute may or may not be beneficial dependent on the schedule,
these options are exposed as two additional transformations,
`tir.transform.RemoveBranchingThroughOvercompute` and
`tir.transform.RemoveOvercomputeThroughBranching`.


# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

## TIR Changes

### New TIR Op, `tir::builtin::assume`

A built-in operator that takes a single `PrimExpr` as an argument.  At
compile-time, an error should be raised if the argument can be
statically proven to be false at the point of call.  When lowering,
the `tir::builtin::assume` should be replaced with a no-op.
`tir::builtin::assume` is similar to the existing `tir::AssertStmt`,
but does not result in a runtime assertion for conditions that cannot
be proven.  This is equivalent to the [LLVM `__builtin_assume`
intrinsic](https://clang.llvm.org/docs/LanguageExtensions.html#builtin-assume).

The primary use of `assume` in this RFC is to allow local
simplifications within a `PrimFunc` to take advantage of information
that would otherwise require full end-to-end analysis of a model.
(See examples in [Points of Communication](#points-of-communication).)

* An assumption may only be inserted if it is statically proven, or if
  it is asserted by a user about a user-provided value.

* When splitting a PrimFunc into multiple PrimFuncs (e.g. factoring
  out a subroutine, hoisting an initial preprocessing stage into an
  independent PrimFunc), an assumption may become separated from the
  expressions that had initially been used to prove the assumption.

* An assumption may only be removed if it is statically proven.  A
  user-provided assumption may never be removed, as it may already
  have been used to perform irreversible simplifications.

* The expression within an assumption should be visited and mutated
  identically to any other `PrimExpr`.  This ensures that passes that
  redefine variables (e.g. by inlining a Let binding) do not result in
  an invalid expression in the `PrimExpr`.

### New TIR Op, `tir::builtin::undef`

A placeholder that represents a valid, but arbitrary value.  For
consumers, this is used in `T.assume()` expressions to indicate that
it is legal to access the address, but that no further constraints are
placed on the value present in the buffer.  For producers, this is
used to allow simplifications that change the value stored in the
output padding and would otherwise be forbidden.  (e.g. Leaving
partial computations written to padding by vectorized operations,
rather than zero-ing them out.)

* Multiplication of `0 * undef` may be simplified to zero, for both
  integer and floating-point types.

* A pure expression that uses `undef` can be simplified to `undef`.

* `undef` may not occur in the indices used to access a buffer.

* Two separate invocations instances of `undef` may not be assumed to
  be identical.  For example, the expression `undef - undef` may not
  be simplified to zero.  If this behavior is desired, the `undef` may
  be assigned in a `tir::LetStmt`,

* Storing a value of `undef` to a buffer is a no-op, and is removed
  during lowering.  (See [section on
  `tir.transform.RemoveUndefStore`](#new-lowering-transform-remove-tundef).)

See [section on element-wise
transformations](#apply-operator-element-wise-over-the-transformation-padding)
for example usage.


## Transformations/Metaschedule Primitives

### Enhancement - `cache_read`, `cache_write`

Can be used outside of any loop, with the same scope as the uncached
buffer.  The layout of the cache can then be transformed to operate on
a reshaped buffer without modifying the calling signature of the
original `PrimFunc`.

TODO: Check if this is already allowed.


### Enhancement - transform_layout

The `te.Stage.transform_layout` and `tir.Schedule.transform_layout`
methods will be updated to take an additional argument `pad_value:
Optional[Union[int, float, PrimExpr, Callable]]`.

For a transformation that introduces padding and with a defined
`pad_value`, a new stage is inserted following each write stage of the
transformed buffer.  This new stage writes `pad_value` to the
introduced padding.

```python
# Before transforming A_cache and B_cache
@T.prim_func
def func(A: T.Buffer[14, "float32"], B: T.Buffer[14, "float32"]):
    # A read cache of the input A
    A_cache = T.alloc_buffer(14, "float32")
    for i in T.serial(14):
        with T.block("A_cache"):
            A_cache[i] = A[i]

    # The computation itself, doubling the input value
    B_cache = T.alloc_buffer(14, "float32")
    for i in T.serial(14):
        with T.block("compute"):
            B_cache[i] = 2 * A_cache[i]

    # Copying from the write cache into the output B
    for i in T.serial(14):
        with T.block("B_cache"):
            B[i] = B_cache[i]


# After applying
# sched.transform_layout(block='compute', buffer='A_cache', lambda i: [i//4, i%4], pad_value=-1)
# sched.transform_layout(block='compute', buffer='B_cache', lambda i: [i//4, i%4], pad_value=-2)
@T.prim_func
def func(A: T.Buffer[14, "float32"], B: T.Buffer[14, "float32"]):
    A_cache = T.alloc_buffer(14, "float32")

    # When copying into the read cache, the loop iteration remains the
    # same, but writes to the transformed locations in `A_cache`.
    for i in T.serial(14):
        with T.block("A_cache"):
            A_cache[i // 4, i % 4] = A[i]

    # Immediately following the stage that produces values in the
    # transformed A_cache, a new stage is added that writes the
    # pad_value to the padding.
    for io, ii in T.grid(4, 4):
        with T.block("A_cache_padding"):
            if 4 * io + ii >= 14:
                A_cache[io, ii] = -1

    # The compute stage is unchanged, other than the updated indices
    # for A_cache and B_cache.
    B_cache = T.alloc_buffer(14, "float32")
    for i in T.serial(14):
        with T.block("compute"):
            B_cache[i // 4, i % 4] = 2 * A_cache[i // 4, i % 4]

    # Immediately following the stage that produces values in the
    # transformed A_cache, a new stage is added that writes the
    # pad_value to the padding.
    for io, ii in T.grid(4, 4):
        with T.block("B_cache_padding"):
            if 4 * io + ii >= 14:
                B_cache[io, ii] = -2

    # When copying into the read cache, the loop iteration remains the
    # same, but reads from the transformed locations in `B_cache`.
    for i in T.serial(14):
        with T.block("B_cache"):
            B[i] = B_cache[i // 4, i % 4]
```

If `pad_value` is defined and the transformed buffer does not have a
write stage within the body of the function, then it is an input
argument.  In this case, a new stage is added at the beginning of the
function, which calls `T.assume` for each input.

For buffer consumers, the constraint is added to the body as a call to
the `T.assume` builtin.  For buffer producers, the buffer constraint
is updated, and an additional loop is added to write `pad_value` to
the padding that has been introduced.

```python
# Before transforming A and B
@T.prim_func
def func(A: T.Buffer[14, "float32"], B: T.Buffer[14, "float32"]):
    # The computation, doubling the input value
    B_cache = T.alloc_buffer(14, "float32")
    for i in T.serial(14):
        with T.block("compute"):
            B[i] = 2 * A[i]


# After applying
# sched.transform_layout(block='compute', buffer='A', lambda i: [i//4, i%4], pad_value=-1)
# sched.transform_layout(block='compute', buffer='B', lambda i: [i//4, i%4], pad_value=-2)
@T.prim_func
def func(A: T.Buffer[(4, 4), "float32"], B: T.Buffer[(4, 4), "float32"]):
    # The buffer A does not have a write stage within this buffer.
    # Therefore, a new stage is inserted that calls T.assume.  The
    # assumption provided states that either the transformed indices
    # correspond to a set of indices in the pre-transformation buffer
    # (4*io + 11 < 14), or the value stored in the buffer is the
    # pad_value `A[io, ii] == -1`.
    for io, ii in T.grid(4, 4):
        T.assume(4 * io + ii < 14 or A[io, ii] == -1)

    # The computation, doubling the input value
    for i in T.serial(14):
        with T.block("compute"):
            B[i] = 2 * A[i]

    # The buffer B is an argument to the function, but contains a
    # write stage.  Therefore, we add a stage that writes the
    # pad_value after the write stage.
    for io, ii in T.grid(4, 4):
        with T.block("B_cache_padding"):
            if 4 * io + ii >= 14:
                B[io, ii] = -2
```

It is expected that the loop that writes padding may be simplified
later.  In this case, the loop over `io` can be removed, and the range
of the loop over `ii` can be reduced to `2 <= ii < 4`.  However, the
default implementation should not perform these simplification yet, as
this form is useful for [merging
loopnests](#utility-merge-adjacent-loops) after [rewriting for
sequential buffer
access](#new-utility-reorder-loops-according-to-buffer).

In TE, the write stage of a buffer is the stage that outputs the
transformed tensor.  In TIR, the write stage of a buffer is any block
that writes to all values of the pre-transformation tensor.

If a transformed buffer is an argument to the PrimFunc, then this
transformation alters the interface of the PrimFunc.  Whether this is
allowed strongly depends on the context in which the PrimFunc is being
used.

* If a PrimFunc must remain compatible with the current calling
  context, `transform_layout` may not be applied to argument buffers.
  For example, when creating an optimization candidate of a subgraph,
  if there is no legalization pass to handle layout disagreements
  between adjacent subgraphs, the candidate must remain compatible
  with the calling scope.

* If a PrimFunc is being modified as part of a transformation that
  also changes the context, `transform_layout` may be applied to
  argument buffers.  For example, if an end-to-end model is
  represented within a single `IRModule`, a transformation may alter a
  subgraph's calling convention and the call into the subgraph at the
  same time.

* If a PrimFunc is being modified independent independent of any
  context, `transform_layout` may be applied to argument buffers.  For
  example, a PrimFunc that is being prepared for use as a subgraph,
  but is not yet part of a graph, may be altered.


### New Utility - Reorder Loops According to Buffer

By default in S-TIR, `transform_layout` modifies the underlying layout
of a buffer, but does not re-order loops that iterate over the buffer.
The loop iterators can be re-written using split/fuse/reorder, but
doing so requires the user to manually translate the layout
transformation into the appropriate sequence of schedule primitives.

A new utility method `Schedule.sequential_buffer_access` should be
introduced, which generates and applies the sequence of
split/fuse/reorder schedule primitives such that the loop iterators are
rewritten for sequential access of a specific buffer.

```python
# Original function
@T.prim_func
def func(A: T.Buffer[(16,), "int32"]):
    with T.block('compute'):
        for i in T.serial(16):
            A[i] = i


# sched.transform_layout(block='compute', buffer='A', lambda i: [i//4, i%4])
@T.prim_func
def func(A: T.Buffer[(4, 4), "int32"]):
    with T.block('compute'):
        for i in T.serial(16):
            A[i // 4, i % 4] = i


# sched.sequential_buffer_access(block='compute', buffer='A')
@T.prim_func
def func(A: T.Buffer[(4, 4), "int32"]):
    with T.block('compute'):
        for io, ii in T.grid(4, 4):
            A[io, ii] = 4 * io + ii
```

This transformation is similar to what can be done using
split/fuse/reorder, but has two key differences.  First, it presents a
simpler user experience, as a transformed buffer can be accessed
sequentially without needing to duplicate the information in the
transformation.

Similar to `Schedule.split`, if the loop extents do not evenly divide
the transformation being applied, this primitive must introduce
conditionals to avoid accessing elements that were not previously
accessed.

```python
# Original function
@T.prim_func
def func(A: T.Buffer[(14,), "int32"]):
    with T.block('compute'):
        for i in T.serial(14):
            A[i] = i


# sched.transform_layout(block='compute', buffer='A', lambda i: [i//4, i%4])
@T.prim_func
def func(A: T.Buffer[(4, 4), "int32"]):
    with T.block('compute'):
        for i in T.serial(14):
            A[i // 4, i % 4] = i


# sched.sequential_buffer_access(block='compute', buffer='A')
@T.prim_func
def func(A: T.Buffer[(4, 4), "int32"]):
    with T.block('compute'):
        for io, ii in T.grid(4, 4):
            if 4 * io + ii < 14:
                A[io, ii] = 4 * io + ii
```

`Schedule.sequential_buffer_access` can operate on input buffers as
well as output buffers.

```python
# Original function
@T.prim_func
def func(
    A: T.Buffer[(16,), "int32"],
    F: T.Buffer[(3,), "int32"],
    B: T.Buffer[(14,), "int32"],
):
    with T.block('compute'):
        for i in T.serial(14):
            B[i] = 0.0
            for f in T.serial(3):
                B[i] = B[i] + F[f] * A[i + f]


# After transforming A's layout and B's layout, before rewriting loops
#
# sched.transform_layout(block='compute', buffer='A', lambda i: [i//4, i%4])
# sched.transform_layout(block='compute', buffer='B', lambda i: [i//4, i%4])
@T.prim_func
def func(
    A: T.Buffer[(4, 4), "int32"],
    F: T.Buffer[(3,), "int32"],
    B: T.Buffer[(4, 4), "int32"],
):
    with T.block('compute'):
        for i in T.serial(14):
            B[i // 4, i % 4] = 0.0
            for f in T.serial(3):
                B[i // 4, i % 4] = B[i // 4, i % 4] + F[f] * A[(i + f) // 4, (i + f) % 4]


# Option 1: Rewriting loops to match B's layout
# sched.sequential_buffer_access(block='compute', buffer='A')
#
# New iterators defined by B's access indices
# io = i//4
# ii = i%4
#
# Invert to find non-reduction axes to be replaced.
# i = 4*io + ii
@T.prim_func
def func(
    A: T.Buffer[(4, 4), "int32"],
    F: T.Buffer[(3,), "int32"],
    B: T.Buffer[(4, 4), "int32"],
):
    with T.block('compute'):
        for io, ii in T.grid(4, 4):
            if 4 * io + ii < 14:
                B[io, ii] = 0.0
                for f in T.serial(3):
                    # A's indices simplify from
                    #      [(i + f) // 4, (i + f) % 4]
                    #   => [(4*io + ii + f) // 4, (4*io + ii + f) % 4]
                    #   => [io + (ii + f) // 4, (ii + f) % 4]
                    B[io, ii] = B[io, ii] + F[f] * A[io + (ii + f) // 4, (ii + f) % 4]


# Option 2: Rewriting loops to match A's layout
# sched.sequential_buffer_access(block='compute', buffer='A')
#
# New iterators defined by A's access indices
# io = (i+f)//4
# ii = (i+f)%4
#
# Invert to find non-reduction axes to be replaced.
# i = 4*io + ii - f
@T.prim_func
def func(
    A: T.Buffer[(4, 4), "int32"],
    F: T.Buffer[(3,), "int32"],
    B: T.Buffer[(4, 4), "int32"],
):
    # Because the initialization of B[i//4, i%4] does not depend on f,
    # it cannot be expressed solely in terms of io and ii.  Therefore,
    # the initialization must be split into a separate loopnest.
    with T.block('init_compute'):
        for i in T.serial(14):
            B[i // 4, i % 4] = 0.0

    with T.block('compute'):
        for io,ii in T.grid(4,4):
            for f in T.serial(3):
                if 0 <= 4*io + ii - f < 14:
                    # B's indices simplify from
                    #      [i // 4, i%4]
                    #   => [(4*io + ii - f) // 4, (4*io + ii - f)%4]
                    #   => [io + (ii - f) // 4, (ii - f)%4]
                    B[io + (ii - f) // 4, (ii - f) % 4] = (
                        B[io + (ii - f) // 4, (ii - f) % 4] + F[f] * A[io, ii]
                    )
```

In some cases, it may not be possible to separate out the
initialization and computation in order to rewrite the loops for
sequential buffer accesss.  In this case,
`Schedule.sequential_buffer_access` will raise an error.

```python
# Original function
@T.prim_func
def conv1d_cumsum(
    A: T.Buffer[(16,), "int32"],
    F: T.Buffer[(3,), "int32"],
    B: T.Buffer[(14,), "int32"],
):
    with T.block('compute'):
        for i in T.serial(14):
            if i == 0:
                B[i] = 0
            else:
                B[i] = B[i - 1]

            for f in T.serial(3):
                B[i] = B[i] + F[f] * A[i + f]


# After transforming A's layout and B's layout, before rewriting loops
#
# sched.transform_layout(block='compute', buffer='A', lambda i: [i//4, i%4])
# sched.transform_layout(block='compute', buffer='B', lambda i: [i//4, i%4])
@T.prim_func
def conv1d_cumsum(
    A: T.Buffer[(4, 4), "int32"],
    F: T.Buffer[(3,), "int32"],
    B: T.Buffer[(4, 4), "int32"],
):
    with T.block('compute'):
        for i in T.serial(14):
            if i == 0:
                B[i // 4, i % 4] = 0
            else:
                B[i // 4, i % 4] = B[(i - 1) // 4, (i - 1) % 4]

            for f in T.serial(3):
                B[i // 4, i % 4] = B[i // 4, i % 4] + F[f] * A[(i + f) // 4, (i + f) % 4]


# Intermediate formed when attempting to re-order access to be
# sequential along A's layout.  This is not a legal transformation,
# because the initialization step requires the previous result the
# computation loop.  Therefore, Schedule.sequential_buffer_access will
# raise an error.
#
# sched.sequential_buffer_access(block='compute', buffer='A')
@T.prim_func
def conv1d_cumsum(
    A: T.Buffer[(4, 4), "int32"],
    F: T.Buffer[(3,), "int32"],
    B: T.Buffer[(4, 4), "int32"],
):
    with T.block('init_compute'):
        for i in T.serial(14):
            if i == 0:
                B[i // 4, i % 4] = 0
            else:
                B[i // 4, i % 4] = B[(i - 1) // 4, (i - 1) % 4]

    with T.block('compute'):
        for i in T.serial(14):
            for f in T.serial(3):
                B[i // 4, i % 4] = B[i // 4, i % 4] + F[f] * A[(i + f) // 4, (i + f) % 4]
```

This utility is not required for the TE interface, as the loopnest of
an output tensor is automatically rewritten to a row-major traversal.


### Enhancement - Predicate for DomainTouched

In `tvm::arith::DomainTouched`, track the condition for which a buffer
is touched, in addition to the indices that are touched.

### Enhancement - Remove No Op

Changes to be made to `tvm::tir::NoOpRemover`, which implements the
`tir.transform.RemoveNoOp` transform.

* If two sequential `BufferStore` occur, both of which write to the
  same buffer/index, and the second value stored does not read out the
  first value, then the first store is a no-op.

* If there exist two sequential blocks, the buffers/indices written by
  the second block are a superset of the buffers/indices written by
  the first block, and the second block does not read the
  buffer/indices written by the first block, then the first block is a
  no-op.

* Reading a value then immediately writing it back is a no-op.  A
  `BufferLoad` that is immediately used as a value to a `BufferStore`,
  with the same buffer and indices, can be removed.

  This functionality is currently part of
  `tvm::arith::StmtSimplifier`, but is needed here to recognize
  strings of no-op.  (Thought: Merge the Simplify and RemoveNoOp
  passes?)

* Writing a value that is known to exist within the buffer is a no-op.

  ```python
  # Before RemoveNoOp
  @T.prim_func
  def sum(A: T.Buffer[16, "float32"], B: T.Buffer[1, "float32"]):
      T.assume(B[0] == 0.0)

      B[0] = 0.0
      for i in T.serial(16):
          B[0] = B[0] + A[i]

  # After RemoveNoOp
  @T.prim_func
  def sum(A: T.Buffer[16, "float32"], B: T.Buffer[1, "float32"]):
      T.assume(B[0] == 0.0)

      for i in T.serial(16):
          B[0] = B[0] + A[i]
  ```


### Enhancement - Simplify

Changes to be made to `tvm::arith::StmtSimplifier` mutator, used in
the `tir.transform.Simplify` transform.

* When visiting an `IfThenElseStmt`, if the `then_case` and
  `else_case` are identical, replace with
  `SeqStmt({Evaluate(condition)}, then_case)`.

  Currently, the `tvm::arith::StmtSimplifier` mutator, checks if a
  condition can be proven, but doesn't do any checks on the body.

  TODO: Double-check that functionality doesn't already exist.

* If two sequential `IfThenElseStmt` have identical conditions, they
  should be merged.  Conditions are identical if each condition can be
  used to prove the other is true, even if they do not have the same
  functional form.

  ```python
  # Before merging identical conditionals
  @T.prim_func
  def func(A: T.Buffer[16, "float32"], B: T.Buffer[16, "float32"]):
      for i in T.serial(16):
          if i < 8:
              A[i] = 0.0
          else:
              A[i] = 1.0

          if i//8 == 0:
              B[i] = 2.0
          else:
              B[i] = 3.0

  # After merging identical conditionals
  @T.prim_func
  def func(A: T.Buffer[16, "float32"], B: T.Buffer[16, "float32"]):
      for i in T.serial(16):
          if i < 8:
              A[i] = 0.0
              B[i] = 2.0
          else:
              A[i] = 1.0
              B[i] = 3.0
  ```

  Similarly, if two sequential `IfThenElseStmt` have complementary
  conditions, they should be merged, with the `else_case` of the
  second conditional appended to the `then_case` of the first, and
  vice versa.  Conditions are complementary if assuming either
  condition can be used to prove the other is false.

  (Example usage in [later producer/consumer
  section](#explicitly-write-next-operators-desired-default-at-end-of-function).)

  ```python
  # Before merging complementary conditionals
  @T.prim_func
  def func(A: T.Buffer[(4,4), "float32"], B: T.Buffer[(4,4), "float32"]):
      for i,j in T.grid(4,4):
          if 4*i + j < 14:
              A[i] = 0.0
          else:
              A[i] = 1.0

          if i==3 and j>=2:
              B[i] = 2.0
          else:
              B[i] = 3.0


  # After merging complementary conditionals
  @T.prim_func
  def func(A: T.Buffer[(4,4), "float32"], B: T.Buffer[(4,4), "float32"]):
      for i,j in T.grid(4,4):
          if 4*i + j < 14:
              A[i] = 0.0
              B[i] = 3.0
          else:
              A[i] = 1.0
              B[i] = 2.0
  ```

  Because the body of one conditional may alter the result of the next
  conditional, conditionals should not be merged if they depend on
  buffer values for data-dependent conditionals.  Only conditionals
  that do not depend on mutable values should be merged.

  ```python
  # Data-dependent conditional, may not be merged
  @T.prim_func
  def func(A: T.Buffer[16, "float32"], B: T.Buffer[16, "float32"]):
      for i in T.serial(16):
          if A[i] < 0.0:
              A[i] = A[i] + 1.0

          if A[i] < 0.0:
              A[i] = 0.0


  # INCORRECT result of illegal merging of conditionals
  @T.prim_func
  def func(A: T.Buffer[16, "float32"], B: T.Buffer[16, "float32"]):
      for i in T.serial(16):
          if A[i] < 0.0:
              A[i] = A[i] + 1.0
              A[i] = 0.0
  ```

* When encountering a `T.assume` statement, this should be used for
  later simplifications.

  ```python
  # Before simplification
  @T.prim_func
  def func(A: T.Buffer[16, "int32"], n: T.int32):
      T.assume(n >= 0 and n < 8)

      for i in T.serial(16):
          A[i] = n//8

  # After simplification.  Because the range of `n` is provided in the
  # assumption, n//8 can be simplified.
  @T.prim_func
  def func(A: T.Buffer[16, "int32"], n: T.int32):
      T.assume(n >= 0 and n < 8)

      for i in T.serial(16):
          A[i] = 0
  ```

  These assumptions are statements only known to be true at the
  location of the `T.assume` call.  For assumptions based on value
  stored in a buffer, the assumption may be invalidated by later
  writes to the buffer.

  ```python
  # Before simplification
  @T.prim_func
  def func(A: T.Buffer[16, "int32"], B: T.Buffer[1, "int32"]):
      T.assume(B[0] == 0)

      if A[0] == B[0]:
          for i in T.serial(16):
              B[0] = B[0] + A[i]

  # After simplification
  @T.prim_func
  def func(A: T.Buffer[16, "int32"], B: T.Buffer[1, "int32"]):
      T.assume(B[0] == 0)

      # The first access of B[0] may be replaced with 0 using the
      # assumption.
      if A[0] == 0:
          # These later accesses of B[0] may not be replaced, because
          # for all loop iterations i!=0, the value stored in B[0] has
          # been overwritten since the T.assume call.
          for i in T.serial(16):
              B[0] = B[0] + A[i]
  ```

### New Transform - Hoist Expression

A new utility `HoistExpression`, which is a generalization of the
current `HoistIfThenElse` pass.  The transformation `HoistExpression`
would apply to the entire body of the `PrimFunc`, and would be used to
avoid duplication of functionality between `HoistIfThenElse` and
`HoistExpression`.

`HoistExpression` would also be exposed as a metaschedule primitive,
acting within a specified block of the `PrimFunc`, with the
configuration options given below.

```c++
enum class HoistConditional {
  kNone = 0,
  kIfElseStmt = (1<<0),
  kIfElseExpr = (1<<1),
  kBooleanExpression = (1<<2),
};

enum class HoistLetBinding {
  kNone = 0,
  kRequiredByCondition = (1<<0),
  kLetStmt = (1<<1),
  kLetExpr = (1m<<2),
};
```

* The values in `HoistConditional` are bit flags, indicating which
  conditionals should be hoisted.

  * `HoistConditional::kNone` - Do not hoist conditionals

  * `HoistConditional::kIfElseStmt` - If set, attempt to hoist
    conditionals that occur within `IfThenElseNode::condition`.

  * `HoistConditional::kIfElseExpr` - If set, attempt to hoist
    conditionals that occur as the condition of a
    `builtin::if_then_else` call.

  * `HoistConditional::kBooleanExpression` - If set, attempt to hoist
    any `PrimExpr` whose data type is `DataType::Bool()`.

* The values in `HoistLetBindings` are bit flags, indicating which
  bindings should be hoisted.

  * `HoistLetBinding::kNone` - Do not hoist any let bindings.

  * `HoistLetBinding::kRequiredByCondition` - If set, hoist a let
    binding if it is required in order to hoist a conditional.

  * `HoistLetBinding::kLetStmt = (1<<1)` - If set, attempt to hoist
    any let bindings performed using `LetStmt`.

  * `HoistLetBinding::kLetExpr` - If set, attempt to hoist any let
    bindings performed using `Let`.

The existing pass `HoistIfElse` is roughly equivalent to using
`HoistExpression` with `HoistConditional::kIfElseStmt` and
`HoistLetBinding::kNone`.  The one exception is that `HoistIfElse`
occurs after all let bindings have been inlined, and does not check
let bindings when determining if a condition can be hoisted.

```python
# Original function
@T.prim_func
def func(A: T.Buffer[(4,4), "float32"]):
    for i in T.serial(4):
        is_in_bounds = i < 3
        if is_in_bounds:
            A[i] = 0.0

# Incorrectly hoisted by `HoistIfThenElse`
@T.prim_func
def func(A: T.Buffer[(4,), "float32"]) -> None:
    is_in_bounds = T.var("bool")
    if is_in_bounds:
        for i in T.serial(4):
            is_in_bounds = i < 3
            A[i] = 0.0
```

### New Transform - Reduce Loop Extents

Reduce the extent of loops based on conditionals present in the body
of the loop.

For any non-vectorized `tir::For` loop (`ForKind::kSerial` or
`ForKind::kParallel`), if the body is a conditional and the
conditional's `else_case` is empty, determine if the expression is of
the form `(loop $CMP_OP const) && (...)`.  If so, use the comparison
operator to reduce the loop extent, such that loop skips values for
which the comparison is provably false.

TODO: Double-check that this isn't already implemented elsewhere.

TODO: Check if it is implementable using `IntSetAnalyzer`.

Below is an example of how this can work along-side `HoistExpression`
to simplify the initialization of padding.

```python
# Original function.
@T.prim_func
def func(A: T.Buffer[(4, 4), "float32"]):
    for i, j in T.grid(4, 4):
        if i == 0 and j < 2:
            A[i, j] = 0.0


# After hoisting with HoistConditional::kBooleanExpression
@T.prim_func
def func(A: T.Buffer[(4, 4), "float32"]):
    for i in T.serial(4):
        if i == 0:
            for j in T.serial(4):
                if j < 2:
                    A[i, j] = 0.0


# After reducing the extents of serial loops
@T.prim_func
def func(A: T.Buffer[(4, 4), "float32"]):
    i = 0
    for j in T.serial(2):
        A[i, j] = 0.0
```



### Utility - Merge Adjacent Loops

If it does not impact the resulting computation, loops may be merged
together.  This is a valid transformation if both loops are serial
loops, the loops have the same indices, and if the merging respects
data dependencies.  This would be exposed as a metaschedule primitive,
which takes input of the `LoopRV` to be merged.

For adjacent loops, to prove that there is no data dependency, two
conditions must hold.

1. For all loop indices `i` and `j` where `i > j`, the set of indices
   written by the first loop in iteration `i` is distinct from the set
   of indices accessed by the second loop in iteration `j`.  That is,
   merging the loops wouldn't cause the second loop body to read
   partial values, nor would it cause the first loop body to overwrite
   a value produced by the second loop body.

2. For all loop indices `i` and `j` where `i < j`, the set of indices
   read by the second loop in iteration `i` is distinct from the set
   of indices written by the second loop in iteration `j`.  That is,
   merging the loops wouldn't cause the second loop body to overwrite
   values that are still required by the first loop body.

Element-wise loops do not have any data dependencies, and adjacent
element-wise loops may be merged.

```python
# Before merging adjacent loops
@T.prim_func
def func(A: T.Buffer[(16,), "float32"]):
    for i in T.serial(16):
        A[i] = 0.0

    for i in T.serial(16):
        A[i] = 1.0


# 1. a. In iteration i, loop 1 writes to index [i].
#    b. In iteration j, loop 2 accesses index [j].
#    c. intersection([i], [j]) = [i] if i==j else [].
#    d. If i>j, the intersection is empty
#
# 2. a. In iteration i, loop 1 reads from index [].
#    b. In iteration j, loop 2 writes to index [j]
#    c. intersection([], [j]) = []
#    c. For all i,j, the intersection is empty
#
# Therefore, this merger is valid

# After merging adjacent loops
@T.prim_func
def func(A: T.Buffer[(16,), "float32"]):
    for i in T.serial(16):
        A[i] = 0.0
        A[i] = 1.0
```

The second loop may read indices that were written in an earlier
iteration.  Merging would not impact the result.

```python
# Before merging adjacent loops
@T.prim_func
def func(A: T.Buffer[(16,), "float32"]):
    for i in T.serial(16):
        A[i] = 0.0

    for i in T.serial(16):
        if i > 0:
            A[i] = A[i - 1] + 1.0


# 1. a. In iteration i, loop 1 writes to index [i].
#    b. In iteration j, loop 2 accesses index [j,j-1].
#    c. i>j implies that i!=j and i!=j-1.
#    c. For all i,j where i<j,
#
# 2. a. In iteration i, loop 1 reads from index [].
#    b. In iteration j, loop 2 writes to index [j]
#    c. For all i,j, intersection([], [j]) = [].
#
# Therefore, this merger is valid


# After merging adjacent loops
@T.prim_func
def func(A: T.Buffer[(16,), "float32"]):
    for i in T.serial(16):
        A[i] = 0.0
        if i > 0:
            A[i] = A[i - 1] + i
```

The second loop may not read indices that were written in a later
iteration of the first loop.  In this case, merging would impact the
output values.

```python
# Before merging adjacent loops
@T.prim_func
def func(A: T.Buffer[(16,), "float32"]):
    for i in T.serial(16):
        A[i] = i

    for i in T.serial(16):
        if 0 < i < 15:
            A[i] = A[i - 1] + A[i] + A[i + 1]


# 1. a. In iteration i, loop 1 writes to index [i].
#    b. In iteration j, loop 2 accesses index [j-1,j,j+1].
#    c. If i==j+1, then intersection([j+1], [j-1,j,j+1]) = [j+1],
#       which is non-empty.
#
# Therefore, this merger is not valid.
```

### New Primitive - Remove Branching Through Overcompute

A new transform which attempts to reduce branching by allowing
overcompute.  It takes an argument to specify which block it should be
applied within.

For each `IfThenElseStmt`, check if the
`IfThenElseStmtNode::else_case` is a simplified form of the
`IfThenElseStmtNode::then_case`.  This check is done by simplifying
`then_case`, under the assumption that `condition` is false, and
substituting the known value in a `BufferConstraint` in any
`BufferLoad` for which the predicate can be proven to be true.  If
this simplified form is identical to the `else_case`, then the entire
if/else block can be replaced with `then_case`.  Otherwise, this check
is repeated to see if the `then_case` can be simplified down to the
`else_case`.  If neither simplification holds, then no change is made.

For example, consider the following example.  This is a 1-d
convolution, where both the input and output buffers have a layout
transformation applied.

```python
# Original function
@T.prim_func
def func(
    A: T.Buffer[(16,), "float32"],
    F: T.Buffer[(3,), "float32"],
    B: T.Buffer[(14,), "float32"],
):
    with T.block('compute'):
        for i in T.serial(14):
            B[i] = 0.0
            for f in T.serial(3):
                B[i] = B[i] + A[i + f]


# sched.transform_layout(block='compute', buffer='A', lambda i: [i//4, i%4])
@T.prim_func
def func(
    A: T.Buffer[(4, 4), "float32"],
    F: T.Buffer[(3,), "float32"],
    B: T.Buffer[(14,), "float32"],
):
    with T.block('compute'):
        for i in T.serial(14):
            B[i] = 0.0
            for f in T.serial(3):
                B[i] = B[i] + A[(i + f) // 4, (i + f) % 4]


# sched.transform_layout(block='compute', buffer='B', lambda i: [i//4, i%4], pad_value=0.0)
@T.prim_func
def func(
    A: T.Buffer[(4, 4), "float32"],
    F: T.Buffer[(3,), "float32"],
    B: T.Buffer[(4, 4), "float32"],
):
    with T.block('compute'):
        for i in T.serial(14):
            B[i // 4, i % 4] = 0.0
            for f in T.serial(3):
                B[i // 4, i % 4] = B[i // 4, i % 4] + A[(i + f) // 4, (i + f) % 4]

        for io,ii in T.grid(4,4):
            if io==3 and ii>=2:
                B[io,ii] = 0.0


# sched.sequential_buffer_access(block='compute', buffer='B')
@T.prim_func
def func(
    A: T.Buffer[(4, 4), "float32"],
    F: T.Buffer[(3,), "float32"],
    B: T.Buffer[(4, 4), "float32"],
):
    with T.block('compute'):
        for io, ii in T.grid(4, 4):
            if 0 <= 4*io + ii < 14:
                B[io, ii] = 0.0
                for f in T.serial(3):
                    B[io, ii] = B[io, ii] + A[io + (ii + f) // 4, (ii + f) % 4]

        for io,ii in T.grid(4,4):
            if io==3 and ii>=2:
                B[io,ii] = 0.0
```


We'd like to remove the conditional `if 0 <= 4*io + ii < 14` in the
compute loop.  In order to do so, we need to prove that the body of
the conditional is a no-op in the case where the conditional is false.

Using the [updated `DomainTouched`
utility](#enhancement-remove-no-op), this else-block would be a no-op.
It is a write to `B[io,ii]` predicated on `4*io+ii >= 14`, followed by
a write to `B[io,ii]` predicated on `io==3 and ii>=2`, without a read
in between.  Since these predicates are equivalent, the first write is
a no-op.

```python
# sched.remove_branching_through_overcompute(block='compute')
@T.prim_func
def func(
    A: T.Buffer[(4, 4), "float32"],
    F: T.Buffer[(3,), "float32"],
    B: T.Buffer[(4, 4), "float32"],
):
    with T.block('compute'):
        for io, ii in T.grid(4, 4):
            B[io, ii] = 0.0
            for f in T.serial(3):
                B[io, ii] = B[io, ii] + A[io + (ii + f) // 4, (ii + f) % 4]

        for io,ii in T.grid(4,4):
            if io==3 and ii>=2:
                B[io,ii] = 0.0
```

### New Primitive - Remove Overcompute Through Branching

This is the reverse of [removing branching through
overcompute](#new-primitive-remove-branching-through-overcompute).
For each buffer access, insert a conditional based on the final value
of the buffer's padding, hoist the conditionals, and simplify.

TODO: Since branching is the default behavior, do we need the reverse
path?

```python
# Function with overcompute.  B has the constant value of zero in the
# buffer padding, located at io==3 and ii>=2.  This can be inferred
# from the schedule, by analysis of the "B_pad_value" block.
@T.prim_func
def func(
    A: T.Buffer[(4, 4), "float32"],
    F: T.Buffer[(3,), "float32"],
    B: T.Buffer[(4, 4), "float32"],
):
    with T.block('compute'):
        for io, ii in T.grid(4, 4):
            B[io, ii] = 0.0
            for f in T.serial(3):
                B[io, ii] = B[io, ii] + A[io + (ii + f) // 4, (ii + f) % 4]

    with T.block('B_pad_value'):
        for io,ii in T.grid(4,4):
            if io==3 and ii>=2:
                B[io,ii] = 0.0

# sched.reduce_overcompute_through_branching(block='compute')
#
# Step 1: Introduce conditionals
@T.prim_func
def func(
    A: T.Buffer[(4, 4), "float32"],
    F: T.Buffer[(3,), "float32"],
    B: T.Buffer[(4, 4), "float32"],
):
    for io, ii in T.grid(4, 4):
        if io==3 and ii>=2:
            B[io, ii] = 0.0
        else:
            B[io, ii] = 0.0

        for f in T.serial(3):
            if io==3 and ii>=2:
                B[io, ii] = 0.0
            else:
                B[io, ii] = B[io, ii] + A[io + (ii + f) // 4, (ii + f) % 4]

    for io,ii in T.grid(4,4):
        if io==3 and ii>=2:
            B[io,ii] = 0.0

# sched.reduce_overcompute_through_branching(block='compute')
#
# Step 2: Hoist conditionals
@T.prim_func
def func(
    A: T.Buffer[(4, 4), "float32"],
    F: T.Buffer[(3,), "float32"],
    B: T.Buffer[(4, 4), "float32"],
):
    for io, ii in T.grid(4, 4):
        if io==3 and ii>=2:
            B[io, ii] = 0.0

            for f in T.serial(3):
                B[io, ii] = 0.0

        else:
            B[io, ii] = 0.0

            for f in T.serial(3):
                B[io, ii] = B[io, ii] + A[io + (ii + f) // 4, (ii + f) % 4]

    for io,ii in T.grid(4,4):
        if io==3 and ii>=2:
            B[io,ii] = 0.0

# sched.reduce_overcompute_through_branching(block='compute')
#
# Step 3: Remove no-ops
@T.prim_func
def func(
    A: T.Buffer[(4, 4), "float32"],
    F: T.Buffer[(3,), "float32"],
    B: T.Buffer[(4, 4), "float32"],
):
    for io, ii in T.grid(4, 4):
        if io==3 and ii>=2:
            pass

        else:
            B[io, ii] = 0.0

            for f in T.serial(3):
                B[io, ii] = B[io, ii] + A[io + (ii + f) // 4, (ii + f) % 4]

    for io,ii in T.grid(4,4):
        if io==3 and ii>=2:
            B[io,ii] = 0.0

# sched.reduce_overcompute_through_branching(block='compute')
#
# Step 4: Simplify
@T.prim_func
def func(
    A: T.Buffer[(4, 4), "float32"],
    F: T.Buffer[(3,), "float32"],
    B: T.Buffer[(4, 4), "float32"],
):
    for io, ii in T.grid(4, 4):
        if io<3 or (io==3 and ii<2):
            B[io, ii] = 0.0

            for f in T.serial(3):
                B[io, ii] = B[io, ii] + A[io + (ii + f) // 4, (ii + f) % 4]

    for io,ii in T.grid(4,4):
        if io==3 and ii>=2:
            B[io,ii] = 0.0
```


### New Lowering Transform - Remove `T.assume`

This introduces a new lowering pass
`tir.transform.RemoveCompileTimeAssume`, which occurs at the start of
phase 1, and which replaces all `Call` nodes that use the
`tir::builtin::assume` with a no-op.

After this pass, the `PrimFunc` should not contain any calls to the
builtin `T.assume`.

### New Lowering Transform - Remove `T.undef`

This introduces a new lowering pass
`tir.transform.RemoveStoreUndef`, which occurs at the start of
phase 1.  For all `BufferStore` nodes, if the value being written
contains `T.undef()`, replace the store with a no-op.

After this pass, the `PrimFunc` should not contain any calls to the
builtin `T.undef()`.


## Implementation options

To ensure that this functionality is sufficient, testing whether
several different desired implementations can be written in terms of
these transformations.

For simplicity, these examples do not show the `T.block` grouping, and
all transformations act on the entire body unless otherwise specified.

### Never write to transformation padding

When applying a layout transform, use flow control or predicated
stores such that the buffer is only written to transformed indices
that can be represented in the logical indices.  This access would
otherwise occur when the loopnest surrounding a computation is
rewritten to be in row-major order of the transformed layout.

This is the default behavior for a producer when `pad_value` is
`None`.

```python
# Initial function
@T.prim_func
def func(A: T.Buffer[(14,), "int32"]):
    for i in T.serial(14):
        A[i] = 42


# sched.transform_layout(A, lambda i: [i//4, i%4])
@T.prim_func
def func(A: T.Buffer[(4, 4), "int32"]):
    for i in T.serial(14):
        A[i // 4, i % 4] = 42


# sched.sequential_buffer_access(A)
@T.prim_func
def func(A: T.Buffer[(4, 4), "int32"]):
    for io, ii in T.grid(4, 4):
        if 0 <= 4 * io + ii < 14:
            A[io, ii] = 42
```


### Never read from transformation padding

When applying a layout transform, use flow control or predicated
stores such that the buffer is only accessed at transformed indices
that can be represented in the logical indices.  This access would
otherwise occur when the loopnest surrounding a computation is
rewritten to be in row-major order of the transformed layout.

This is the default behavior for a consumer when `pad_value` is
`None`, or when no simplifications exist that can make use of the
`pad_value`.


```python
# Initial function
@T.prim_func
def func(A: T.Buffer[(14,), "int32"], B: T.Buffer[(1,), "int32"]):
    B[0] = 0
    for i in T.serial(14):
        B[0] = B[0] + A[i]

# sched.transform_layout(A, lambda i: [i//4, i%4])
@T.prim_func
def func(A: T.Buffer[(4,4), "int32"], B: T.Buffer[(1,), "int32"]):
    B[0] = 0
    for i in T.serial(14):
        B[0] = B[0] + A[i//4, i%4]


# sched.sequential_buffer_access(A)
@T.prim_func
def func(A: T.Buffer[(4,4), "int32"], B: T.Buffer[(1,), "int32"]):
    B[0] = 0
    for io,ii in T.grid(4,4):
        if 4*io + ii < 14:
            B[0] = B[0] + A[io, ii]
```



### Allocate internal buffer containing transformation padding

When applying a layout transformation, allocate an internal buffer
whose shape is the transformed input shape.  Each value in the
internal buffer is initialized to corresponding value in the input
buffer if it is within the logical extents of the input, or to the
operation's desired default otherwise.  All access of the input values
are then done through the internal allocation.

This does not insert any `T.assume` statements, because the pad value
can be inferred from the TIR graph.

```python
# Initial function
@T.prim_func
def func(A: T.Buffer[(14,), "int32"], B: T.Buffer[(1,), "int32"]):
    B[0] = 0
    for i in T.serial(14):
        B[0] = B[0] + A[i]

# sched.cache_read(A, "local")
@T.prim_func
def func(A: T.Buffer[(14,), "int32"], B: T.Buffer[(1,), "int32"]) -> None:
    A_local = T.alloc_buffer([14], dtype="int32", scope="local")
    for i in T.serial(14):
        A_local[i] = A[i]

    B[0] = 0
    for i in T.serial(14):
        B[0] = B[0] + A_local[i]

# sched.transform_layout(A_local, lambda i: [i//4, i%4], pad_value=0)
@T.prim_func
def func(A: T.Buffer[(14,), "int32"], B: T.Buffer[(1,), "int32"]) -> None:
    A_local = T.alloc_buffer([4,4], dtype="int32", scope="local")
    for i in T.serial(14):
        A_local[i//4, i%4] = A[i]

    for io,ii in T.grid(4,4):
        if io==3 and ii>=2:
            A_local[io,ii] = 0

    B[0] = 0
    for i in T.serial(14):
        B[0] = B[0] + A_local[i//4, i%4]


# sched.sequential_buffer_access(A_local)
@T.prim_func
def func(A: T.Buffer[(14,), "int32"], B: T.Buffer[(1,), "int32"]) -> None:
    A_local = T.alloc_buffer([4,4], dtype="int32", scope="local")
    for io,ii in T.grid(4,4):
        if 4*io+ii < 14:
            A_local[io,ii] = A[4*io+ii]

    for io,ii in T.grid(4,4):
        if io==3 and ii>=2:
            A_local[io,ii] = 0

    B[0] = 0
    for io,ii in T.grid(4,4):
        if 4*io+ii < 14:
            B[0] = B[0] + A_local[io,ii]

# sched.remove_branching_through_overcompute()
@T.prim_func
def func(A: T.Buffer[(14,), "int32"], B: T.Buffer[(1,), "int32"]) -> None:
    A_local = T.alloc_buffer([4,4], dtype="int32", scope="local")
    for io,ii in T.grid(4,4):
        if 4*io+ii < 14:
            A_local[io,ii] = A[4*io+ii]

    for io,ii in T.grid(4,4):
        if io==3 and ii>=2:
            A_local[io,ii] = 0

    B[0] = 0
    for io,ii in T.grid(4,4):
        B[0] = B[0] + A_local[io,ii]
```



### Explicitly write next operator's desired default at end of function

If a layout transformation is applied, the writer is required to fill
the transformation padding with the default value.  In the example below,
`consumer` does not require a conditional to validate the indices
provided, because `producer` has filled the transformation padding with
zero.

```python
# Initial producer/consumer
@script.ir_module
class MyModule:
    @T.prim_func
    def producer(A: T.Buffer[(14,), "int32"]):
        for i in T.serial(14):
            A[i] = 1000 + i

    @T.prim_func
    def consumer(A: T.Buffer[(14,), "int32"], B: T.Buffer[(1,), "int32"]):
        B[0] = 0
        for i in T.serial(14):
            B[0] = B[0] + A[i]


# sched.transform_layout(A, lambda i: [i//4, i%4], pad_value=0)
@script.ir_module
class MyModule:
    @T.prim_func
    def producer(A: T.Buffer[(4, 4), "int32"]):
        for i in T.serial(14):
            A[i // 4, i % 4] = 1000 + i

        for io, ii in T.grid(4, 4):
            if 4 * io + ii >= 14:
                A[io, ii] = 0

    @T.prim_func
    def consumer(A: T.Buffer[(4, 4), "int32"], B: T.Buffer[(1,), "int32"]):
        for io, ii in T.grid(4, 4):
            T.assume(4 * io + ii < 14 or A[io, ii] == 0)

        B[0] = 0
        for i in T.serial(14):
            B[0] = B[0] + A[i // 4, i % 4]


# sched.sequential_buffer_access(A)
@script.ir_module
class MyModule:
    @T.prim_func
    def producer(A: T.Buffer[(4, 4), "int32"]):
        for io, ii in T.grid(4, 4):
            if 4 * io + ii < 14:
                A[io, ii] = 1000 + i

        for io, ii in T.grid(4, 4):
            if 4 * io + ii >= 14:
                A[io, ii] = 0

    @T.prim_func
    def consumer(A: T.Buffer[(4, 4), "int32"], B: T.Buffer[(1,), "int32"]):
        for io, ii in T.grid(4, 4):
            T.assume(4 * io + ii < 14 or A[io, ii] == 0)

        B[0] = 0
        for io, ii in T.grid(4, 4):
            if 4 * io + ii < 14:
                B[0] = B[0] + A[io, ii]


# sched.merge_adjacent_loops()
@script.ir_module
class MyModule:
    @T.prim_func
    def producer(A: T.Buffer[(4, 4), "int32"]):
        for io, ii in T.grid(4, 4):
            if 4 * io + ii < 14:
                A[io, ii] = 1000 + i
            if 4 * io + ii >= 14:
                A[io, ii] = 0

    @T.prim_func
    def consumer(A: T.Buffer[(4, 4), "int32"], B: T.Buffer[(1,), "int32"]):
        for io, ii in T.grid(4, 4):
            T.assume(4 * io + ii < 14 or A[io, ii] == 0)

        B[0] = 0
        for io, ii in T.grid(4, 4):
            if 4 * io + ii < 14:
                B[0] = B[0] + A[io, ii]


# Simplify
@script.ir_module
class MyModule:
    @T.prim_func
    def producer(A: T.Buffer[(4, 4), "int32"]):
        for io, ii in T.grid(4, 4):
            if 4 * io + ii < 14:
                A[io, ii] = 1000 + i
            else:
                A[io, ii] = 0

    @T.prim_func
    def consumer(A: T.Buffer[(4, 4), "int32"], B: T.Buffer[(1,), "int32"]):
        for io, ii in T.grid(4, 4):
            T.assume(4 * io + ii < 14 or A[io, ii] == 0)

        B[0] = 0
        for io, ii in T.grid(4, 4):
            if 4 * io + ii < 14:
                B[0] = B[0] + A[io, ii]


# sched.remove_branching_through_overcompute()
@script.ir_module
class MyModule:
    @T.prim_func
    def producer(A: T.Buffer[(4, 4), "int32"]):
        for io, ii in T.grid(4, 4):
            if 4 * io + ii < 14:
                A[io, ii] = 1000 + i
            else:
                A[io, ii] = 0

    @T.prim_func
    def consumer(A: T.Buffer[(4, 4), "int32"], B: T.Buffer[(1,), "int32"]):
        for io, ii in T.grid(4, 4):
            T.assume(4 * io + ii < 14 or A[io, ii] == 0)

        B[0] = 0
        for io, ii in T.grid(4, 4):
            B[0] = B[0] + A[io, ii]
```

This requires both implementations to use the same `pad_value = 0`, so
that the producer writes the value out, and the consumer can simplify
the conditional using the pad_value.

This also means that the available simplifications depends on the pad
value.  When deciding the pad value, these constraints may flow from
consumer to producer.  If `consumer` computed the product of an array,
rather than the sum, then `producer` would need to write `1` as the
default value instead of `0`, so that the simplification at the end
can work.

### Implicitly write default value of next operator

This has the same constraints as the [previous
section](#explicitly-write-next-operators-desired-default-at-end-of-function),
that we have a producer writing a default value to the transformation
padding and a consumer that assumes that default value (i.e. semantics
of `default(0)`).  However, for some operators and default values, the
loop to set the default value is unnecessary.

As an example, consider a 1-d buffer `A` with logical shape `[16]`,
using a filter `F` of size `[3]`, and using padding of `2`
on both sides.  Ignoring the batch and channel dimenions, this is the
same computation as `tvm.topi.nn.conv1d(A, F, padding=2)`, and will
produce an output `B` with shape `[18]`.

```python
# Initial compute definition
@T.prim_func
def func(
    A: T.Buffer[(16,), "float32"],
    F: T.Buffer[(3,), "float32"],
    B: T.Buffer[(18,), "float32"],
):
    for Bi in T.serial(18):
        B[Bi] = 0.0
        for fi in T.serial(3):
            Ai = Bi - fi + 2
            if 0 <= Ai < 16:
                B[Bi] = B[Bi] + F[fi] * A[Ai]
```

If the input has layout transformation `sched.transform_layout(A,
lambda i: [(i+2)//8, (i+2)%8], pad_value=0)`, the transformed input
shape is `[3, 8]`, with transformation padding inserted at transformed
indices `[(0, i) for i in range(0, 2)]` and at `[(2, i) for i in
range(2,8)]`.

```
       ┌─A, Logical-index-space──────────────────────┐
       │                                             │
┌──┬──┬▼─┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬─▼┬──┬──┬──┬──┬──┬──┐
│  │  │00│01│02│03│04│05│06│07│08│09│10│11│12│13│14│15│  │  │  │  │  │  │
│00│01│02│03│04│05│06│07│08│09│10│11│12│13│14│15│16│17│18│19│20│21│22│23│
└▲─┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴─▲┘
 │                                                                     │
 └─A, Physical-index-space-────────────────────────────────────────────┘

 ┌─A, Transformed-index-space──────────────────┐
 │                                             │
 │      ┌────┬────┬────┬────┬────┬────┬────┬───▼┐
 │      │    │    │ 00 │ 01 │ 02 │ 03 │ 04 │ 05 │
 │      │ 00 │ 01 │ 02 │ 03 │ 04 │ 05 │ 06 │ 07 │
 │      ├────┼────┼────┼────┼────┼────┼────┼────┤
 │      │ 06 │ 07 │ 08 │ 09 │ 10 │ 11 │ 12 │ 13 │
 │      │ 08 │ 09 │ 10 │ 11 │ 12 │ 13 │ 14 │ 15 │
 │      ├────┼────┼────┼────┼────┼────┼────┼────┤
 │      │ 14 │ 15 │    │    │    │    │    │    │
 └──────► 16 │ 17 │ 18 │ 19 │ 20 │ 21 │ 22 │ 23 │
        └────┴────┴────┴────┴────┴────┴────┴────┘
```


```python
# sched.transform_layout(A, lambda i: [(i+2)//8, (i+2)%8], pad_value=0)
@T.prim_func
def func(
    A: T.Buffer[(3, 8), "float32"],
    F: T.Buffer[(3,), "float32"],
    B: T.Buffer[(18,), "float32"],
):
    for io, ii in T.grid(3, 8):
        T.assume(
            (io == 0 and ii >= 2)
            or (io > 0 and io < 2)
            or (io == 2 and ii < 2)
            or A[io, ii] == 0.0
        )

    for Bi in T.serial(18):
        B[Bi] = 0.0
        for fi in T.serial(3):
            Ai = Bi - fi + 2
            if 0 <= Ai < 16:
                B[Bi] = B[Bi] + F[fi] * A[(Ai + 2) // 8, (Ai + 2) % 8]
```

We'll apply the same layout transformation to `B`.  Even though it has
a different logical shape `[18]`, it has the same transformed shape
`[3,8]`.

```
       ┌─B, Logical-index-space──────────────────────────────────┐
       │                                                         │
┌──┬──┬▼─┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬─▼┬──┬──┐
│  │  │00│01│02│03│04│05│06│07│08│09│10│11│12│13│14│15│16│17│18│19│  │  │
│00│01│02│03│04│05│06│07│08│09│10│11│12│13│14│15│16│17│18│19│20│21│22│23│
└▲─┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴─▲┘
 │                                                                     │
 └─B, Physical-index-space-────────────────────────────────────────────┘

 ┌─B, Transformed-index-space──────────────────┐
 │                                             │
 │      ┌────┬────┬────┬────┬────┬────┬────┬───▼┐
 │      │    │    │ 00 │ 01 │ 02 │ 03 │ 04 │ 05 │
 │      │ 00 │ 01 │ 02 │ 03 │ 04 │ 05 │ 06 │ 07 │
 │      ├────┼────┼────┼────┼────┼────┼────┼────┤
 │      │ 06 │ 07 │ 08 │ 09 │ 10 │ 11 │ 12 │ 13 │
 │      │ 08 │ 09 │ 10 │ 11 │ 12 │ 13 │ 14 │ 15 │
 │      ├────┼────┼────┼────┼────┼────┼────┼────┤
 │      │ 14 │ 15 │ 16 │ 17 │    │    │    │    │
 └──────► 16 │ 17 │ 18 │ 19 │ 20 │ 21 │ 22 │ 23 │
        └────┴────┴────┴────┴────┴────┴────┴────┘
```

```python
# sched.transform_layout(B, lambda i: [(i+2)//8, (i+2)%8], pad_value=0)
@T.prim_func
def func(
    A: T.Buffer[(3, 8), "float32"],
    F: T.Buffer[3, "float32"],
    B: T.Buffer[(3, 8), "float32"],
):
    for io, ii in T.grid(3, 8):
        T.assume(
            (io == 0 and ii >= 2)
            or (io > 0 and io < 2)
            or (io == 2 and ii < 2)
            or A[io, ii] == 0.0
        )

    for Bi in T.serial(18):
        B[(Bi + 2) // 8, (Bi + 2) % 8] = 0.0
        for fi in T.serial(3):
            Ai = Bi - fi + 2
            if 0 <= Ai < 16:
                B[(Bi + 2) // 8, (Bi + 2) % 8] = (
                    B[(Bi + 2) // 8, (Bi + 2) % 8]
                    + F[fi] * A[(Ai + 2) // 8, (Ai + 2) % 8]
                )

    for io, ii in T.grid(3, 8):
        if (io == 0 and ii < 2) or (io == 2 and ii >= 6):
            B[io, ii] = 0.0
```

By rewriting the loop iterators to be in terms of `A` and hoisting
conditionals, we can express this in terms of a slow loop with
conditionals to handle the array borders and a fast loop that doesn't
contain any conditionals.

```python
# sched.sequential_buffer_access(A)
@T.prim_func
def func(
    A: T.Buffer[(3, 8), "float32"],
    F: T.Buffer[3, "float32"],
    B: T.Buffer[(3, 8), "float32"],
):
    for io, ii in T.grid(3, 8):
        T.assume(
            (io == 0 and ii >= 2)
            or (io > 0 and io < 2)
            or (io == 2 and ii < 2)
            or A[io, ii] == 0.0
        )

    for Bi in T.serial(18):
        B[(Bi + 2) // 8, (Bi + 2) % 8] = 0.0

    for io, ii in T.grid(3, 8):
        # Ai = 8*io + ii - 2
        for fi in T.serial(3):
            # Bi = Ai + fi - 2
            # Bi = 8*io + ii + fi - 2
            if 0 <= 8 * io + ii - 2 < 16 and 0 <= 8 * io + ii + fi - 2 < 18:
                # (Bi+2)//8 = (8*io + ii + fi - 2 + 2)//8 = io + (ii+fi)//8
                # (Bi+2)%8 = (8*io + ii + fi - 2 + 2)%8 = (ii+fi)//8
                B[io + (ii + fi) // 8, (ii + fi) % 8] = (
                    B[io + (ii + fi) // 8, (ii + fi) % 8] + F[fi] * A[io, ii]
                )

    for io, ii in T.grid(3, 8):
        if (io == 0 and ii < 2) or (io == 2 and ii >= 6):
            B[io, ii] = 0.0


# sched.sequential_buffer_access(B), only on the first block
@T.prim_func
def func(
    A: T.Buffer[(3, 8), "float32"],
    F: T.Buffer[3, "float32"],
    B: T.Buffer[(3, 8), "float32"],
):
    for io, ii in T.grid(3, 8):
        T.assume(
            (io == 0 and ii >= 2)
            or (io > 0 and io < 2)
            or (io == 2 and ii < 2)
            or A[io, ii] == 0.0
        )

    for io, ii in T.grid(3, 8):
        Bi = 8 * io + ii - 2
        if 0 <= Bi < 18:
            B[io, ii] = 0.0

    for io, ii in T.grid(3, 8):
        for fi in T.serial(3):
            if 0 <= 8 * io + ii - 2 < 16 and 0 <= 8 * io + ii + fi - 2 < 18:
                B[io + (ii + fi) // 8, (ii + fi) % 8] = (
                    B[io + (ii + fi) // 8, (ii + fi) % 8] + F[fi] * A[io, ii]
                )

    for io, ii in T.grid(3, 8):
        if (io == 0 and ii < 2) or (io == 2 and ii >= 6):
            B[io, ii] = 0.0


# TODO: Not technically adjacent, but the main compute loop doesn't
# touch any of the indices in-between.
#
# Option: Explicitly search for complementary conditions in neighboring loops?
#
# Option: Write the padding values before the compute instead of
# after.  Would probably need to be a scheduling option.
#
# Option: Write the padding both before and after the compute.  That
# way, the compute loop could take advantage of either one.  Then, if
# no other simplifications happen, new block-based analysis in
# ReduceNoOp to remove would remove the duplicate.

# sched.merge_adjacent_loops()
# sched.merge_complementary_conditions()
@T.prim_func
def func(
    A: T.Buffer[(3, 8), "float32"],
    F: T.Buffer[3, "float32"],
    B: T.Buffer[(3, 8), "float32"],
):
    for io, ii in T.grid(3, 8):
        T.assume(
            (io == 0 and ii >= 2)
            or (io > 0 and io < 2)
            or (io == 2 and ii < 2)
            or A[io, ii] == 0.0
        )

    for io, ii in T.grid(3, 8):
        B[io, ii] = 0.0

    for io, ii in T.grid(3, 8):
        for fi in T.serial(3):
            if 0 <= 8 * io + ii - 2 < 16 and 0 <= 8 * io + ii + fi - 2 < 18:
                B[io + (ii + fi) // 8, (ii + fi) % 8] = (
                    B[io + (ii + fi) // 8, (ii + fi) % 8] + F[fi] * A[io, ii]
                )


# Not technically a transform, but rewriting the conditionals for
# better visibility.
#
# TODO: Define if/how this should be done as part of
# buffer_sequential_access
#
# Option: Express the padding predicate as an OR of the
# pre-transformation and post-transformation versions (i.e. `((io==0)
# and (ii<2)) or (4*io+ii<2)`), so that it can be identified
# by the simplifier based on expressions in terms of either the
# pre- or post-transformation axes.


@T.prim_func
def func(
    A: T.Buffer[(3, 8), "float32"],
    F: T.Buffer[3, "float32"],
    B: T.Buffer[(3, 8), "float32"],
):
    for io, ii in T.grid(3, 8):
        T.assume(
            (io == 0 and ii >= 2)
            or (io > 0 and io < 2)
            or (io == 2 and ii < 2)
            or A[io, ii] == 0.0
        )

    for io, ii in T.grid(3, 8):
        B[io, ii] = 0.0

    for io, ii in T.grid(3, 8):
        for fi in T.serial(3):
            if (
                (not (io == 0 and ii < 2))
                and (not (io == 2 and ii >= 2))
                and (not (io == 0 and ii + fi < 2))
                and (not (io == 2 and ii + fi >= 4))
            ):
                B[io + (ii + fi) // 8, (ii + fi) % 8] = (
                    B[io + (ii + fi) // 8, (ii + fi) % 8] + F[fi] * A[io, ii]
                )


# sched.hoist_expressions('boolean_expressions')
# Part 1, hoisting conditions on io
@T.prim_func
def func(
    A: T.Buffer[(3, 8), "float32"],
    F: T.Buffer[3, "float32"],
    B: T.Buffer[(3, 8), "float32"],
):
    for io, ii in T.grid(3, 8):
        T.assume(
            (io == 0 and ii >= 2)
            or (io > 0 and io < 2)
            or (io == 2 and ii < 2)
            or A[io, ii] == 0.0
        )

    for io, ii in T.grid(3, 8):
        B[io, ii] = 0.0

    for io in T.serial(3):
        if io == 0:
            for ii in T.serial(8):
                for fi in T.serial(3):
                    if (not (ii < 2)) and (not (ii + fi < 2)):
                        B[io + (ii + fi) // 8, (ii + fi) % 8] = (
                            B[io + (ii + fi) // 8, (ii + fi) % 8] + F[fi] * A[io, ii]
                        )
        elif io == 2:
            for ii in T.serial(8):
                for fi in T.serial(3):
                    if (not (ii >= 2)) and (not (ii + fi >= 4)):
                        B[io + (ii + fi) // 8, (ii + fi) % 8] = (
                            B[io + (ii + fi) // 8, (ii + fi) % 8] + F[fi] * A[io, ii]
                        )
        else:
            for ii in T.serial(8):
                for fi in T.serial(3):
                    B[io + (ii + fi) // 8, (ii + fi) % 8] = (
                        B[io + (ii + fi) // 8, (ii + fi) % 8] + F[fi] * A[io, ii]
                    )


# sched.hoist_expressions('boolean_expressions')
# Part 2, simplifying for readability
@T.prim_func
def func(
    A: T.Buffer[(3, 8), "float32"],
    F: T.Buffer[3, "float32"],
    B: T.Buffer[(3, 8), "float32"],
):
    for io, ii in T.grid(3, 8):
        T.assume(
            (io == 0 and ii >= 2)
            or (io > 0 and io < 2)
            or (io == 2 and ii < 2)
            or A[io, ii] == 0.0
        )

    for io, ii in T.grid(3, 8):
        B[io, ii] = 0.0

    for io in T.serial(3):
        if io == 0:
            for ii in T.serial(8):
                for fi in T.serial(3):
                    if (ii >= 2) or (ii + fi >= 2):
                        B[io + (ii + fi) // 8, (ii + fi) % 8] = (
                            B[io + (ii + fi) // 8, (ii + fi) % 8] + F[fi] * A[io, ii]
                        )
        elif io == 2:
            for ii in T.serial(8):
                for fi in T.serial(3):
                    if (ii < 2) or (ii + fi < 4):
                        B[io + (ii + fi) // 8, (ii + fi) % 8] = (
                            B[io + (ii + fi) // 8, (ii + fi) % 8] + F[fi] * A[io, ii]
                        )
        else:
            for ii in T.serial(8):
                for fi in T.serial(3):
                    B[io + (ii + fi) // 8, (ii + fi) % 8] = (
                        B[io + (ii + fi) // 8, (ii + fi) % 8] + F[fi] * A[io, ii]
                    )


# sched.hoist_expressions('boolean_expressions')
# Part 3, hoist ii condition
@T.prim_func
def func(
    A: T.Buffer[(3, 8), "float32"],
    F: T.Buffer[3, "float32"],
    B: T.Buffer[(3, 8), "float32"],
):
    for io, ii in T.grid(3, 8):
        T.assume(
            (io == 0 and ii >= 2)
            or (io > 0 and io < 2)
            or (io == 2 and ii < 2)
            or A[io, ii] == 0.0
        )

    for io, ii in T.grid(3, 8):
        B[io, ii] = 0.0

    for io in T.serial(3):
        if io == 0:
            for ii in T.serial(8):
                if ii >= 2:
                    for fi in T.serial(3):
                        B[io + (ii + fi) // 8, (ii + fi) % 8] = (
                            B[io + (ii + fi) // 8, (ii + fi) % 8] + F[fi] * A[io, ii]
                        )
                else:
                    for fi in T.serial(3):
                        if ii + fi >= 2:
                            B[io + (ii + fi) // 8, (ii + fi) % 8] = (
                                B[io + (ii + fi) // 8, (ii + fi) % 8]
                                + F[fi] * A[io, ii]
                            )
        elif io == 2:
            for ii in T.serial(8):
                if ii < 2:
                    for fi in T.serial(3):
                        B[io + (ii + fi) // 8, (ii + fi) % 8] = (
                            B[io + (ii + fi) // 8, (ii + fi) % 8] + F[fi] * A[io, ii]
                        )
                else:
                    for fi in T.serial(3):
                        if ii + fi < 4:
                            B[io + (ii + fi) // 8, (ii + fi) % 8] = (
                                B[io + (ii + fi) // 8, (ii + fi) % 8]
                                + F[fi] * A[io, ii]
                            )
        else:
            for ii in T.serial(8):
                for fi in T.serial(3):
                    B[io + (ii + fi) // 8, (ii + fi) % 8] = (
                        B[io + (ii + fi) // 8, (ii + fi) % 8] + F[fi] * A[io, ii]
                    )
```

So far, the transformations have used the indices defined in the
transformation, but haven't taken significant advanced of the
assumptions provided for the padding in the input `A`.  These can be
used by simplifying the expression.

```python
# sched.simplify()
#
# Within the if block of `io==0` and the else block of `ii >= 2`,
# `A[io,ii]` is accessed for `io==0` and `ii<2`.  With these indices,
# the `T.assume` statement simplifies to `A[io,ii] == 0.0`.  After
# substituting this known value in, the entire body simplifies to a
# no-op in the else-case, and the conditional can be removed without
# impacting the result.
#
# Similarly, within the if block of `io==2` and the else block of `ii
# < 2`, `A[io,ii]` is accessed for `io==2` and `ii>=2`.  With these
# indices, the `T.assume` statement also simplifies to `A[io,ii] ==
# 0.0`, and results in simplification of that block to a no-op.


@T.prim_func
def func(
    A: T.Buffer[(3, 8), "float32"],
    F: T.Buffer[3, "float32"],
    B: T.Buffer[(3, 8), "float32"],
):
    for io, ii in T.grid(3, 8):
        T.assume(
            (io == 0 and ii >= 2)
            or (io > 0 and io < 2)
            or (io == 2 and ii < 2)
            or A[io, ii] == 0.0
        )

    for io, ii in T.grid(3, 8):
        B[io, ii] = 0.0

    for io in T.serial(3):
        if io == 0:
            for ii in T.serial(8):
                if ii >= 2:
                    for fi in T.serial(3):
                        B[io + (ii + fi) // 8, (ii + fi) % 8] = (
                            B[io + (ii + fi) // 8, (ii + fi) % 8] + F[fi] * A[io, ii]
                        )
        elif io == 2:
            for ii in T.serial(8):
                if ii < 2:
                    for fi in T.serial(3):
                        B[io + (ii + fi) // 8, (ii + fi) % 8] = (
                            B[io + (ii + fi) // 8, (ii + fi) % 8] + F[fi] * A[io, ii]
                        )
        else:
            for ii in T.serial(8):
                for fi in T.serial(3):
                    B[io + (ii + fi) // 8, (ii + fi) % 8] = (
                        B[io + (ii + fi) // 8, (ii + fi) % 8] + F[fi] * A[io, ii]
                    )
```

If we introduce overcompute, we can simplify even further.  Similar to
the simplifications, proving that these changes are valid also relies
on the information provided by `T.assume`.

In order to remove the `ii >= 2` conditional, we must prove that an
else block with the same body would be a no-op.  We already know that
`io==0` in order to reach the conditional, and entering the else block
would require that `ii < 2`.  Using these two conditions, our
assumption tells us that `A[io,ii] == 0.0`, which can be used to
simplify the newly introduced else block down to a no-op.  Therefore,
the `ii >= 2` conditional does not need to be checked.

Removing the `ii < 2` conditional is also possible, but requires an
additional modification.  The same logic shows that the within the
inserted else block, `io==2` and `ii>=2`, so `A[io,ii]==0.0`.  This
allows the expression to be simplified to an access of `B[2 + (ii +
fi) // 8, (ii + fi) % 8]`.  However, this is only a no-op if the
indices are in-bounds, and could cause a segfault at runtime if the
access is out-of-bounds.  For some cases, `ii + fi >= 8`, which would
cause an out-of-bounds access of `B[3, _]`.  To avoid this, we can
modify the accessed indices to be reduced mod 3.  We don't need any
specific locations to be accessed, only for the access to be valid.

TODO: Motivate this modification.  Maybe all buffer access gets the
`floormod(i, axis_size)`, which can be simplified out wherever it ise
necessary?

```python
# sched.remove_branching_through_overcompute()
#
# Part 1, remove conditional on (io==0 and ii>=2), and remove
# conditional on (io==2 and ii<2)
@T.prim_func
def func(
    A: T.Buffer[(3, 8), "float32"],
    F: T.Buffer[3, "float32"],
    B: T.Buffer[(3, 8), "float32"],
):
    for io, ii in T.grid(3, 8):
        T.assume(
            (io == 0 and ii >= 2)
            or (io > 0 and io < 2)
            or (io == 2 and ii < 2)
            or A[io, ii] == 0.0
        )

    for io, ii in T.grid(3, 8):
        B[io, ii] = 0.0

    for io in T.serial(3):
        if io == 0:
            for ii in T.serial(8):
                for fi in T.serial(3):
                    B[io + (ii + fi) // 8, (ii + fi) % 8] = (
                        B[io + (ii + fi) // 8, (ii + fi) % 8] + F[fi] * A[io, ii]
                    )
        elif io == 2:
            for ii in T.serial(8):
                for fi in T.serial(3):
                    B[(io + (ii + fi) // 8) % 3, (ii + fi) % 8] = (
                        B[(io + (ii + fi) // 8) % 3, (ii + fi) % 8] + F[fi] * A[io, ii]
                    )
        else:
            for ii in T.serial(8):
                for fi in T.serial(3):
                    B[io + (ii + fi) // 8, (ii + fi) % 8] = (
                        B[io + (ii + fi) // 8, (ii + fi) % 8] + F[fi] * A[io, ii]
                    )


# sched.remove_branching_through_overcompute()
#
# Part 2, remove conditional on io

@T.prim_func
def func(
    A: T.Buffer[(3, 8), "float32"],
    F: T.Buffer[3, "float32"],
    B: T.Buffer[(3, 8), "float32"],
):
    for io, ii in T.grid(3, 8):
        T.assume(
            (io == 0 and ii >= 2)
            or (io > 0 and io < 2)
            or (io == 2 and ii < 2)
            or A[io, ii] == 0.0
        )

    for io, ii in T.grid(3, 8):
        B[io, ii] = 0.0

    for io in T.serial(3):
        for ii in T.serial(8):
            for fi in T.serial(3):
                B[(io + (ii + fi) // 8) % 3, (ii + fi) % 8] = (
                    B[(io + (ii + fi) // 8) % 3, (ii + fi) % 8] + F[fi] * A[io, ii]
                )
```

Other than the addition of an additional `floormod`, this is identical
to our fast loop, but can be applied to all locations across the
buffer, avoiding conditionals altogether.

Intuitively, this takes advantage of the fact that a filter applied to
a window containing only zero will have an output of zero.  As a
result, the physical indices in `B` located at `[(0,i) for i in
range(0,2)]` and at `[(2,i) for i in range(6,8)]` will be zero,
because they correspond to the filter being entirely within the
transformation padding.

Furthermore, this same convolution can then be applied with input `B`
to produce output `C` with logical shape `[20]`.  The transformation
padding in `B` doesn't have any
contiguous regions that are as large as the filter, so there are no
output physical indices that must be zero.  Using
`s[C].transform_layout(lambda i: [(i+2)//8, (i+2)%8])` would result in
a transformed shape of `[3,8]`.

```
 ┌─C, Logical-index-space──────────────────────────────────────────────┐
 │                                                                     │
┌▼─┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬─▼┐
│  │  │00│01│02│03│04│05│06│07│08│09│10│11│12│13│14│15│16│17│18│19│  │  │
│00│01│02│03│04│05│06│07│08│09│10│11│12│13│14│15│16│17│18│19│20│21│22│23│
└▲─┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴─▲┘
 │                                                                     │
 └─C, Physical-index-space-────────────────────────────────────────────┘

 ┌─C, Transformed-index-space──────────────────┐
 │                                             │
 │      ┌────┬────┬────┬────┬────┬────┬────┬───▼┐
 │      │    │    │ 00 │ 01 │ 02 │ 03 │ 04 │ 05 │
 │      │ 00 │ 01 │ 02 │ 03 │ 04 │ 05 │ 06 │ 07 │
 │      ├────┼────┼────┼────┼────┼────┼────┼────┤
 │      │ 06 │ 07 │ 08 │ 09 │ 10 │ 11 │ 12 │ 13 │
 │      │ 08 │ 09 │ 10 │ 11 │ 12 │ 13 │ 14 │ 15 │
 │      ├────┼────┼────┼────┼────┼────┼────┼────┤
 │      │ 14 │ 15 │ 16 │ 17 │ 18 │ 19 │    │    │
 └──────► 16 │ 17 │ 18 │ 19 │ 20 │ 21 │ 22 │ 23 │
        └────┴────┴────┴────┴────┴────┴────┴────┘
```

### Apply operator element-wise over the transformation padding

For some operators, particularly element-wise operators, values loaded
from the transformation padding of the input would only influence the
values stored to the transformation padding of the output.  In the
example below, the same transformation is applied to input buffer `A`
and output buffer `B`.

```python
# Initial function
@T.prim_func
def func(A: T.Buffer[(14,), "int32"], B: T.Buffer[(14,), "int32"]):
    for i in T.serial(14):
        B[i] = 2 * A[i]


# sched.transform_layout(A, lambda i: [i//4, i%4], pad_value=lambda io,ii: tir.undef())
@T.prim_func
def func(A: T.Buffer[(4, 4), "int32"], B: T.Buffer[(14,), "int32"]):
    # This assumption doesn't tell us anything about the value of A,
    # but does tell us that it is valid to read these locations of A,
    # and won't contain uninitialized values, which could result in
    # undefined behavior on some targets.
    for io,ii in T.grid(4,4):
        T.assume(4*io+ii < 14 or A[io,ii] == T.undef())

    for i in T.serial(14):
        B[i] = 2 * A[i // 4, i % 4]


# sched.transform_layout(B, lambda i: [i//4, i%4], pad_value=lambda io,ii: tir.undef())
@T.prim_func
def func(A: T.Buffer[(4, 4), "int32"], B: T.Buffer[(4, 4), "int32"]):
    for io,ii in T.grid(4,4):
        T.assume(4*io+ii < 14 or A[io,ii] == T.undef())

    for i in T.serial(14):
        B[i // 4, i % 4] = 2 * A[i // 4, i % 4]

    # This will be removed later during lowering, and is used to
    # signify that the function is allowed to alter the return value
    # in these indices.
    for io,ii in T.grid(4,4):
        if 4*io + ii >= 14:
            B[io,ii] = T.undef()


# sched.sequential_buffer_access(B)
@T.prim_func
def func(A: T.Buffer[(4, 4), "int32"], B: T.Buffer[(4, 4), "int32"]):
    for io, ii in T.grid(4, 4):
        if 4 * io + ii < 14:
            B[io, ii] = 2 * A[io, ii]

    for io,ii in T.grid(4,4):
        if 4*io + ii >= 14:
            B[io,ii] = T.undef()


# sched.remove_branching_through_overcompute()
#
# If we were to replace the `if 4 * io + ii < 14` conditional with the
# following if/else block, the else block would only write to indices
# that are later overwritten by T.undef().
#
# if 0 <= 4 * io + ii < 14:
#     B[io, ii] = 2 * A[io, ii]
# else:
#     B[io, ii] = 2 * A[io, ii]
#
# Since a write is a no-op if it is overwritten by another write, we
# can remove the conditional.
@T.prim_func
def func(A: T.Buffer[(4, 4), "int32"], B: T.Buffer[(4, 4), "int32"]):
    for io,ii in T.grid(4,4):
        T.assume(4*io+ii < 14 or A[io,ii] == T.undef())

    for io, ii in T.grid(4, 4):
        B[io, ii] = 2 * A[io, ii]

    for io,ii in T.grid(4,4):
        if 4*io + ii >= 14:
            B[io,ii] = T.undef()

# The T.assume and stores to T.undef are removed later, when lowering
# the function.
#
# tir.transform.RemoveAssume
# tir.transform.RemoveStoreUndef
@T.prim_func
def func(A: T.Buffer[(4, 4), "int32"], B: T.Buffer[(4, 4), "int32"]):
    for io, ii in T.grid(4, 4):
        B[io, ii] = 2 * A[io, ii]
```

This uses the `T.undef()` placeholder to determine that the
overcompute can be performed without impacting any desired output,
followed by removing these placeholders.

### Multiple Buffer Semantics

If multiple transformations are applied to a single buffer, the
semantics of padding may differ.  The general rule is that the
semantics for each index follow whichever transformation introduced
them.



```python
# Initial function
@T.prim_func
def func(A: T.Buffer[(14, 60), "int32"], B: T.Buffer[(14,), "int32"]):
    for i in T.serial(14):
        B[i] = 0
        for j in T.serial(60):
            B[i] = B[i] + A[i, j]

# First transformation of A, reads on `io==3 and ii>=2` are valid, but return undefined value.
# sched.transform_layout(
#     A,
#     lambda i, j: [i // 4, i % 4, j],
#     pad_value=lambda io, ii, j: tir.undef(),
# )
@T.prim_func
def func(A: T.Buffer[(4, 4, 60), "int32"], B: T.Buffer[(14,), "int32"]):
    for io,ii,j in T.grid(4,4,60):
        T.assume(4*io + ii < 14 or A[io,ii,j]==T.undef())

    for i in T.serial(14):
        B[i] = 0
        for j in T.serial(60):
            B[i] = B[i] + A[i // 4, i % 4, j]


# Second transformation of A, reads on `jo==7 and ji>=4` are valid and return 0
# sched.transform(
#     A,
#     lambda io, ii, j: [io, ii, j // 8, j % 8],
#     pad_value=0,
# )
@T.prim_func
def func(A: T.Buffer[(4, 4, 8, 8), "int32"], B: T.Buffer[(14,), "int32"]):
    for io,ii,jo,ji in T.grid(4,4,8,8):
        T.assume(8*jo + ji < 60 or A[io,ii,jo,ji]==0)

    for io,ii,j in T.grid(4,4,60):
        T.assume(4*io + ii < 14 or A[io,ii,j//8,j%8]==T.undef())

    for i in T.serial(14):
        B[i] = 0
        for j in T.serial(60):
            B[i] = B[i] + A[i // 4, i % 4, j // 8, j % 8]


# B's layout padding may contain an arbitrary value
# sched.transform(
#     B,
#     lambda i: [i // 4, i % 4],
#     pad_value=lambda io, ii: tir.undef(),
# )
@T.prim_func
def func(A: T.Buffer[(4, 4, 8, 8), "int32"], B: T.Buffer[(4, 4), "int32"]):
    for io,ii,jo,ji in T.grid(4,4,8,8):
        T.assume(8*jo + ji < 60 or A[io,ii,jo,ji]==0)

    for io,ii,j in T.grid(4,4,60):
        T.assume(4*io + ii < 14 or A[io,ii,j//8,j%8]==T.undef())

    for i in T.serial(14):
        B[i // 4, i % 4] = 0
        for j in T.serial(60):
            B[i // 4, i % 4] = B[i // 4, i % 4] + A[i // 4, i % 4, j // 8, j % 8]

    for io,ii in T.grid(4,4):
        if 4*io + ii >= 14:
            B[io,ii] = T.undef()

# sched.sequential_buffer_access(A)
@T.prim_func
def func(A: T.Buffer[(4, 4, 8, 8), "int32"], B: T.Buffer[(4, 4), "int32"]):
    for io,ii,jo,ji in T.grid(4,4,8,8):
        T.assume(8*jo + ji < 60 or A[io,ii,jo,ji]==0)

    for io,ii,jo,ji in T.grid(4,4,8,8):
        if 8*jo + ji < 60:
            T.assume(4*io + ii < 14 or A[io,ii,jo,ji]==T.undef())

    for io, ii in T.grid(4, 4):
        if 4 * io + ii < 14:
            B[io, ii] = 0
            for jo, ji in T.grid(8, 8):
                if 8 * jo + ji < 60:
                    B[io, ii] = B[io, ii] + A[io, ii, jo, ji]

    for io,ii in T.grid(4,4):
        if 4*io + ii >= 14:
            B[io,ii] = T.undef()

# sched.remove_branching_through_overcompute()
@T.prim_func
def func(A: T.Buffer[(4, 4, 8, 8), "int32"], B: T.Buffer[(4, 4), "int32"]):
    for io,ii,jo,ji in T.grid(4,4,8,8):
        T.assume(8*jo + ji < 60 or A[io,ii,jo,ji]==0)

    for io,ii,jo,ji in T.grid(4,4,8,8):
        if 8*jo + ji < 60:
            T.assume(4*io + ii < 14 or A[io,ii,jo,ji]==T.undef())

    for io, ii in T.grid(4, 4):
        B[io, ii] = 0
        for jo, ji in T.grid(8, 8):
            B[io, ii] = B[io, ii] + A[io, ii, jo, ji]

    for io,ii in T.grid(4,4):
        if 4*io + ii >= 14:
            B[io,ii] = T.undef()

# tir.transform.RemoveCompileTimeAssumptions()
# tir.transform.RemoveStoreUndef()
@T.prim_func
def func(A: T.Buffer[(4, 4, 8, 8), "int32"], B: T.Buffer[(4, 4), "int32"]):
    for io, ii in T.grid(4, 4):
        B[io, ii] = 0
        for jo, ji in T.grid(8, 8):
            B[io, ii] = B[io, ii] + A[io, ii, jo, ji]

```

In this example, `A` has different pading values stored along the `i`
and `j` dimensions.  Because padding along `jo` and `ji` has zeros,
the `B[io,ii] = B[io,ii] + 0` can be reduced to a no-op.  Because
padding along `io` and `ii` is later overwritten by `T.undef()`,
the `B[io,ii] = 0` and `B[io,ii] = B[io,ii] + A[io,ii,jo,ji]` are
overwritten, and therefore can be inserted as a no-op.


## Points of Communication

* At memory planning, the executor must be able to query the buffer
  semantics supported by each implementation of an operator.  This may
  be possible in an automatic manner (e.g. pairing conditional
  statements with the constraint they are maintaining, and identifying
  simplifications if those conditionals are removed), or may require
  manual annotation.

* When scheduling, all operators that share a buffer must use the same
  layout transformation (or sequence of layout transformations), and
  must have the same buffer constraints applied.

* When splitting a single `PrimFunc` into multiple functions, such as
  hoisting a stage into an independent `PrimFunc`, buffer annotations
  should be used to expose non-local information that may be used for
  local simplification.  The choice of how much information to expose
  should be made when hoisting the stage, and must be provable using
  the hoisted stage.

  ```python
  # Initial function, with two stages contained in a single function.
  @T.prim_func
  def single_func_step0(A: T.Buffer[14, "int32"], C: T.Buffer[1, "int32"]):
      B = T.alloc_buffer([4, 4], "int32")
      with T.block("transform"):
          for io, ii in T.grid(4, 4):
              if 4 * io + ii < 14:
                  B[io, ii] = A[4 * io + ii]

      with T.block("compute"):
          C[0] = 0
          for io, ii in T.grid(4, 4):
              if 4 * io + ii < 14:
                  C[0] = C[0] + B[io, ii]


  # This is a valid transformation, because it only impacts the values
  # in an internal cached buffer.
  @T.prim_func
  def single_func_step1(A: T.Buffer[14, "int32"], C: T.Buffer[1, "int32"]):
      with T.block("transform"):
          B = T.alloc_buffer([4, 4], "int32")
          for io, ii in T.grid(4, 4):
              if 4 * io + ii < 14:
                  B[io, ii] = A[4 * io + ii]
              else:
                  B[io, ii] = 0

      with T.block("compute"):
          C[0] = 0
          for io, ii in T.grid(4, 4):
              if 4 * io + ii < 14:
                  C[0] = C[0] + B[io, ii]

  # This is a valid transformation, removing the branch in the "compute"
  # block.  This is allowed, because the inserted statements `C[0] =
  # C[0] + B[3,2]` and `C[0] = C[0] + B[3,3]` can be proven to be no ops
  # by first inspecting "transform" and determining that `B[3,2] == 0`
  # and `B[3,3] == 0`.
  @T.prim_func
  def single_func_step2(A: T.Buffer[14, "int32"], C: T.Buffer[1, "int32"]):
      with T.block("transform"):
          B = T.alloc_buffer([4, 4], "int32")
          for io, ii in T.grid(4, 4):
              if 4 * io + ii < 14:
                  B[io, ii] = A[4 * io + ii]
              else:
                  B[io, ii] = 0

      with T.block("compute"):
          C[0] = 0
          for io, ii in T.grid(4, 4):
              C[0] = C[0] + B[io, ii]


  # Initial module, with two stages contained in two different functions.
  @ir_module
  class split_module:
      @T.prim_func
      def transform_A(A: T.Buffer[14, "int32"], B: T.Buffer[(4, 4), "int32"]):
          for io, ii in T.grid(4, 4):
              if 4 * io + ii < 14:
                  B[io, ii] = A[4 * io + ii]

      @T.prim_func
      def compute_C(B: T.Buffer[(4, 4), "int32"], C: T.Buffer[1, "int32"]):
          C[0] = 0
          for io, ii in T.grid(4, 4):
              if 4 * io + ii < 14:
                  C[0] = C[0] + B[io, ii]


  @ir_module
  class split_module:
      # This is NOT a valid transformation of transform_A, because it
      # changes the resulting value of an output buffer.
      @T.prim_func
      def transform_A(A: T.Buffer[14, "int32"], B: T.Buffer[(4, 4), "int32"]):
          for io, ii in T.grid(4, 4):
              if 4 * io + ii < 14:
                  B[io, ii] = A[4 * io + ii]
              else:
                  B[io, ii] = 0

      # This is NOT a valid transformation of compute_C, because the
      # inserted statements `C[0] = C[0] + B[3,2]` and `C[0] = C[0] +
      # B[3,3]` CANNOT be proven to be no-ops, nor can it even be
      # determined that `B[3,2]` and `B[3,3]` are safe to access as they
      # may contain uninitialized values.
      @T.prim_func
      def compute(B: T.Buffer[(4, 4), "int32"], C: T.Buffer[1, "int32"]):
          C[0] = 0
          for io, ii in T.grid(4, 4):
              C[0] = C[0] + B[io, ii]




  # Initial module, with two stages contained in two different
  # functions, with T.assume and T.undef statements.
  @ir_module
  class split_module:
      @T.prim_func
      def transform_A(A: T.Buffer[14, "int32"], B: T.Buffer[(4, 4), "int32"]):
          for io, ii in T.grid(4, 4):
              if 4 * io + ii < 14:
                  B[io, ii] = A[4 * io + ii]

          for io,ii in T.grid(4,4):
              if 4 * io + ii >= 14:
                  B[io,ii] = T.undef()

      @T.prim_func
      def compute_C(B: T.Buffer[(4, 4), "int32"], C: T.Buffer[1, "int32"]):
          for io,ii in T.grid(4,4):
              T.assume(4*io+ii < 14 or B[io,ii]==T.undef())

          C[0] = 0
          for io, ii in T.grid(4, 4):
              if 4 * io + ii < 14:
                  C[0] = C[0] + B[io, ii]


  @ir_module
  class split_module:
      # This is a valid transformation of transform_A.  The additional
      # statements `B[3,2] = 0` and `B[3,3] = 0` are no-ops, because
      # they are overwritten by `T.undef()`.
      #
      # Because stores of `T.undef()` are removed during lowering, a
      # write of `T.undef()` effectively acts as permission to change
      # these locations in the buffer.
      @T.prim_func
      def transform_A(A: T.Buffer[14, "int32"], B: T.Buffer[(4, 4), "int32"]):
          for io, ii in T.grid(4, 4):
              if 4 * io + ii < 14:
                  B[io, ii] = A[4 * io + ii]
              else:
                  B[io, ii] = 0

          for io,ii in T.grid(4,4):
              if 4 * io + ii >= 14:
                  B[io,ii] = T.undef()

      # This is NOT a valid transformation of compute_C, because the
      # inserted statements `C[0] = C[0] + B[3,2]` and `C[0] = C[0] +
      # B[3,3]` CANNOT be proven to be no-ops.  We can prove that it is
      # safe to access `B[3,2]` and `B[3,3]`, but that isn't enough to
      # prove that the inserted statements would be no-ops.
      #
      # Because `T.assume` calls are removed during lowering, a
      # `T.assume(buf[indices] == T.undef())` effectively acts as
      # permission to access the buffer at those indices.
      @T.prim_func
      def compute(B: T.Buffer[(4, 4), "int32"], C: T.Buffer[1, "int32"]):
          for io,ii in T.grid(4,4):
              T.assume(4*io+ii < 14 or B[io,ii]==T.undef())

          C[0] = 0
          for io, ii in T.grid(4, 4):
              C[0] = C[0] + B[io, ii]


  # Initial module, with two stages contained in two different
  # functions, with T.assume of a known value.
  @ir_module
  class split_module:
      @T.prim_func
      def transform_A(A: T.Buffer[14, "int32"], B: T.Buffer[(4, 4), "int32"]):
          for io, ii in T.grid(4, 4):
              if 4 * io + ii < 14:
                  B[io, ii] = A[4 * io + ii]

          for io,ii in T.grid(4,4):
              if 4 * io + ii >= 14:
                  B[io,ii] = 0

      @T.prim_func
      def compute_C(B: T.Buffer[(4, 4), "int32"], C: T.Buffer[1, "int32"]):
          for io,ii in T.grid(4,4):
              T.assume(4*io+ii < 14 or B[io,ii]==0)

          C[0] = 0
          for io, ii in T.grid(4, 4):
              if 4 * io + ii < 14:
                  C[0] = C[0] + B[io, ii]

  @ir_module
  class split_module:
      @T.prim_func
      def transform_A(A: T.Buffer[14, "int32"], B: T.Buffer[(4, 4), "int32"]):
          for io, ii in T.grid(4, 4):
              if 4 * io + ii < 14:
                  B[io, ii] = A[4 * io + ii]

          for io,ii in T.grid(4,4):
              if 4 * io + ii >= 14:
                  B[io,ii] = 0

      # This is a valid transformation of compute_C, because the
      # inserted statements `C[0] = C[0] + B[3,2]` and `C[0] = C[0] +
      # B[3,3]` be proven to be no-ops.  Where the single_func version
      # used earlier stages to provide the values of `B[3,2]` and
      # `B[3,3]`, here they are determined from the `T.assume()`
      # statement.
      #
      # Because `T.assume` calls are removed during lowering, a
      # `T.assume(buf[indices] == T.undef())` exposes non-local
      # constraints to the compute_C.
      @T.prim_func
      def compute_C(B: T.Buffer[(4, 4), "int32"], C: T.Buffer[1, "int32"]):
          for io,ii in T.grid(4,4):
              T.assume(4*io+ii < 14 or B[io,ii]==0)

          C[0] = 0
          for io, ii in T.grid(4, 4):
              C[0] = C[0] + B[io, ii]
  ```



# Drawbacks
[drawbacks]: #drawbacks

This heavily relies on the simplifier to identify alterations as being
no-ops, in order to prove that a branch can be removed.


# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

- Should we forbid padding in the layout transformation altogether?

  This is the behavior prior to this RFC, giving an error if
  transformation padding would be introduced.  For non-bijective
  transformations, the user would instead change the compute
  definition to introduce padding.

  This is unnecessarily restrictive, especially for predefined models
  where the shape is fixed, or automatically determined from previous
  operators.

- Should we pick a single set of buffer semantics, and apply it
  exclusively, rather than allowing options?

  If a single set of buffer semantics were selected, all efficient
  implementations should be expressible using those semantics.  From
  the examples of implementations discussed above, there is no one set
  of buffer semantics that would allow all of the examples.

- Should we add another option for buffer load semantics, in which a
  consumer is allowed to write to the transformation padding?

  This option could be used in cases where a producer has written a
  default value into the transformation padding, but the producer's
  default value is not the default value required by the consumer.

  Deciding against this option, as it could be instead expressed as a
  pre-processing step inserted as a separate operator in the Relay
  graph, and doesn't need to be expressed in interface between Relay
  and operators.  This constraints that would be introduced by this
  set of semantics are also not easily expressed in Relay, which
  assumes a single producer for each tensor.

- Should `transform_layout` specify the implementation style
  (e.g. "write padding after compute" or "predicated loads"), rather
  than introducing a separate `remove_branching_through_overcompute`?

  Deciding against this option, for two main reasons.  First, some
  implementation styles can only be used if constraints are met for
  both the input and output buffers (e.g. [propagating undef
  values from input padding to output padding](#apply-operator-
  element-wise-over-the-transformation-padding)), which makes it
  unclear which `layout_transform` should rewrite the access pattern.
  Second, enumerating these access patterns for all possible input and
  output buffers would be non-trivial, especially for operators with
  multiple input buffers, each of which may have a different layout.

- Should `Schedule.sequential_buffer_access` be an independent
  schedule primitive, rather than a wrapper around existing
  primitives?

  No.  The separate primitives are expected to be irreducible.

# Prior art
[prior-art]: #prior-art

- The `tir::builtin::assume` has the same semantics as [LLVM's
`__builtin_assume`
intrinsic](https://clang.llvm.org/docs/LanguageExtensions.html#builtin-assume).

- The `tir::builtin::undef` has similar semantics to [LLVM's
  `undef`](https://llvm.org/docs/LangRef.html#undefvalues).  The
  primary use by TVM is described "A store of an undefined value can
  be assumed to not have any effect; we can assume that the value is
  overwritten with bits that happen to match what was already there."

# Unresolved questions
[unresolved-questions]: #unresolved-questions

- Should the logical and transformed shape be queryable from a
  `runtime::Module`?

  This could be useful for a user to determine what transformed shape
  should be provided for a given logical shape.

- Should it be legal for transformation padding to be elided?

  For example, if the row-major traversal of elements is identical for
  both the logical and transformed layout, the transformed buffer
  could be expressed as an alias of a flattened buffer.  This would
  require that all accesses of the buffer use load/store semantics #1,
  such that access into transformation padding is invalid.

  ```python
  # Before transformation
  A = T.alloc_buffer(14)
  for i in range(14):
      A[i]

  # After transformation
  A_backing = T.alloc_buffer(14)
  A = T.decl_buffer([4,4], data=A_backing.data)
  for i in range(14):
      A[i//4, i%4]
  ```

# Future possibilities
[future-possibilities]: #future-possibilities

- Defining `BufferConstraint` based on compute definitions may be
  useful, and may allow for simplifications.  For example,
  `topi.nn.pad` could provide a constraint on the padding it applies
  around the perimeter.

- Potential to reduce number of computations by using
  `HoistExpression` to hoist portions of non-boolean expressions.

  ```python
  # Additive terms relying on i are re-computed for each loop over k.
  #
  # Total: 6*n_i*n_j*n_k multiplications, 4*n_i*n_j*n_k additions
  for i,j,k in T.grid(n_i, n_j, n_k):
      A[i*n_j*n_k + j*n_k + k] = B[i*n_j*n*k + j*n_k + k]

  # Using current tir.transform.CommonSubexprElimTIR
  #
  # Total: 3*n_i*n_j*n_k multiplications, 2*n_i*n_j*n_k additions
  for i,j,k in T.grid(n_i, n_j, n_k):
      index = i*n_j*n_k + j*n_k + k
      A[index] = B[index]

  # Additive terms are computed once for each i loop, then re-used for
  # each j/k loop.
  #
  # i_stride: 1 multiplication
  # i_term: n_i multiplication
  # ij_term: n_i*n_j multiplication, n_i*n_j addition
  # index: n_j*n_j*n_k addition
  # Total: n_i*n_j + n_i + 1 multiplications, n_j*n_j*n_k + n_i*n_j additions
  i_stride = n_j*n_k
  for i in T.serial(n_i):
      i_term = i*i_stride
      for j in T.serial(n_j):
          ij_term = i_term + j*n_k
          for j in T.serial(n_k):
              index = ij_term + k
              A[index] = B[index]
  ```

- Automatically determine options for `pad_value` that would provide
  useful simplifications for an individual operator.  This could
  define a search space for graph-level optimization, w

  Currently uncertain if this is possible, and how difficult it would
  be to implement.  It would likely involve identifying conditional
  statements that correspond to `BufferConstraint`, then identifying
  specific patterns within the body of the conditional (e.g. whether a
  buffer value of zero or one results in a no-op, whether the indices
  have similar expressions to later writes).
