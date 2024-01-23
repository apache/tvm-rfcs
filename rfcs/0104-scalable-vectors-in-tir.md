- Feature Name: Scalable vectors in TIR
- Start Date: 2023-08-24
- RFC PR: [apache/tvm-rfcs#104](https://github.com/apache/tvm-rfcs/pull/104)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary
[summary]: #summary

As part of the work to improve support for vector length agnostic (VLA) programming in TVM, we have been prototyping scalable vectors in TIR. The discussion related to codegen in this RFC would be specific to Scalable Vector Extension (SVE) in Arm(R) architecture, but the TIR enhancements will be architecture agnostic. To read more about SVE, see the [SVE overview](https://developer.arm.com/documentation/102476/0100/Overview?lang=en)

This RFC corresponds to the second part of [the meta-RFC](https://discuss.tvm.apache.org/t/meta-rfc-vector-length-agnostic-vla-vectorization/13596) (with some changes to the initial plan).


# Motivation
[motivation]: #motivation

Due to the significant improvements in LLVM over the past few years the full scale of changes as described in the [first RFC](https://github.com/apache/tvm-rfcs/pull/94) was no longer necessary. Since LLVM can lower to SVE from fixed bound loops, there wasn't a need to create explicit SVE loops at TVM's codegen level.

Expressing scalability at the TIR level would be valuable though as it gives the schedule author more control over using scalable vectors and would make it possible to explicitly use features that are present in SVE, such as gather load and scatter store operations. Moreover, it would open up the path to support [Scalable Matrix Extension (SME)](https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/scalable-matrix-extension-armv9-a-architecture) in TVM.

Finally, some of the proposed changes could also benefit vector length specific vectorization, especially changes around dealing with loops that are not entirely vectorizable (this is discussed in [Predication](#predication)).

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

In SVE the size of scalable vectors is expressed through compile time unknown value called `vscale` (the name has also been adopted by [LLVM](https://llvm.org/docs/LangRef.html#llvm-vscale-intrinsic)). It essentially represents how many chunks of 128 bits fits into a given SVE hardware implementation. So e.g. `vscale = 1` implies 128 bit vectors and `vscale = 2` implies 256 bit vectors. The same SVE code will run on both of these implementations. 

## Vscale in TIR

We can introduce a TIR intrinsic for `vscale` that we can use in `Ramp` and `Broadcast` nodes to represent the scalable vectors in TIR. In the backend codegen `tir.vscale` can be directly mapped to `llvm.vscale`.

### An example

Here's how we could create scalable TIR by using the scheduling primitives: 
```
@T.prim_func
def before_split(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
    for i in range(128):
        with T.block("B"):
            vi = T.axis.spatial(128, i)
            T.reads(A[vi])
            T.writes(B[vi])
            B[vi] = A[vi] * 2.0

sch = tir.Schedule(before_split)
i, = sch.get_loops(sch.get_block("B"))
vscale = tvm.tir.vscale()
sch.split(i, factors=[tvm.tir.ceildiv(128, 4 * vscale), 4 * vscale])
```
would result in 
```
@T.prim_func
def after_split(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
    for i0, i1 in T.grid(tvm.tir.ceildiv(128, 4 * vscale), 4 * vscale):
        with T.block("B"):
            vi = T.axis.spatial(128, i0 * 4 * vscale + i1)
            T.reads(A[vi])
            T.writes(B[vi])
            B[vi] = A[vi] * 2.0
```
The multiplier 4 in front of the `vscale` comes form `128 / size_in_bits(float32)`. It encodes the minimum vector length in the given architecture, which is known at a compile time. After vectorizing the inner loop and lowering the TIR, we get:
```
@T.prim_func
def lowered(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
    for i_0 in range(tvm.tir.ceildiv(128, 4 * vscale)):
        B_1 = T.Buffer((128,), data=B.data)
        A_1 = T.Buffer((128,), data=A.data)
        B_1[i_0 * 4 * vscale:i_0 * 4 * vscale + 4 * vscale] = A_1[i_0 * 4 * vscale:i_0 * 4 * vscale + 4 * vscale] * T.Broadcast(T.float32(2), 4 * vscale)
```

For simplicity, this example corresponds to a case where we know that the axis dimension is divisible by the real hardware vector length. In general, we can't make this assumption - see [Predication](#predication) for more discussion.

Currently the LLVM backend maps ramps and broadcasts to LLVM vectors. Since LLVM represents the scalable vectors the same way as fixed length vectors, we can create the scalable vector operations in LLVM with minimal changes to the codegen. Here's a (shortened) example of LLVM with scalable vectors:

```
entry:
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(4000) %agg.result, i8 0, i64 4000, i1 false)
  %0 = tail call i64 @llvm.vscale.i64()
  %.neg = mul nuw nsw i64 %0, 1016
  %n.vec = and i64 %.neg, 1000
  %1 = tail call i64 @llvm.vscale.i64()
  %2 = shl nuw nsw i64 %1, 2
  %3 = tail call i64 @llvm.vscale.i64()
  %4 = shl nuw nsw i64 %3, 2
  %5 = tail call i64 @llvm.vscale.i64()
  %6 = shl nuw nsw i64 %5, 2
  %7 = tail call i64 @llvm.vscale.i64()
  %8 = shl nuw nsw i64 %7, 3
  br label %vector.body

vector.body:
  %index = phi i64 [ 0, %entry ], [ %index.next, %vector.body ]
  %9 = getelementptr inbounds i32, ptr %arr0, i64 %index
  %wide.load = load <vscale x 4 x i32>, ptr %9, align 4
  %10 = getelementptr inbounds i32, ptr %9, i64 %2
  %wide.load9 = load <vscale x 4 x i32>, ptr %10, align 4
  %11 = getelementptr inbounds i32, ptr %arr1, i64 %index
  %wide.load10 = load <vscale x 4 x i32>, ptr %11, align 4
  %12 = getelementptr inbounds i32, ptr %11, i64 %4
  %wide.load11 = load <vscale x 4 x i32>, ptr %12, align 4
  %13 = add nsw <vscale x 4 x i32> %wide.load10, %wide.load
  %14 = add nsw <vscale x 4 x i32> %wide.load11, %wide.load9
  %15 = getelementptr inbounds [1000 x i32], ptr %agg.result, i64 0, i64 %index
  store <vscale x 4 x i32> %13, ptr %15, align 4
  %16 = getelementptr inbounds i32, ptr %15, i64 %6
  store <vscale x 4 x i32> %14, ptr %16, align 4
  %index.next = add nuw i64 %index, %8
  %17 = icmp eq i64 %index.next, %n.vec
  br i1 %17, label %middle.block, label %vector.body
```


# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

Making ramps and broadcasts scalable would mean changing the data types of `ramp->lanes` and `broadcast->lanes` from `int` to `PrimExpr` and correctly dealing with the accesses to the lanes throughout the codebase. E.g. for `Ramp` the current node definition
```
class RampNode : public PrimExprNode {
 public:
  /*! \brief The base value. */
  PrimExpr base;
  /*! \brief The stride of each step. */
  PrimExpr stride;
  /*! \brief Total number of lanes. */
  int lanes;
 ...
```
would change to
```
class RampNode : public PrimExprNode {
 public:
  /*! \brief The base value. */
  PrimExpr base;
  /*! \brief The stride of each step. */
  PrimExpr stride;
  /*! \brief Total number of lanes. */
  PrimExpr lanes;
...
```
The are some advantages of representing the scalable vectors through modified `Ramp` and `Broadcast` nodes:
* LLVM's `IRBuilder` has same APIs for scalable and fixed length vectors, so basic support would require minimal changes in the LLVM backend
* We could take advantage of all the current vector related infrastructure in TVM (e.g. the [simplification rules](https://github.com/apache/tvm/blob/main/src/arith/rewrite_simplify.cc))
* We could mix fixed length and scalable vectors in scheduling


The main areas that need changes are:

1. `VectorizeLoop` pass - We need to handle creating scalable vectors from loops and correctly vectorizing the associated broadcasts.

2. `tir.split` and `tir.tile` (and TE equivalents) - These primitives would be the main interface through which we can create scalable vectors and, therefore, they should support compile time unknown factors

3. `runtime::DataType` - We need to express scalable lanes in `runtime::DataType` in a way that doesn't break the DLPack standard ABI. It was [suggested by #tqchen](https://github.com/apache/tvm-rfcs/pull/18#issuecomment-930488098) to define
    ```
    int kScalableVectorLaneMark = -1
    ```
    to represent scalable lanes in `DataType`. The lanes of scalable vectors are usually expressed as a multiple of `vscale`. In order to make the lanes value (including the scalar multiplier) recoverable from `DataType`, we will use a corresponding negative value, e.g. for `T.ramp(0, 1, 4 * vscale())` would have its `runtime::DataType` lanes set to -4.

4. **Codegen** - Most of the codegen changes would entail extending the current vector support to scalable vectors. Besides handling the `tir.vscale -> llvm.vscale` conversion, an important enhancement would be support for strided ramps and broadcasts. Unlike Arm(R) Neon(TM) instruction set, SVE supports gather load and scatter store operations, which in TIR could be represented either by a `BufferLoad`/`Bufferstore` which is indexed by a `Ramp` with `stride != 1`, e.g.
    ```
    A[T.ramp(0, 2, 4 * tir.vscale())]
    ```
    or when the index is a non-linear expression involving `Ramp`, e.g.
    ```
    A[(10 * T.ramp(0, 1, 8 * tir.vscale())) % 4]
    ```
    The current CPU codegen supports strided ramps by scalarising, while with SVE these could be turned into `llvm.masked.gather` and `llvm.masked.scatter` intrinsics. 

## Predication
Predication is used in vectorization when the axis dimension is not divisible by the vector length.

Let's start with a simple loop and split and vectorize it:
```
@T.prim_func
def main(A: T.Buffer((60,), "float32"), B: T.Buffer((60,), "float32")):
    for i in range(60):
        with T.block("B"):
            vi = T.axis.spatial(60, i)
            T.reads(A[vi])
            T.writes(B[vi])
            B[vi] = A[vi]
    
sch = tvm.tir.Schedule(main)

l, = sch.get_loops(sch.get_block("B"))
vscale = tvm.tir.vscale()
sch.split(l, factors=[tvm.tir.ceildiv(60, 4 * vscale), 4 * vscale])

l0, l1 = sch.get_loops(sch.get_block("B"))
sch.vectorize(l1)
```
This will give us
```
@T.prim_func
def main(A: T.Buffer((60,), "float32"), B: T.Buffer((60,), "float32")):
    for i_0 in range(tvm.tir.ceildiv(60, 4 * vscale)):
        for i_1 in T.vectorized(4 * vscale):
            with T.block("B"):
                vi = T.axis.spatial(60, i_0 * 4 * vscale + i_1)
                T.where(i_0 * 4 * vscale + i_1 < 60)
                T.reads(A[vi])
                T.writes(B[vi])
                B[vi] = A[vi]
```
Actually vectorizing the TIR (i.e. creating the `Ramp` nodes for the buffer operation indices) will be left for `VectorizeLoop` pass in the TIR lowering pipeline. Currently this pass will keep the loops as scalar loops if the loop can't be exactly vectorized. In case of SVE we can use the predication instead. Since we don't know the vector length at the compile time, `VectorizeLoop` will always emit predicated loads and stores for SVE.

This would entail introducing another TIR intrinsic, `tir.get_active_lane_mask()` which is analogous to [`llvm.get.active.lane.mask.*`](https://llvm.org/docs/LangRef.html#llvm-get-active-lane-mask-intrinsics), i.e. it would take a variable (loop induction var) and a bound and produce a bit mask corresponding to the active lanes. We'd also need to implement support for predication in `BufferLoad` and `BufferStore` operations. In TVMScript we can support predicates by adding `Buffer::store` and `Buffer::load` methods which can accept predicate and create `BufferStore` and `BufferLoad` objects. Expressing predicated operations in TVMScript would then look like:
```
@T.prim_func
def main(A: T.Buffer((60,), "float32"), B: T.Buffer((60,), "float32")):
    for i0 in range(ceildiv(60, 4 * vscale)):
        let pred = T.get_active_lane_mask(i0 * 4 * vscale, 60)
        B.store(
            A.load([T.ramp(i0 * 4 * vscale, 1, 4 * vscale)], predicate=pred),
            [T.ramp(i0 * 4 * vscale, 1, 4 * vscale)], predicate=pred)
```
These loads and stores will then be lowered to [`llvm.masked.*`](https://llvm.org/docs/LangRef.html#masked-vector-load-and-store-intrinsics) intrinsics.

Predication is not exclusive to VLA programming so the implementation could also benefit fixed length vectors.



# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives
## Alternative to predication - cleanup loop
When using the cleanup loop, the TIR example above would instead lower to:
```
@T.prim_func
def main(A: T.Buffer((60,), "float32"), B: T.Buffer((60,), "float32")):
    for i0 in range(floordiv(60, 4 * vscale)):
        B[T.ramp(i0 * 4 * vscale, 1, 4 * vscale)] = A[T.ramp(i0 * 4 * vscale, 1, 4 * vscale)]

    for j0 in range(60 % 4 * vscale):
        B[4 * vscale * floordiv(60, 4 * vscale) + j0] = A[4 * vscale * floordiv(60, 4 * vscale) + j0]
```
These loads and stores can be lowered to "regular" LLVM vectorized load and store instructions with [scalable vector type](https://llvm.org/docs/LangRef.html#vector-type).


# Prior art
[prior-art]: #prior-art

## MLIR support of SVE

There is ongoing work to improve the VLA support in MLIR stack, see [this RFC](https://discourse.llvm.org/t/rfc-scalable-vectorisation-in-linalg/70419) for a nice summary about where it is at and what's the plan for the future. The changes are conceptually similar to the ideas outlined in this PR, making changes to the `linalg` and the `transform` dialects. 

## Halide support of SVE

Halide supports 128 bit SVE vectors - see the [PR](https://github.com/halide/Halide/pull/6781).

# Future possibilities
[future-possibilities]: #future-possibilities

## Scalable Matrix Extension (SME)

SME is an architecture extension that is targeting matrix multiply style operations by computing outer product between two scalable vectors. See the [introduction to SME](https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/scalable-matrix-extension-armv9-a-architecture) for more information.

Unlike for SVE, the LLVM will not support generating SME code from fixed bound loops, so the plan is to target the SME LLVM intrinsics from `tensorize` and define the corresponding loop nests through scalable loop bounds. 

## Exposing scalable vectors to tuning

In the future, we could include scalable vector support into meta schedule, making it possible for the tuner to choose between scalable and fixed length vectors or mix them in a same program.

### AutoTVM style manual design space description

New APIs similar to `sample_perfect_tile` could be added that would sample `vscale` dependent factors, e.g.
```
# sampling scalable tiles only
sch.sample_scalable_tile(l0, n=3, max_innermost_vscale_multiplier=2)

# sampling scalable and fixed length tiles
sch.sample_mixed_tile(l0, n=6, max_innermost_fixed_size_factor=8, max_innermost_vscale_multiplier=2)
```

### Autoscheduler style rule generation

Splitting a loop and vectorizing the innermost dimension is a very common pattern in scheduling for scalable vectors. In the presence of sampling primitives that can support `vscale` dependent splitting, we can create additional builtin composite schedule rules similar to the `ScheduleRuleMultiLevelTiling` family. This could create a wide variety of programs with scalable vectors for the tuner to trial.
