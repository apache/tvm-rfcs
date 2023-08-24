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

Finally, some of the proposed changes could also benefit vector length specific vectorization, especially changes around dealing with loops that are not entirely vectorizable (this is discussed in [Predication](#2-generate-a-cleanup-loop)).

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

In SVE the size of scalable vectors is expressed through compile time unknown value called `vscale` (the name has also been adopted by [LLVM](https://llvm.org/docs/LangRef.html#llvm-vscale-intrinsic)). It essentially represents how many chunks of 128 bits fits into a given SVE hardware implementation. So e.g. `vscale = 1` implies 128 bit vectors and `vscale = 2` implies 256 bit vectors. The same SVE code will run on both of these implementations. 

## New TIR primitive - `tir.vfactor`

To express scalability in TVM stack, we would introduce a new TIR primitive called `tir.vfactor` (name pending) which is defined as `vscale * 128 / size_of_dtype_in_bits`. It is proportional to `vscale` and essentially represents the number of elements of type `dtype` that would fit into a scalable vector. This would make representing scalable vectors as `Ramp` and `Broadcast` nodes more intuitive:
```
A[T.ramp(0, 1, tir.vfactor())] = T.broadcast(3, tir.vfactor())
```

There are some advantages to defining a new node with special meaning over using `tir.Any` or `tir.Var`:
1. Special rules for `vfactor` nodes - e.g.
    ```
    vf = tir.vfactor()
    vf2 = tir.vfactor()
    assert(vf == vf2)
    ```
2. When splitting a fixed bound loop by a scalable factor, the bound of the outer loop will depend on `vfactor`, meaning that it will be encountered outside of the `Ramp` and `Broadcast` nodes and codegen needs to be able to distinguish between regular var and a var that needs to turned into `llvm.vscale`.

The current idea is to use `vfactor` throughout the TVM lowering and only convert it to `llvm.vscale` during codegen but there is a case to be made for another primitive, `tir.vscale` (see [2. in unresolved questions](#2-should-we-associate-data-type-with-tirvfactor)). Since the main interface to scalable vectors would be through scheduling language, we have so far opted for `vfactor`, but whether it would be appropriate to implement both of them or prefer one over the other is up for discussion.

## An example

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
vf = tvm.tir.vfactor()
sch.split(i, factors=[tvm.tir.ceildiv(128, vf), vf])
```
would result in 
```
@T.prim_func
def after_split(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
    for i0, i1 in T.grid(tvm.tir.ceildiv(128, vf), vf):
        with T.block("B"):
            vi = T.axis.spatial(128, i0 * vf + i1)
            T.reads(A[vi])
            T.writes(B[vi])
            B[vi] = A[vi] * 2.0
```
After vectorizing the inner loop and lowering the TIR, we get:
```
@T.prim_func
def lowered(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
    for i_0 in range(tvm.tir.ceildiv(128, vf)):
        B_1 = T.Buffer((128,), data=B.data)
        A_1 = T.Buffer((128,), data=A.data)
        B_1[i_0 * vf:i_0 * vf + vf] = A_1[i_0 * vf:i_0 * vf + vf] * T.Broadcast(T.float32(2), vf)
```
In the codegen the `vfactor` node will be turned into a `llvm.vscale` with the appropriate scaling factor that will depend on the data type of the `Ramp`/`Broadcast` elements.

For simplicity, this example corresponds to a simple case where we know that the axis dimension is divisible by `vfactor`. In general, we can't make this assumption - see [Predication](#1-predication) for more discussion.

## Support in Tensor Exression

Corresponding functionality would be added to TE, i.e. we'd need to add `te.vfactor` to make scalable vectors accessible from TE scheduling.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

Making ramps and broadcasts scalable would mean changing the data types of `ramp->lanes` and `broadcast->lanes` from `int` to `PrimExpr` and correctly dealing with the accesses to the lanes throughout the codebase. The main areas that need changes are:

1. `VectorizeLoop` pass - We need to handle creating scalable vectors from loops and correctly vectorizing the associated broadcasts

2. `tir.split` and `tir.tile` (and TE equivalents) - These primitives would be the main interface through which we can create scalable vectors and, therefore, they should support compile time unknown factors

3. `runtime::DataType` - We need to express scalable lanes in `runtime::DataType` in a way that doesn't break the DLPack standard ABI. It was [suggested by #tqchen](https://github.com/apache/tvm-rfcs/pull/18#issuecomment-930488098) to define
    ```
    int kScalableVectorLaneMark = -1
    ```
    to represent scalable lanes in `DataType`. Somewhat awkwardly the `runtime::DataType` uses `int` for lanes, while DLDataType uses `uint16_t`. However, we only use the wrapper to access the lanes, so as long as we handle `uint16_t -> int` conversion in a careful manner, this approach would work.

4. **Codegen** - Most of the codegen changes can be contained in a specialised AArch64 codegen that subclasses `CodeGenCPU`. Besides handling the `tir.vfactor -> llvm.vscale` conversion, an important enhancement would be support for strided ramps and broadcasts. Unlike Arm(R) Neon(TM) instruction set, SVE supports gather load and scatter store operations, which in TIR could be represented either by a `BufferLoad`/`Bufferstore` which is indexed by a `Ramp` with `stride != 1`, e.g.
    ```
    A[T.ramp(0, 2, tir.vfactor())]
    ```
    or when the index is a non-linear expression involving `Ramp`, e.g.
    ```
    A[(10 * T.ramp(0, 1, tir.vfactor())) % 4]
    ```
    The current CPU codegen supports strided ramps by scalarising, while with SVE these could be turned into `llvm.masked.gather` and `llvm.masked.scatter` intrinsics. 

# Unresolved questions
[unresolved-questions]: #unresolved-questions

## 1. Predication
This is to deal with the cases where the axis dimension is not divisible by `vfactor`.

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
vf = tvm.tir.vfactor()
sch.split(l, factors=[tvm.tir.ceildiv(60, vf), vf])

l0, l1 = sch.get_loops(sch.get_block("B"))
sch.vectorize(l1)
```
This will give us
```
@T.prim_func
def main(A: T.Buffer((60,), "float32"), B: T.Buffer((60,), "float32")):
    for i_0 in range(tvm.tir.ceildiv(60, vf)):
        for i_1 in T.vectorized(vf):
            with T.block("B"):
                vi = T.axis.spatial(60, i_0 * vf + i_1)
                T.where(i_0 * vf + i_1 < 60)
                T.reads(A[vi])
                T.writes(B[vi])
                B[vi] = A[vi]
```
Actually vectorizing the TIR (i.e. creating the `Ramp` nodes for the buffer operation indices) will be left for `VectorizeLoop` pass in the TIR lowering pipeline. Currently this pass will keep the loops as scalar loops if the loop can't be exactly vectorized. We need a different strategy for scalable vectors.

There are largely two options:
1. Implement predication for `BufferStore` and `BufferLoad` nodes
2. Generate a cleanup loop to deal with the loop tail

LLVM handles both of these cases and ideally TVM should eventually as well. It is not yet decided though which option to prioritize. With the two strategies available we could eventually choose between them through a `PassConfig`/command line option or introduce a heuristics based model. 

Let's look at the options in more detail...

### 1. Implement predication for `BufferStore` and `BufferLoad` nodes

This would entail introducing another TIR primitive, `tir.get_active_lane_mask()` which is analogous to [`llvm.get.active.lane.mask.*`](https://llvm.org/docs/LangRef.html#llvm-get-active-lane-mask-intrinsics), i.e. it would take a variable (loop induction var) and a bound and produce a bit mask corresponding to the active lanes. We'd also need to implement support for predication in `BufferLoad` and `BufferStore` operations. In TIR it would look like 
```
@T.prim_func
def main(A: T.Buffer((60,), "float32"), B: T.Buffer((60,), "float32")):
    for i0 in range(ceildiv(60, vf)):
        B_1 = T.Buffer((60,), data=B.data)
        A_1 = T.Buffer((60,), data=A.data)
        let pred = tir.get_active_lane_mask((i0 * vf), 60)
        B[T.ramp(i0 * vf, 1, vf), predicate=pred] = A[T.ramp(i0 * vf, 1, vf), predicate=pred]
```
These loads and stores can then be lowered to [`llvm.masked.*`](https://llvm.org/docs/LangRef.html#masked-vector-load-and-store-intrinsics) intrinsics. 

### 2. Generate a cleanup loop
The other option is to create an additional scalar cleanup loop, e.g.
```
@T.prim_func
def main(A: T.Buffer((60,), "float32"), B: T.Buffer((60,), "float32")):
    for i0 in range(floordiv(60, vf)):
        B[T.ramp(i0 * vf, 1, vf)] = A[T.ramp(i0 * vf, 1, vf)]

    for j0 in range(60 % vf):
        B[vf * floordiv(60, vf) + j0] = A[vf * floordiv(60, vf) + j0]
```
These loads and stores can be lowered to "regular" LLVM vectorized load and store instructions with [scalable vector type](https://llvm.org/docs/LangRef.html#vector-type). 

Also note that this optimisation is not specific to scalable vectors and could also benefit fixed length vector implementations. 

## 2. Should we associate data type with `tir.vfactor`
When we use `tir.vfactor` as a loop bound, e.g.
```
@T.prim_func
def after_split(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128,))
    B = T.match_buffer(b, (128,))
    for i0 in T.grid(ceildiv(128, tir.vfactor)):
        B[T.ramp(i0 * vf, 1, vf)] = A[T.ramp(i0 * vf, 1, vf)]
```
we have to know during codegen to translate the `tir.vfactor` into `(16 / size_of(dtype)) * vscale`, so we will need to know about the data type of the tensor we initially did the splitting on. There are various options to deal with that:

1. Associate a datatype with `vfactor` node, i.e.
    ```
    ...
    for i0 in T.grid(128 / tir.vfactor("float32")):
        ...
    ```
    This is probably the least effort option, but with `vfactor` essentially representing the number of elements, associating a data type with it would not make much sense.

2. Define `tir.vscale` in addition to `tir.vfactor` - 
    Since the multiplier that we use to translate `vfactor` into `vscale` in the codegen is based on the size of the datatype, we could do that translation at the time of creating the `For` loops with scalable bounds and end up with
    ```
    ...
    for i0 in T.grid(128 / (4 * tir.vscale())): # 4 comes from float32
        ...
    ```
    `tir.vscale` can then be directly mapped to `llvm.vscale` in the codegen.
3. Annotate the loops with the data type, i.e. 
    ```
    ...
    for i0 in T.grid(128 / tir.vfactor(), annotations={"vf_dtype": "float32"}):
        ...
    ```
    The annotation can then be used in the codegen to correctly convert `tir.vfactor` into `llvm.vscale`.

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

We could use `vfactor` with TVM's tuners to help us choose between fixed length and scalable vectors, e.g. we could include both into AutoTVM's tuning space:
```
cfg.define_knob("xi", [te.vscale_factor(), 2 * te.vscale_factor(), 4 * te.vscale_factor(), 16, 32, 64])
```
