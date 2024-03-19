- Feature Name: Scalable Matrix Extension enablement
- Start Date: 2024-01-31
- RFC PR: [apache/tvm-rfcs#107](https://github.com/apache/tvm-rfcs/pull/107)
- GitHub Issue: [apache/tvm#16734](https://github.com/apache/tvm/issues/16734)

# Summary
[summary]: #summary

The Scalable Matrix Extension (SME) in the Arm® architecture builds on the Scalable Vector Extensions (SVE and SVE2), adding new capabilities for efficiently processing matrix operations ([read more](https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/scalable-matrix-extension-armv9-a-architecture)). This RFC explores how TVM can be utilized to generate code for the SME ISA to achieve improved inference performance on supported Arm®-based hardware implementing the SME extension.

# Motivation
[motivation]: #motivation

The vast majority of ML models are bound by the performance of matrix multiply style operations, which the SME ISA aims to accelerate. Unlike SVE, we cannot rely on the automatic vectorization functionality provided by LLVM to generate SME code. Code generation for SME must, therefore, be introduced at a higher level in the stack - in this case, TVM.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation
## User overview
[user-overview]: #user-overview
Similar to other architecture extensions for CPU devices, SME support will be specified through TVM's `Target` object. For example:
```python
tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mattr=+sme")
```
Notes:
 - `+sme` is used to express the capability of the target hardware and does not necessarily mean any SME code will be generated.
 - TVM will be required to be built with LLVM version 16 or above.

## SME concepts
[sme-concepts]: #sme-concepts
This section provides a brief introduction of key SME concepts. For more information please see this [blog post](https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/scalable-matrix-extension-armv9-a-architecture) or the [Architecture Reference Manual](https://developer.arm.com/documentation/ddi0616/latest/).
### Vector length agnostic programming
[vector-length-agnostic-programming]: #vector-length-agnostic-programming
The concept of vector-length agnostic programming was already introduced as part of RFC [#104](https://github.com/apache/tvm-rfcs/pull/104). It introduces a compile-time unknown constant "Vector Length" (VL) which denotes the vector length of the hardware in bits.

### Streaming mode
[streaming-mode]: #streaming-mode
A new mode of processor execution called "Streaming Mode" that focuses on throughput-oriented applications has been introduced. It must be enabled for the execution of SME instructions.

By introducing a separate mode for SME instructions, a hardware implementation can choose to use a different vector length for streaming and non-streaming processing. To differentiate between these lengths we call the streaming-mode vector length, "Streaming Vector Length" (SVL).

Some instructions become _illegal_ when entering streaming-mode and an attempt to execute such instruction will result in a run-time exception.

### Matrix tile storage
[matrix-tile-storage]: #matrix-tile-storage
SME instructions operate on an architectural register state called "ZA storage". It can be viewed as a two-dimensional tile, of size SVL * SVL, which is capable of accumulating the results of several SME operations. Vectors of length SVL can be loaded/stored from/to the ZA storage tile using instructions such as `ld1w.horiz` or `st1w.vert`, where `horiz` and `vert` indicate the orientation of the operation on the tile. An additional `slice_index` parameter is used to indicate the offset into the tile.

Note that ZA storage is divided into several sub-tiles depending on the type of data being stored. This concept has been omitted from the RFC to reduce complexity. Examples in this RFC will consider using only a single sub-tile.

## Developer overview
[developer-overview]: #developer-overview
Dissimilar to [SVE](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0094-aarch64-backend-with-sve.md), we will not be able to rely on the auto-vectorization functionality provided by LLVM to generate SME instructions from fixed-bound loops, therefore they must be introduced from TVM.

A series of new "SME" schedules, for each operator of interest, will be introduced into the TVM Operator Inventory (TOPI). They will leverage the `tensorize` scheduling primitive to call tensor intrinsics that directly call either LLVM intrinsics or hand-crafted assembly routines.

The following is an example showing how a simple outer product operation (<img src="https://latex.codecogs.com/svg.image?&space;z=x\otimes&space;y" >) can be scheduled:
```python
@T.prim_func
def before(x: T.handle, y: T.handle, z: T.handle):
    X = T.match_buffer(x, (16,), "float32")
    Y = T.match_buffer(y, (16,), "float32")
    Z = T.match_buffer(z, (16, 16), "float32")

    with T.block("root"):
        for a, b in T.grid(16, 16):
            v_a, v_b = T.axis.remap("SS", [a, b])
            Z[v_a, v_b] = X[v_a] * Y[v_b]

sch = tvm.tir.Schedule(before)
a, b = sch.get_loops(sch.get_block("root"))
outer_a, inner_a = sch.split(a, factors=(16 / (4 * T.vscale()), 4 * T.vscale()))
outer_b, inner_b = sch.split(b, factors=(16 / (4 * T.vscale()), 4 * T.vscale()))
sch.reorder(outer_a, outer_b, inner_a, inner_b)

sch.mod.show()
@T.prim_func
def after_reordering_and_splitting(x: T.handle, y: T.handle, z: T.handle):
    X = T.match_buffer(x, (16,), "float32")
    Y = T.match_buffer(y, (16,), "float32")
    Z = T.match_buffer(z, (16, 16), "float32")

    with T.block("root"):
        for a_outer, b_outer, a_inner, b_inner in T.grid(16 / (4 * T.vscale()), 16 / (4 * T.vscale()), 4 * T.vscale(), 4 * T.vscale()):
            a = a_outer * (4 * T.vscale()) + a_inner
            b = b_outer * (4 * T.vscale()) + b_inner
            v_a, v_b = T.axis.remap("SS", [a, b])
            Z[v_a, v_b] = X[v_a] * Y[v_b]

sch.tensorize(inner_a, ARM_SME_OUTER_PRODUCT_TENSOR_INTRIN)

after_tensorize_and_annotation = sch.mod["main"]
after_tensorize_and_annotation.with_attr({
    "aarch64_pstate_sm": "enabled",
    "aarch64_pstate_za": "new",
})

after_tensorize_and_annotation.show()
@T.prim_func
def after_tensorize_and_annotation(x: T.handle, y: T.handle, z: T.handle):
    T.func_attr({"aarch64_pstate_sm": "enabled", "aarch64_pstate_za": "new"})
    X = T.match_buffer(x, (16,), "float32")
    Y = T.match_buffer(y, (16,), "float32")
    Z = T.match_buffer(z, (16, 16), "float32")

    with T.block("root"):
        for a_outer, b_inner in T.grid(16 / (4 * T.vscale()), 16 / (4 * T.vscale())):
            a = a_outer * (4 * T.vscale())
            b = b_outer * (4 * T.vscale())

            # Load an SVL chunk of "X" and "Y" input tensors
            a_svl = X[T.ramp(a, 1, 4 * T.vscale())]
            b_svl = Y[T.ramp(b, 1, 4 * T.vscale())]

            # Calculate outer product on loaded SVL vectors
            T.evaluate(T.call_llvm_intrin("llvm.aarch64.sme.mopa", a_svl, b_svl))

            # Store the (partial) result of size SVLxSVL to the output
            # tensor "Z" using multiple SVL length stores
            for i in T.serial(4 * T.vscale()):
                T.evaluate(T.call_llvm_intrin(
                    "llvm.aarch64.sme.st1w.horiz",
                    Z.access_ptr("w", offset=a * i * Z.stride + b),
                    slice_index=i, ...
                ))
```
First, the loops `a` and `b` are restructured for tensorization. They are split by a compile-time unknown constant, `4 * tir.vscale()` (the multiplier of 4 is calculated by: minimum SVL size in bits / datatype bits, where the minimum SVL size is 128-bits), then reordered to form an inner loop nest of size SVLxSVL. This is made possible by the work in RFC [#104](https://github.com/apache/tvm-rfcs/pull/104) and the result is shown in `after_reordering_and_splitting`.

The inner loop nest is then replaced with a tensor intrinsic, `ARM_SME_OUTER_PRODUCT_TENSOR_INTRIN`, which introduces the LLVM SME intrinsics as can be seen in the `after` PrimFunc. To simplify the example, some arguments for these intrinsic calls have been omitted. `llvm.aarch64.sme.ld1w` loads an SVL-sized vector to an SVE "Z" register and `llvm.aarch64.sme.st1w` stores an SVL-sized vector from the ZA storage tile. `llvm.aarch64.sme.mopa` performs the outer product.

Finally, we annotate the scheduled TIR PrimFunc with two function attributes: `aarch64_pstate_sm` and `aarch64_pstate_za`. Both attributes are used to denote a change in the processor state. "sm" refers to the aforementioned [streaming-mode](#streaming-mode) while "za" refers to the [ZA storage](#matrix-tile-storage). The values these attributes can take and how they are consumed will be covered in more detail [below](#modifying-processor-state).

Initially, the schedule is intended to be registered in the TVM Operator Inventory (TOPI) where it will be specified as the preferred schedule when the target device supports SME.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

## Modifying processor state
[modifying-processor-state]: #modifying-processor-state
To successfully execute SME operations, the processor state needs to be altered from the regular mode of execution; both the [streaming-mode](#streaming-mode) and [ZA storage](#matrix-tile-storage) must be enabled. In assembly, this looks like:
```
smstart sm  // Enable streaming-mode
smstart za  // Enable ZA storage

...         // Execute some SME operations

smstop za   // Disable ZA storage
smstop sm   // Disable streaming-mode
```

To abstract this interface, LLVM provides several [function annotations](https://llvm.org/docs/AArch64SME.html#introduction) that ensure the correct processor state is used when a function is called. For code generation in TVM, the following function attributes will be exposed (other attributes may be exposed in the future if/when they are needed):

Streaming mode:
- aarch64_pstate_sm_enabled - Enter streaming-mode before executing the marked function.
- aarch64_pstate_sm_compatible - The marked function may run in either streaming or non-streaming-mode.

ZA Storage:
- aarch64_pstate_za_new - A new ZA storage state is created from scratch and not shared with the function caller.

### A0: Function-level annotations
[function-level-annotations]: #function-level-annotations
Attributes are added to `func_attr` after an operation has been scheduled. Choosing to add function attributes at the function level in TIR seems appropriate as it maintains a 1:1 correspondence with the generated functions and function attributes in LLVM. Additionally, a finer granularity of control of the processor state is unlikely to be needed since SME schedules will be added per ML operator. The example below illustrates a common case when streaming-mode is enabled and ZA storage is used:
```python
@T.prim_func()
def my_prim_func():
    T.func_attr({"aarch64_pstate_sm": "enabled", "aarch64_pstate_za": "new"})
    T.evaluate(0)
```
The function attributes will be consumed during code generation using the AArch64-specific LLVM backend mentioned in RFC [#94](https://github.com/apache/tvm-rfcs/pull/94). Attributes must be placed on the `_compute_` function, otherwise, if they are placed on the containing function, the processor state will be reverted when calling the `_compute_` function. After code generation, the result is similar to the following LLVM:
 ```
 define dllexport i32 @default_function(...) #0 {
  ...
  tail call fastcc void @default_function_compute_(...)
  ...
}

define internal fastcc void @default_function_compute_(...) #2 {
  ...
}

attributes #0 = { "target-cpu"="generic" "target-features"="+sme" }
attributes #2 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: write) "aarch64_pstate_sm_enabled" "aarch64_pstate_za_new" "target-cpu"="generic" "target-features"="+sme" }
 ```

In the future, it may also be necessary to add the annotations to the containing function (`default_function`) for optimizations such as reducing the number of transitions to streaming-mode between function calls that use SME intrinsics, however, this needs more consideration. In this instance, the function annotations on the `_compute_` function will need to be dissimilar from the containing function to avoid nesting of the states and therefore unnecessary transitions between the states at the function call site.

### A1: Block-level annotations
[block-level-annotations]: #block-level-annotations
The current compilation path from Relay strategy to TE to TIR does not allow for function annotations to be inserted during scheduling and propagated through compilation to code generation. For example, [LowerSchedule](https://github.com/apache/tvm/blob/81fd9f3476ed64b4d23ab8f38286a9bd7accc947/src/driver/driver_api.cc#L379-L380) constructs an `IRModule` from a `te.Schedule` but does not allow user-defined function attributes to be attached to the contained `PrimFunc` during conversion.

Therefore, annotations can be inserted at the block level, for example:
```python
@T.prim_func()
def my_prim_func():
    T.block("root"):
        T.block_attr({"pragma_aarch64_pstate_sm": "enabled", "pragma_aarch64_pstate_za": "new"})
        T.evaluate(0)
```
Similar to A0, the attributes can be consumed using the AArch64-specific LLVM backend and they must be placed on the `_compute_` function.

### Comparison
[comparison]: #comparison
A1 has some drawbacks when compared to A0. Firstly, annotations must be prefixed with `pragma_` otherwise the attributes will be removed during compilation. Secondly, A1 no longer has a 1:1 correspondence with LLVM functions meaning some validation may be needed to check for misuse, for example, if there are multiple blocks within a function with processor state attributes, which attributes should the generated LLVM function use?

However, function attributes added during scheduling are removed when being lowered to TIR which is problematic for A0. A1 is simpler to incorporate into code generation - it simply requires overriding the `AttrStmtNode` visitor and adding the attributes to the current LLVM function in context (`function_`). Therefore, we currently take the A1 approach.

## SME tensor intrinsics
[sme-tensor-intrinsics]: #sme-tensor-intrinsics
Section ["Developer overview"](#Developer-overview) discusses the use of a tensor intrinsic to replace the naive compute definition. SME operations can be inserted using either LLVM intrinsics or hand-crafted assembly kernels from the [Arm Compute Library (ACL)](https://github.com/ARM-software/ComputeLibrary). Both of these methods have previously been leveraged in TVM for other schedules e.g. [LLVM intrinsic based](https://github.com/apache/tvm/commit/4c77f0fc24b3b8dc4cf840ebaed215b4da9732b9#diff-0dc88eafe8a7c25c723416b6ad7e6faa758c5a312ee003d3e5bb7d9d146b2327R70), [assembly based](https://github.com/apache/tvm/commit/958c27123a45a9629e57cee20dbca28263c836bd#diff-9c48f6ea05361142654c9dd7417b9060631caf58104000de773c9da1f67d07aeR270).

### B0: LLVM intrinsics
[llvm-intrinsics]: #llvm-intrinsics
LLVM exposes architecture-specific intrinsics that we can use to target SME operations such as [mopa](https://github.com/llvm/llvm-project/blob/main/llvm/test/CodeGen/AArch64/sme-intrinsics-mopa.ll) which computes the outer product of two input vectors. These can be called directly from within the tensor intrinsic, for example:
```python
T.call_llvm_intrin(
    "void",                                 # Return type of the intrinsic
    "llvm.aarch64.sme.mopa"                 # Intrinsic name
    T.uint32(5),                            # Number of arguments
    0,                                      # ZA storage sub-tile index
    T.Broadcast(T.int1(1), 4 * vscale),     # Predicate for input A
    T.Broadcast(T.int1(1), 4 * vscale),     # Predicate for input B
    A.access_ptr(...),                      # Input A base ptr
    B.access_ptr(...),                      # Input B base ptr
)
```

### B1: Assembly micro-kernels
[assembly-micro-kernels]: #assembly-micro-kernels
We may also incorporate hand-crafted ACL routines into the tensor intrinsics for more complex operations, for example:
```python
c_code = """
extern "C" void my_gemm_kernel(...) {
    __asm__ __volatile__(
        "smstart sm\n"
        "smstart za\n"
        "fmopa za0.s, p0/M, p1/M, z0.s, z1.s\n"
        ...
        "smstart za\n"
        "smstart sm\n"
    )
}
"""
ll = tvm.contrib.clang.create_llvm([c_code], cc="clang++")
T.block_attr({"pragma_import_llvm": ll})
T.call_extern("int32", "my_gemm_kernel", ...)
```

### ACLE intrinsics
[acle-intrinsics]: #acle-intrinsics
Defining the tensor intrinsics using higher-level SME [C Language Extensions (ACLE)](https://arm-software.github.io/acle/main/acle.html#scalable-matrix-extension-sme) intrinsics was considered, but it is not yet available in released versions of LLVM. Additionally, they tend to have a one-to-one correspondence with the LLVM intrinsics mentioned in B0 anyway.
### Comparison
[comparison]: #comparison
The generated TIR function for B1 is opaque meaning it will not be able to take advantage of optimizations provided by TVM. On the other hand, B0 exposes more information to the compiler making TIR optimizations more easily accessible in the future. B0 also allows LLVM's powerful optimizations to be used.

Similarly, B0 is more readable than B1, therefore making improvements and maintenance more accessible to a wider range of developers, rather than being required to read and interpret assembly.

B1, however, is likely quicker to incorporate into the codebase as the kernels are already written and it can require less knowledge of the underlying algorithm. This means users of TVM can take advantage of the benefits of SME more quickly, especially for operators with complex algorithms. It also means that the annotations described in ["Modifying processor state"](#modifying-processor-state) are not necessary as the processor state modification can occur within the kernel itself.

In early experiments, B0 performs comparably to B1 for a simple dense layer (matrix multiply). A combination of B0 and B1 will likely be necessary to achieve high performance for a variety of operations.

## Registering STIR SME schedules in Relay Strategy
[regiester-stir-schedule]: #register-stir-schedule
A series of STIR schedules will be created for each operator of interest. We wish to use these together with previous optimizations contributed via the more traditional TE/TOPI scheduling.

It is possible to leverage `ScheduleFnDatabase` to schedule a compute definition using STIR. If an STIR schedule is not defined for the compute definition, we can fall back to the TE/TOPI schedules. An example compile flow:
```python
# Compute definitions for both scheduling strategies are defined in
# Relay strategy as before.

def apply_stir_outer_product_sme_schedule(sch: tvm.tir.Schedule) -> None:
    sch.get_block("outer_product_sme")
    # Scheduling for SME intrinsics...
    sch.tensorize(...)

def arm_cpu_stir_strategy(sch: tir.Schedule) -> bool:
    # Detect which compute function to apply based on compute block name
    if sch.has_block("outer_product_sme"):
        apply_stir_outer_product_sme_schedule(sch)
        return True

    # Fallback to TE based scheduling in Relay strategy
    return False

with meta_schedule.database.ScheduleFnDatabase(arm_cpu_stir_strategy):
    tvm.relay.build(mod, target, ...)
```

## Testing
[testing]: #testing
Upstream CI does not have a device capable of testing SME-generated code for correctness. For this reason, we need to rely on a simulator that can emulate SME operations such as the [Fixed Virtual Platform (FVP)](https://developer.arm.com/Tools%20and%20Software/Fixed%20Virtual%20Platforms). A similar FVP is also currently in use for Arm® Cortex®-M/Arm® Ethos™-U correctness tests, allowing the same testing infrastructure to be utilized with little modification.

The tests will be compiled via AOT and run using the AOT testing infrastructure under `python/tvm/testing/aot.py`, similar to how it is utilized currently. AOT compilation was selected as it is capable of producing a self-contained application that can run bare-metal on the FVP, this improves the latency of the tests (since an OS is not required to initialize), thus, ensuring the time taken to run all the tests is minimalized.

To ensure SME operations are being generated by the schedules, tests to check SME assembly instructions after compilation will also be created.

# Drawbacks
[drawbacks]: #drawbacks
Adding new schedules that utilize SME creates additional complexity in the codebase and may create a maintenance burden in the future, however, this should be outweighed by the potential performance benefits.

Testing SME code generation using a simulator will add additional burden on CI, however, until devices that implement SME are available, this is the only option for checking functional correctness.

# Prior art
[prior-art]: #prior-art
There is initial work in MLIR to enable lowering from the `linalg` dialect to LLVM SME intrinsics. Detailed discussion can be found in the [RFC](https://discourse.llvm.org/t/rfc-creating-a-armsme-dialect/67208/1), although, it should be noted that the current approach differs from the original proposal. MLIR has additional stages of lowering that need to be stitched together to produce SME, while the proposal for TVM will add SME LLVM intrinsics directly within the schedule.

[Similar](https://github.com/apache/tvm/pull/13642) tensor intrinsics that use LLVM intrinsics have been contributed to TVM for the Advanced Matrix Extension (AMX).

# Unresolved questions
[unresolved-questions]: #unresolved-questions

The ["Reference-level explanation"](#reference-level-explanation) section exposes several design choices, it would be good to hear thoughts from the community.

# Future possibilities
[future-possibilities]: #future-possibilities

## Streaming-SVE support
SME also supports running many SVE instructions while in streaming-mode to benefit from an extended SVL vector length (depending on the hardware implementation). Schedules that utilize SVE instructions may be able to benefit from a compatible streaming-mode annotation, for example,

```python
T.func_attr({"aarch64_pstate_sm": "compatible"})
```

thus allowing the schedule to be compiled for either SVE or SSVE depending on hardware support with minimal alteration.

## Auto-tensorization
This RFC will implement several tensor intrinsics that utilize SME operations. These intrinsics could be exposed to the Meta Scheduler to allow for automatic tensorization of the input graph.

## SME2 support
This RFC currently only considers SME. SME2 is the next iteration of SME that enables a wider range of applications to leverage the computational efficiency of SME. One notable improvement is support for matrix-vector operations.

## Applying SME schedules to the Relax compilation flow
This RFC has currently considered using SME schedules in the Relay compilation flow. This flow consists of registering compute and schedule definitions in TOPI and applying them via the Relay strategy. This is considered in the ["Registering STIR SME schedules in Relay Strategy"](#register-stir-schedule) section above. As a result, the following components will be provided: a TE compute definition, a STIR schedule and a strategy that matches a compute definition to a schedule based on the annotated name of a block.

The Relax compilation flow takes a [different approach](https://discuss.tvm.apache.org/t/discuss-tvm-core-strategy-for-operator-scheduling-and-tuning/16352) whereby scheduling is completed as an IRModule -> IRModule transform pass. The pass should be able to detect a pattern of the compute definition and apply the relevant scheduling. In the simplest form, it's possible to produce a pass that identifies blocks with the same annotated name and apply the previously defined STIR schedule to each matched block at a time. This pass can be contributed and applied as part of the [dlight](https://github.com/apache/tvm-rfcs/pull/dlight) package.
