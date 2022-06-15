- Feature Name: [RFC] RISC-V CSI-NN2 Compute Library integration
- Start Date: 2022-5-19
- RFC PR: https://github.com/apache/tvm-rfcs/pull/75
- GitHub Issue: [https://github.com/apache/tvm/issues/11506](https://github.com/apache/tvm/issues/11506)

# Summary

Introduce CSI-NN2 Compute Library into TVM to accelerate the inference performance of RISC-V CPU with Vector Extension.

# Motivation

Recently, in the latest Tiny v0.7 list released by AI benchmark MLPerf. Alibaba’s T-Head XuanTie RISC-V C906 processor has achieved first place in all 4 indicators. So, it’s a good time to support RISC-V CPUs with vector extension in TVM.

[CSI-NN2 Compute Library](https://github.com/T-head-Semi/csi-nn2)(CSINN2) is an open-source project that provides hand-crafted assembler routines for RISC-V CPUs with vector extension. It is compatible with RISC-V v0.7.1 and v1.0 vector extension instruction standards. This integration will look at how we can accelerate CPU performance for RISC-V devices like XuanTie C906 in TVM using CSINN2. The idea is that by converting operators from a relay graph to CSINN2 we can achieve faster inference times due to these routines. The initial intention is that this will improve performance for FP32 models. Although, with further improvements to the integration this will extend to quantized models and support for a wider range of operators.

PS: If you are interested in XuanTie C906 processor, [the D1 development board](https://d1.docs.aw-ol.com/en/) is a good choice.

# Guide-level explanation

## Build

- Build with CSI-NN2 support in `build`
  
  - Set in your config.cmake file
    
    ```cmake
    set(USE_OPENMP gnu)
    set(USE_CSINN /path/to/csi-nn2/install)
    set(USE_CSINN_DEVICE_RUNTIME X86)
    ```
  
  - Execute on the command lin
    
    ```shell
    cmake ..;make -j4
    ```

- Cross-compile CSI-NN2 support in `build-rv`
  
  - Set in your config.cmake file
    
    ```cmake
    set(USE_CPP_RPC ON)
    set(USE_LIBBACKTRACE OFF)
    set(USE_CSINN /path/to/csi-nn2)
    set(USE_CSINN_DEVICE_RUNTIME C906)
    ```
  
  - Execute on the command lin
    
    ```shell
    cmake ..;make -j4 runtime tvm_rpc
    ```
  
  After building successfully, we need to copy tvm_rpc and libs which used to device.

## Run

- Export binary library
  
  For a relay graph, following python APIs can be used to generate the binary library.
  
  ```python
  from tvm.relay.op.contrib import csinn
  
  # API to call CSINN2 partitioning
  # Here, module is the relay module
  csinn_module = csinn.partition_for_csinn(module)
  
  # Build the Relay graph.
  with tvm.target.Target("llvm -mtriple=riscv64-unknown-linux-gnu -mcpu=generic-rv64 -mabi=lp64d -mattr=+64bit,+m,+a,+f,+d,+c"):
      factory = tvm.relay.build(csinn_module)
  
  # Export the module
  lib_path = "lib_csinn2.so"
  cross_compile = 'riscv64-unknown-linux-gnu-g++'
  lib.export_library(lib_path, cc=cross_compile)
  ```

- Running RPC service on device.
  
  ```bash
  # on your device
  cd build-rv
  ./tvm_rpc server --host=172.16.202.11(your device ip) --port=9090
  # or using QEMU
  qemu-riscv64 -cpu c906fdv -L /path/to/csi-nn2/tools/gcc-toolchain/sysroot/ ./tvm_rpc server --host=127.0.0.1 --port=9090
  ```

# Reference-level explanation

The Relay graph as lowered from the TVM's frontend will be partitioned into subgraphs via running `AnnotateTarget`, `MergeCompilerRegions` and `PartitionGraph` Relay passes. Our current implementation uses JSON as a level of abstraction between relay operators and CSINN2 functions (or layers). Here is an overview of the flow from compilation to runtime:

- Front-end graph (Currently only NCHW is supported).
- Lower to relay graph.
- Run MergeComposite to create a mapping of relay operators to CSINN2 functions.
- `AnnotateTarget`, `MergeCompilerRegions` and `PartitionGraph`.
- Use the codegen stage to convert Relay operators annotated for CSINN2 to JSON.
- Use CSINNJSONSerializer serialize JSON and constant tensors into `mod.so` .

*CSINN runtime module context*

- Load `mod.so` and deserialize JSON and constant tensors.
- Create CSINN2 functions from JSON representation and cache.
- The cached functions are exposed to the graph runtime as packed functions.

Following code block shows the resultant IRModule post partitioning.

```shell
def @main(%data: Tensor[(1, 3, 24, 24), float32]) -> Tensor[(1, 10, 12, 12), float32] {
  @tvmgen_default_csinn_main_0(%data) /* ty=Tensor[(1, 10, 12, 12), float32] */
}

def @tvmgen_default_csinn_main_0(%csinn_0_i0: Tensor[(1, 3, 24, 24), float32], Inline=1, Compiler="csinn", global_symbol="tvmgen_default_csinn_main_0", Primitive=1) -> Tensor[(1, 10, 12, 12), float32] {
  %1 = fn (%FunctionVar_0_0: Tensor[(1, 3, 24, 24), float32], PartitionedFromPattern="nn.conv2d_nn.bias_add_", Composite="csinn.conv2d") -> Tensor[(1, 10, 12, 12), float32] {
    %0 = nn.conv2d(%FunctionVar_0_0, meta[relay.Constant][0] /* ty=Tensor[(10, 3, 3, 3), float32] */, strides=[2, 2], padding=[1, 1, 1, 1]) /* ty=Tensor[(1, 10, 12, 12), float32] */;
    nn.bias_add(%0, meta[relay.Constant][1] /* ty=Tensor[(10), float32] */) /* ty=Tensor[(1, 10, 12, 12), float32] */
  };
  %1(%csinn_0_i0) /* ty=Tensor[(1, 10, 12, 12), float32] */
}
```

## Build system

The current implementation has two separate build options in CMake. The reason for this split is because the optimized code for RISC-V cannot be used on an x86 machine.  We can set the flag to decide to generate code running on X86 or RISC-V.

```cmake
* USE_CSINN=OFF/ON/path-to-CSINN2
   * OFF - disable CSINN2 support. (default)
   * ON - add support for compiling CSINN2 codegen.
   * path-to-CSINN2 - use a specific version of the CSI-NN2 compute library.
* USE_CSINN_DEVICE_RUNTIME=OFF/X86/C906
   * OFF - disable CSINN2 runtime support. (default)
   * X86 - compiling CSINN2 runtime for x86 device.
   * C906 - cross-compiling CSINN2 runtime for C906 device.
```

# Testing

Firstly, we will be providing unit tests for the components described above.

Secondly, we are planning to use QEMU in the CI to be able to simulate the result running on C906.

Unit tests will be added alongside operator support. Once operator support matures, we will add network tests.

A unit tests will be of two kinds.

- Match operator patterns used by the graph partitioner.
  - It will be done for each operator and for a combination of operators both.
- Correctness of the CSI-NN2 operators against the native TVM output.
  - Actual output can be generated using QEMU.

## Code location

- The definition of relay operators
  
  `python/tvm/relay/op/contrib/csinn.py`

- C++ sources for implementation of passes and the generation of JSON
  
  `src/relay/backend/contrib/csinn/codegen.cc`

- Runtime file
  
  `src/runtime/contrib/csinn/csinn_json_runtime.cc`

- The test directory
  
  `tests/python/contrib/test_csinn`

# Drawbacks

CSI-NN2 provide hand coded operator. Therefore, code generation skips the auto tuning capabilities of TVM. In future, we wish to make use of full power of TVM's auto scheduling.

# Future possibilities

The current implementation uses the JSON runtime to run programs on the device. In the future, we will also generate C source code.
