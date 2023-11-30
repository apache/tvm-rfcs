- Feature Name: sycl_backend
- Start Date: 2023-09-15
- RFC PR: [apache/tvm-rfcs#0000](https://github.com/apache/tvm-rfcs/pull/0000)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary

[summary]: #summary

Add a new backend language——SYCL, enhancing TVM's compatibility and portability across different types of accelerators.

# Motivation

[motivation]: #motivation

What is SYCL?

SYCL is a cross-platform programming language, targeting heterogeneous computing architecture with a host connected to various heterogeneous accelerators. In implementation, SYCL is a high-level abstraction layer that wraps low-level APIs such as OpenCL, CUDA, Level0, HIP, XRT, Vulkan, etc. Compared to the cross-platform OpenCL, SYCL provides a higher-level programming model based on modern C++ and broader device support.

SYCL emerged in 2015 as a high-level abstraction layer for OpenCL. After the SYCL 2020 specification, OpenCL is no longer the only low-level backend for SYCL. Although it has appeared for a short time, SYCL has always received attention from the industry. SYCL is a standard that has some different implementations, such as Intel® oneAPI DPC++, ComputeCpp, HipSYCL, NeoSYCL, and triSYCL.

Due to the excellent expression ability of TVM TensorIR, it is possible to build a SYCL backend around TensorIR. Based on this background, we propose this RFC to add the SYCL backend, enhancing the compatibility and portability of TVM across different types of accelerators.

# Guide-level explanation

**How to use?**

Similar to other backends such as cuda, specify `target='sycl'` in the corresponding TVM API.

```python
tgt = tvm.target.Target(target='sycl') # Target
……
lib = tvm.build(mod, target='sycl') # Runtime module build
……
dev = tvm.device('sycl', 0) # Device that support sycl
inp = tvm.nd.array(data, device=dev) # Model input
```

The following sample code shows that computation with CUDA and SYCL backends respectively, and compare whether the results of the two backends are consistent.

```python
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import numpy as np

dtype = "float32"

# define computation by tvm script
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (8,), dtype=dtype)
        B = T.match_buffer(b, (8,), dtype=dtype)
        for i in range(8):
            with T.block("B"):
                vi = T.axis.spatial(8, i)
                B[vi] = A[vi] + 1.0
# thread binding
sch = tvm.tir.Schedule(MyModule)
block_b = sch.get_block("B")
(i,) = sch.get_loops(block_b)
i_0, i_1 = sch.split(i, factors=[2, 4])
sch.bind(i_0, "blockIdx.x")
sch.bind(i_1, "threadIdx.x")

# initialize input
A_np = np.arange(8).astype(dtype)
B_np = np.zeros((8,)).astype(dtype)

def build(target:str):
    tgt = tvm.target.Target(target=target, host="llvm")
    # build runtime module
    mod = tvm.build(sch.mod, target=tgt)
    # print CUDA/SYCL source code
    # print(mod.imported_modules[0].get_source())
    dev = tvm.device(target, 0)
    A_tvm = tvm.nd.array(A_np, dev)
    B_tvm = tvm.nd.array(B_np, dev)
    mod(A_tvm, B_tvm)
    return B_tvm

cuda_output = build(target="cuda")
sycl_output = build(target="sycl")
tvm.testing.assert_allclose(cuda_output, sycl_output, rtol=1e-5, atol=1e-5)
```

In addition, SYCL backend supports performance optimization using Auto-scheduler. Auto-scheduler sample code reference https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/tune_network_cuda.html, just specify target='sycl'.

**Currently Supported GPU kinds**:

- NVIDIA GPU
- AMD GPU
- Intel GPU

The following are two possible ways to **specify the GPU kind**:

- Set GPU kind when tvm compile. For example, `set(SYCL_GPU "nvidia")` in tvm `config.cmake`, then set `target="sycl"` in user code.
- Set GPU kind by Target. For example, set `target="sycl -gpu=nvidia"` in user code.

Which way is better needs more opinions and further discussion.

# Reference-level explanation

This RFC only adds the SYCL backend to TVM, no other features will be affected. For example, no existing passes of TVM need to modified.

**Added code**. The added code for SYCL backend mainly includes:

- SYCL codegen, from TIR to SYCL kernel code. The input of SYCL codegen is the abstract syntax tree  of TIR, SYCL codegen traverses the TIR syntax tree, and converts TIR to SYCL kernel code.
- SYCL runtime. SYCL host operations, such as memory copy, device information query, kernel submission, etc.

The added codegen and runtime should be compatible with the existing TensorIR infra.

**SYCL compiler.** There are some SYCL-aware compilers, such as DPC++, hipSYCL and ComputeCpp. [Open source DPC++](https://github.com/intel/llvm) is the most popular SYCL compiler, which built on LLVM and uses the Clang front end, SYCL 2020 standards. Intel Proposed Adding [Full SYCL Programming Model Support To Upstream LLVM](https://discourse.llvm.org/t/rfc-add-full-support-for-the-sycl-programming-model/74080). If the proposal is passed, the new version of clang++ will be used as SYCL compiler.

# Drawbacks
In order to make the SYCL backend compatible with the TVM runtime framework, this RFC requires runtime compilation tool for SYCL like NVRTC for cuda, which allows to compile codegen kernel code directly to an executable kernel at runtime. [SYCL's runtime compilation function is still under development](https://github.com/intel/llvm/pull/11985). Instead, this RFC compiles the SYCL kernel code into a dynamic link library for calling during TVM build. TVM build (for example, `tvm.build`) time increases due to the overhead time of compiling to a dynamic link library when `target='sycl'`. This is a temporary solution until SYCL's runtime compilation is available. If there are any problems, please let me know.

# Rationale and alternatives

# Prior art

[This repo](https://github.com/RELOAD22/tvm) has a basic implementation of SYCL codegen and runtime.

# Unresolved questions

# Future possibilities

- support more types of accelerator
- support TVM meta schedule and unity.
- add additional optimizations for specific hardware types