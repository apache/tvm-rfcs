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

Based on this background, we propose this RFC to add the SYCL backend, enhancing the compatibility and portability of TVM across different types of accelerators.

# Guide-level explanation

**How to use?**

Similar to other backends such as cuda, specify `target='sycl'` in the corresponding TVM API.

```python
tgt = tvm.target.Target(target='sycl') #Target
……
lib = relay.build(mod, target='sycl', params=params) #model build
……
dev = tvm.device('sycl', 0) # Device that support sycl
input = tvm.nd.array(data, device=dev) #model input
```

The following sample code shows that operator `gemm` with CUDA and SYCL backends respectively, and compare whether the results of the two backends are consistent.

```python
import numpy as np
import tvm.relay as relay
from tvm.contrib import graph_executor
import tvm.testing
import tvm

# define GEMM
M = 1024
N = 1024
data_shape = (M, N)
dtype = 'float32'
X1 = relay.var("X1", shape=data_shape, dtype=dtype)
X2 = relay.var("X2", shape=data_shape, dtype=dtype)
func = relay.nn.dense(X1, X2)
mod = tvm.IRModule.from_expr(func)
# initialize input
X1_np = np.random.uniform(size=data_shape).astype(dtype)
X2_np = np.random.uniform(size=data_shape).astype(dtype)

def build(target:str):
    # model build
    tgt = tvm.target.Target(target=target, host="llvm")
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=tgt, params=None)
    # print CUDA/SYCL source code
    # print(lib.get_lib().imported_modules[0].get_source()) 
    dev = tvm.device(target, 0)
    module = graph_executor.GraphModule(lib["default"](dev))
    module.set_input("X1", X1_np)
    module.set_input("X2", X1_np)
    module.run()
    tvm_output = module.get_output(0).numpy()
    return tvm_output
    
cuda_output = build(target="cuda")
sycl_output = build(target="sycl")
tvm.testing.assert_allclose(cuda_output, sycl_output, rtol=1e-5, atol=1e-5)
```

In addition, SYCL backend supports performance optimization using Auto-scheduling. Auto-scheduling sample code reference https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/tune_network_cuda.html, just specify target='sycl'.

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

**SYCL compiler.** There are some SYCL-aware compilers, such as DPC++, hipSYCL and ComputeCpp. This RFC uses [Open source DPC++](https://github.com/intel/llvm), which built on LLVM and uses the Clang front end, SYCL 2020 standards.

# Drawbacks

SYCL does not support runtime compilation like NVRTC for cuda now, which allows to compile codegen kernel code directly to an executable kernel at runtime. In order to make the SYCL backend compatible with the TVM runtime framework, this RFC compiles the SYCL kernel code into a dynamic link library for calling during TVM build. TVM build (for example, `relay.build`) time increases due to the overhead time of compiling to a dynamic link library when `target='sycl'`. If there are any problems, please let me know.

# Rationale and alternatives

# Prior art

[This repo](https://github.com/RELOAD22/tvm) has a basic implementation of SYCL codegen and runtime.

# Unresolved questions

# Future possibilities

- support more types of accelerator
- support TVM meta schedule and TVM unity
- add additional optimizations for specific hardware types