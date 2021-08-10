- Feature Name: [RFC] Use CMSIS-NN with TVM
- Start Date: July 2021
- RFC PR: https://github.com/apache/tvm-rfcs/pull/15
- GitHub Issue: https://github.com/apache/tvm/issues/8646

# Acronyms
CMSIS: Common Microcontroller Software Interface Standard
ACL: The Compute Library for the Arm® Architecture
MLF: Model Library Format

# Summary

This RFC introduces plan of integration of CMSIS-NN library into TVM. It consists of efficient kernels targeted for Arm's Cortex-M architecture.

Please refer to the following pages for more details on CMSIS-NN.
https://arm-software.github.io/CMSIS_5/NN/html/index.html
https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/NN

First PR in the series of PRs to fulfill this integration would be graph partitioner for softmax int8. Detailed plan can found below in this RFC.


# Motivation

CMSIS-NN library consists of hand-tuned kernels that are suitable for Cortex-M and are compliant with the quantization scheme used in Tensorflow Lite. They have been optimized for better performance and small memory footprint which is required on these embedded devices and it would make sense for TVM to reuse these while generating code for Cortex-M. They have been integrated with the TensorFlow Lite Micro project.


# Guide-level explanation

TVM's external code generation infrastructure allows for the automatic partitioning and code generation using the external compiler. Partitioned subgraphs containing operator(s) targeted for Cortex-M can then be translated into the CMSIS-NN C APIs which eventually become part of MLF. For this integration, we are heavily dependent on the TVM's infrastructure for external code generation.

If a user runs tvmc, they will get a MLF format archive which calls out to the CMSIS operators.

```
tvmc --target=cmsisnn,c --output-format=mlf --executor=aot
```


# Reference-level explanation

We will enable this integration by considering TFLite networks, but is equally applicable for all other networks that can be translated into Relay IR. TFLite test that contains just a quantized (int8) softmax is first converted as a sequence of following relay operations: *dequantize -> softmax -> quantize* by the TFLite frontend. Please refer to the code snippet below.

```python
def @main(%a: Tensor[(1, 16, 16, 3), int8]) -> Tensor[(1, 16, 16, 3), int8] {
  %0 = qnn.dequantize(%a, 0.02f /* ty=float32 */, 64 /* ty=int32 */) /* ty=Tensor[(1, 16, 16, 3), float32] */;
  %1 = nn.softmax(%0) /* ty=Tensor[(1, 16, 16, 3), float32] */;
  qnn.quantize(%1, 0.02f /* ty=float32 */, 64 /* ty=int32 */, out_dtype="int8") /* ty=Tensor[(1, 16, 16, 3), int8] */
}
```

Here is the API to obtain the partitioned function aimed at CMSIS-NN.

```python
    # API to call CMSIS-NN partitioning
    from tvm.relay.op.contrib import cmsisnn
        # Here, module is the relay module
        cmsisnn_module = cmsisnn.partition_for_cmsisnn(module)        
```

Following code block shows the resultant IRModule.

```python
def @main(%a: Tensor[(1, 16, 16, 3), int8]) -> Tensor[(1, 16, 16, 3), int8] {
  @tvmgen_default_cmsisnn_0(%a) /* ty=Tensor[(1, 16, 16, 3), int8] */
}

def @tvmgen_default_cmsisnn_0(%cmsisnn_0_i0: Tensor[(1, 16, 16, 3), int8], Inline=1, Compiler="cmsisnn", global_symbol="tvmgen_default_cmsisnn_0", Primitive=1) -> Tensor[(1, 16, 16, 3), int8] {
  %2 = fn (%FunctionVar_0_0: Tensor[(1, 16, 16, 3), int8], PartitionedFromPattern="qnn.dequantize_nn.softmax_qnn.quantize_", Composite="cmsisnn.qnn_softmax") -> Tensor[(1, 16, 16, 3), int8] {
    %0 = qnn.dequantize(%FunctionVar_0_0, 0.02f /* ty=float32 */, 64 /* ty=int32 */) /* ty=Tensor[(1, 16, 16, 3), float32] */;
    %1 = nn.softmax(%0) /* ty=Tensor[(1, 16, 16, 3), float32] */;
    qnn.quantize(%1, 0.02f /* ty=float32 */, 64 /* ty=int32 */, out_dtype="int8") /* ty=Tensor[(1, 16, 16, 3), int8] */
  };
  %2(%cmsisnn_0_i0) /* ty=Tensor[(1, 16, 16, 3), int8] */
}
```

Above partitioned function is presented to the CMSIS-NN external code generator for *tir* generation using the TVM's build() API. 

```python
    # Invoke AOT compiler to get the MLF containing CMSIS-NN APIs
    with tvm.target.Target("c -runtime=c --link-params -mcpu=cortex-m55 --executor=aot --unpacked-api=1"):
        factory = tvm.relay.build(cmsisnn_mod)

```

Intermediate *tir* looks like this:

```python
primfn(placeholder_1: handle, out_write_1: handle) -> ()
    attr = {"global_symbol": "main", "tir.noalias": True}
    buffers = {placeholder: Buffer(placeholder_1: Pointer(int8), int8, [1, 300, 300, 3], []),
                out_write: Buffer(out_write_1: Pointer(int8), int8, [1, 300, 300, 3], [])}
    buffer_map = {placeholder_1: placeholder_1, out_write_1: out_write_1} {
    ...
    allocate(placeholder.d.global, uint8, [1,300,300,3]) {
        @tir.call_extern("cmsisnn_softmax_s8", ..., dtype=handle)
    }
}
```
In future, target hooks for `relay_to_tir` implemented as part of [Additional Target Hooks] (https://github.com/apache/tvm-rfcs/pull/10) will be used to obtain the above tir for graph with softmax. These hooks provide us with the flexibility to reuse memory planning and much of the TVM's code generation capabilities.

At last, code generator identifies the *tir* extern_call(s) and generates *c* code for softmax with the CMSIS-NN API for softmax int8.

Note: There are no changes required in config.cmake as the CMSIS-NN APIs corresponding to the operators are hard coded. The testing infrastructure links them to the CMSIS-NN library. Execution of the networks works similar to what has been described in [Arm Ethos-U Integration] (https://github.com/apache/tvm-rfcs/pull/11).

For more complex operations, CMSIS-NN structures will need to be used. For this purpose, `tir_to_runtime` will be used to extend the existing C Codegen and produce C code with the appropriate headers and calling patterns. Please refer to the [Additional Target Hooks] (https://github.com/apache/tvm-rfcs/pull/10).


# Testing

As we introduce the operators, we will keep on adding individual unit tests. Once the operator support is partially completed, we will start adding network tests. We are planning to use [Arm® Corestone™-300 Fixed Virtual Platform] (https://developer.arm.com/ip-products/subsystem/corstone/corstone-300) to run these tests in the CI. Reference: [Arm Ethos-U Integration] (https://github.com/apache/tvm-rfcs/pull/11/files). In its absence, tests to check the correctness of *tir* can be added.

# Drawbacks

CMSIS-NN APIs provide hand coded kernels. Therefore, code generation skips the auto tuning capabilities of TVM. In future, we wish to make use of full power of TVM's auto scheduling.

# Upstreaming Plan

Before adding other operators from CMSIS-NN, the integration will be enabled only for softmax.

P1: Graph partitioner for CMSIS-NN target
P2: Code generation using existing BYOC
P3: tvmc support to generate code for CMSIS-NN
P4: Move this implementation using `tir_to_runtime` from target hooks
P5: Use of CMSIS-NN data structures while supporting depthwise convolution
P6: Support for Convolution
P7: Support for Fully connected
P8: Support for Max Pooling
P9: Support for Avg Pooling
P10: Support for MatMul


# Prior art

CMSIS-NN integration into TVM builds on top of ACL's integration into TVM. Existing infrastructure of BYOC allows for graph partitioning to detach the operators or chain of operations as a separate subgraph that then can be compiled for Cortex-M.

Reference: [ACL] (https://tvm.apache.org/docs/deploy/arm_compute_lib.html)

Code generation for CMSIS-NN will use the newly introduced target hooks.

Reference: [Additional Target Hooks] (https://github.com/apache/tvm-rfcs/pull/10/files)
