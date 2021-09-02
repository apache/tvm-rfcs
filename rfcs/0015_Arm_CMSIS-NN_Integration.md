- Feature Name: [RFC] Use CMSIS-NN with TVM
- Start Date: July 2021
- RFC PR: https://github.com/apache/tvm-rfcs/pull/15
- GitHub Issue: https://github.com/apache/tvm/issues/8646

# Acronyms
* CMSIS: Common Microcontroller Software Interface Standard
* ACL: The Compute Library for the Arm® Architecture
* MLF: Model Library Format
* Cortex-M: Arm® Cortex®-M processor

# Summary

This RFC introduces plan of integration of CMSIS-NN library into TVM. It consists of efficient kernels targeted for Cortex-M architecture.

Please refer to the following pages for more details on CMSIS-NN.
* [CMSIS-NN user manual](https://arm-software.github.io/CMSIS_5/NN/html/index.html)
* [GITHUB CMSIS-NN Source](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/NN)

First PR in the series of PRs to fulfill this integration would be graph partitioner for softmax int8. Detailed plan can found below in this RFC.


# Motivation

CMSIS-NN library consists of hand-tuned kernels that are suitable for Cortex-M and are compliant with the quantization scheme used in Tensorflow Lite. They have been optimized for better performance and small memory footprint which is required on these embedded devices and it would make sense for TVM to reuse these while generating code for Cortex-M. They have been integrated with the TensorFlow Lite Micro project. In this work, we plan to map TFLite operators to the existing CMSIS-NN APIs without performing any intermediate Relay level translations.


# Guide-level explanation

We will enable this integration by considering TFLite networks, but is equally applicable for all other networks that can be translated into Relay IR.

TVM's BYOC infrastructure allows for the partitioning and code generation using the external compiler. Partitioned subgraphs containing operator(s) targeted for Cortex-M can then be translated into the CMSIS-NN C APIs which eventually become part of MLF.

If a user runs tvmc, they will get a MLF format archive which calls out to the CMSIS operators. The source for the CMSIS-NN is not included in the MLF. Also, the support will remain up to date with changing library as we expect minimal changes to the CMSIS-NN API interface. Source code from github will be used for linking against the MLF by the test setup that allows execution on Cortex-M.

```
tvmc --target=cmsisnn,c --output-format=mlf --executor=aot
```
In the absence of tvmc support, following python APIs can be used to generate the C code. But eventually tvmc will be supporting CMSIS-NN as mentioned above.

```python
from tvm.relay.op.contrib import cmsisnn

# API to call CMSIS-NN partitioning
# Here, module is the relay module
cmsisnn_module = cmsisnn.partition_for_cmsisnn(module)

# Invoke AOT compiler to get the MLF containing CMSIS-NN APIs
with tvm.target.Target("c -runtime=c --link-params -mcpu=cortex-m55 --executor=aot --unpacked-api=1"):
    factory = tvm.relay.build(cmsisnn_mod)
```

# Reference-level explanation

This section details how TFLite softmax int8 is converted into the C code. TFLite frontend first translates softmax int8 into the following sequence of relay operations: *dequantize -> softmax -> quantize*. Please refer to the relay code snippet below obtained from TFLite frontend.

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

The API for partitioning will work through the pattern matching tables for CMSIS-NN which will look like the below snippet. It will include support for operators: Convolution, Depthwise convolution, Fully Connected, Pooling and MatMul.

```python
@register_pattern_table("cmsisnn")
def pattern_table():
    """Get the cmsisnn compiler pattern table."""

    def softmax_pattern():
        pattern = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
        pattern = is_op("nn.softmax")(pattern)
        pattern = is_op("qnn.quantize")(pattern, is_constant(), is_constant())
        return pattern

    def check_quantized_softmax(extract):
       ...

    return [
        ("cmsisnn.qnn_softmax", softmax_pattern(), check_quantized_softmax),
    ]

```

Following code block shows the resultant IRModule post partitioning.

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

Above partitioned function is presented to the CMSIS-NN external code generator for TIR generation using the TVM's build() API.

```python
# Invoke AOT compiler to get the MLF containing CMSIS-NN APIs

with tvm.target.Target("c -runtime=c --link-params -mcpu=cortex-m55 --executor=aot --unpacked-api=1"):
    factory = tvm.relay.build(cmsisnn_mod)
```

Resultant TIR looks like this:

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

In future, target hooks for `relay_to_tir` implemented as part of [Additional Target Hooks](https://github.com/apache/tvm-rfcs/pull/10) will be used to obtain the above TIR and it will be returned to the compilation pipeline. These hooks provide us with the flexibility to reuse memory planning and much of the TVM's code generation capabilities.

At last, code generator identifies the TIR extern_call(s) and generates C code for softmax with the CMSIS-NN API for softmax int8. Both TIR and C are generated when function registered through `tvm.register_func("relay.ext.cmsisnn")` is invoked.

```c++
#ifdef __cplusplus
extern "C" {
#endif
// C code generator produces hard coded values from the network
static const int32_t num_rows = 28;
static const int32_t row_size = 28;
static const int32_t mult = 1;
static const int32_t shift = 0;
static const int32_t diff_min = -128;

static int32_t tvmgen_default_cmsisnn_main_0_(int8_t* in0, int8_t* out0) {

    arm_softmax_s8(in0, num_rows, row_size, mult, shift, diff_min, out0);
    return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
TVM_DLL int32_t tvmgen_default_ethosu_main_0(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code) {
  DLTensor* arg0 = (DLTensor*)(((TVMValue*)args)[0].v_handle);
  DLTensor* ret1 = (DLTensor*)(((TVMValue*)args)[1].v_handle);
  return tvmgen_default_cmsisnn_main_0_(arg0, ret1);
}
#ifdef __cplusplus
}
#endif
```

Note: CMSIS-NN APIs for each operator are hard coded into the generated C file. The C generator can be excluded from the source by setting USE_CMSISNN to OFF in the config.cmake. In orde to link the C file to the CMSIS-NN library, Ethosu test runner infrastructure is used as has been described here: [Arm Ethos-U Integration](https://github.com/apache/tvm-rfcs/pull/11).

Once the entire infrastructure for CMSIS-NN mapping is in place using softmax API, we will add more complex operations such as depthwise convolution and pooling gradually to both the graph partitioning and code generation infrastructure.


# Testing

Unit tests will be added alongside operator support. Once operator support matures, we will add network tests.

A unit tests will be of two kinds.

* Match operator patterns used by the graph partitioner.
    * It will be done for each operator and for a combination of operators both.
* Correctness of the CMSIS-NN operators against the native TVM output.
    * Actual output can be generated using [Corstone-300 reference system](https://github.com/apache/tvm-rfcs/pull/11)
    * In case the reference system is unavailable, checks will be added for TIR's correctness.


# Drawbacks

CMSIS-NN APIs provide hand coded kernels. Therefore, code generation skips the auto tuning capabilities of TVM. In future, we wish to make use of full power of TVM's auto scheduling.

# Prior art

CMSIS-NN integration into TVM builds on top of ACL's integration into TVM. Existing infrastructure of BYOC allows for graph partitioning to detach the operators or chain of operations as a separate subgraph that then can be compiled for Cortex-M.

Reference: [ACL](https://tvm.apache.org/docs/deploy/arm_compute_lib.html)

Evenutally, code generation for CMSIS-NN will use the newly introduced target hooks.

Reference: [Additional Target Hooks](https://github.com/apache/tvm-rfcs/pull/10/files)
