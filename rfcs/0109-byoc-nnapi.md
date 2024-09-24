- Feature Name: byoc_nnapi
- Start Date: 2024-08-01
- RFC PR: [apache/tvm-rfcs#0109](https://github.com/apache/tvm-rfcs/pull/0109)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary
[summary]: #summary

This RFC introduces a new backend Android Neural Network API (NNAPI) for BYOC.

# Motivation
[motivation]: #motivation

Android Neural Networks API (NNAPI) is a graph-level neural network inference API provided by the Android runtime. Prior to this RFC, TVM on Android mobile devices mainly relies on OpenCL for GPU acceleration. This RFC aims to add a new codegen and a runtime via the BYOC framework, which enables execution on custom accelerators from SoC vendors on mobile devices.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

**How to use the NNAPI BYOC backend?**

Use the `partition_for_nnapi()` function to partition operations that are supported by NNAPI from an `IRModule`. The optional `feature_level` keyword argument specifies the highest NNAPI feature level. Operations introduced in feature levels higher than the specified level do not get partitioned.

```python
from tvm.relax.op.contrib.nnapi import partition_for_nnapi

mod = partition_for_nnapi(mod, feature_level=7)
```

Build the module after partitioning. The result of the build can then be exported and deployed to an Android device built with the NNAPI runtime support turned on.

```python
android_target = "llvm -mtriple=aarch64-linux-android"
lib = relax.build(mod, target=android_target)
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

This RFC adds optional support for NNAPI via BYOC without affecting other features in TVM.

**Added code**:

We have an implementation with the following components added to the TVM codebase.

- NNAPI partition function implemented with pattern matching.
- NNAPI codegen that serializes Relax IR subgraphs to JSON runtime modules.
- NNAPI runtime that loads JSON runtime modules and calls API functions to perform model build, compile, and inference.

**Supported ops**:

The implementation supports the following ops in both `float32` and `float16` data types.

- Element-wise unary operations (relu, exp, …)
- Element-wise binary operations (add, multiply, …)
- nn.dense
- nn.conv2d
- nn.max_pool2d

# Drawbacks
[drawbacks]: #drawbacks

In the current implementation, the performance gain of NNAPI is not consistent on the mobile devices due to SoC drivers being unable to accelerate all of the supported operations. This may be mitigated by further integrating a smarter partitioning algorithm that selectively offloads operations based on profiling as seen in the [Prior art](#prior-art) section.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

Instead of using JSON codegen, the integration can also be implemented using C source codegen. See the [Prior art](#prior-art) section.

# Prior art
[prior-art]: #prior-art

This RFC is a successor of [an RFC by us](https://discuss.tvm.apache.org/t/rfc-byoc-android-nnapi-integration/9072) in 2021. The codegen and the runtime has been rewritten from scratch since then to generate and load standardized `JSONRuntimeBased` modules instead of C source code.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

# Future possibilities
[future-possibilities]: #future-possibilities

- Add support for quantized data types to cover Relax QNN dialect or Relax quantize/dequantize operators.
- Add support for dynamic shape operands.
