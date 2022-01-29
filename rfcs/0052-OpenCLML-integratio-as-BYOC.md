- Feature Name: OpenCL ML integration as BYOC
- Start Date: 2022-01-13
- RFC PR: [apache/tvm-rfcs#52](https://github.com/apache/tvm-rfcs/pull/52)
- GitHub Issue: TBD


# Summary
[summary]: #summary

OpenCL ML is an extension (cl_qcom_ml_ops) over OpenCL spec developed by Qualcomm to accelerate the machine learning at operation level. OpenCL SDK is publicly available at OpenCL Machine Learning Acceleration on Adreno GPU - Qualcomm Developer Network. OpenCL ML leverages deep knowledge of Adreno GPU for significant performance benefits. It offers C based DNN API with compatibility to most of the standard frameworks. Its standard OpenCL features like command queues, buffers, events and supports FP16 and FP32 data types. CLML API calls can be interleaved with other OpenCL kernels (i.e., TVM generated kernels) and dispatched to the same command queue. This extension is compatible with existing OpenCL extensions for importing memory, controlling performance and data access.

# Motivation
[motivation]: #motivation

The current OpenCL backend of TVM is very generic and not optimized well for Adreno performance capabilities. Adreno GPU has quite a few proprietary and standard OpenCL paths. OpenCL ML extension offers accelerated ML operations via an SDK interface.

With TVM having the entire framework of frontends, graph level optimizations and OpenCL ML having kernels that perform best on Adreno GPU, in this work we aim to integrate OpenCLML SDK into TVM as a BYOC. This effort brings best of both worlds where TVM handling high level optimizations, sub graphs are scheduled on OpenCL ML based on the support and the operators not supported by OpenCL ML will take TVM’s default OpenCL path. Good thing here is we don’t need separate OpenCL workspaces or command queues for both paths, instead they can share the command queues. Also, data (DLTensor) transfer across subgraphs is seamless with OpenCL ML API’s.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

This RFC aims to introduce OpenCLML runtime as a BYOC option into TVM. In terms of usage, it’s very similar to other BYOC integrations we have in TVM.

Along with all other options we use for OpenCL target, here we introduce the below build options in config.cmake

```USE_CLML``` (ON/OFF) This enables CLML codegen for compilation

```USE_CLML_GRAPH_EXECUTOR``` (ON/OFF) This enables CLML runtime
Btw, OpenCLML SDK provides replacement for default libOpenCL.so. Hence, we don’t need a separate option to point OpenCLML SDK instead just point OpenCLML SDK path for USE_OPENCL.

Introduces front end helper API as ```tvm.relay.op.contrib.clml```. This will help to partition the graph and annotating the subgraphs to OpenCL CLML target.

Given mod and params that represents TVM Module and params the below API does partitioning based on OpenCLML support.

```mod = clml.partition_for_clml(mod, params)```

Post above partitioning we just follow standard ```relay.build``` process.

Talking about runtime, OpenCL ML runtime compilation is same as OpenCL compilation for Android target. Just that USE_OPENCL points to OpenCL ML SDK.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation


Like any other BYOC implementation this RFC enhances/introduces a frontend helper API for partitioning, a codegen for CLML and CLML runtime.

### Frontend:
Front end implements ```tvm.relay.op.contrib.clml``` user API ```partition_for_clml``` and ```is_clml_runtime_enabled``` for partitioning the relay graph to OpenCLML path. It also contains clml specific patten table definition and other transform helpers required for CLML target.

### Codegen:
CLML codegen built over JSONSerializer. Thanks to JSONSerializer for all the infra here and one can focus only on target specific parsing and JSON Node generation. The codegen exports ```relay.ext.clml```, ```relay.op.is_clml_runtime_enabled``` into TVM global space.

### Runtime:
OpenCLML Runtime is again extended over JSONRuntimeBase and implements OpenCL ML initialization, implementation of CLML API invocation corresponding to CLML annotated layers.

OpenCLML runtime support is verified by looking for ```cl_qcom_ml_ops``` into OpenCL extension list.

OpenCLML doesn’t define a new open context instead it reused the context defined by OpenCL runtime through global API ```device_api.opencl```.

OpenCLML has its own CLML tensor objects called ```cl_ml_tensor_memory_desc_qcom```. The runtime defines the copy API from OpenCL to CLML Tensors within the same OpenCL work space without bringing the data back to host.

OpenCLML supports tuning too which generally produces a tuning cache file and reuses for later runs. This implementation supports looking for environment variable ```CLML_IS_TUNNING_RUN``` set to 0/1 to run for tuning and also supports ```CLML_TUNNING_CACHE``` to set the tuning cache file location. While implementation CLML tuning happens at last step of ```BuildEngine``` by calling ```clTuneMLOpQCOM``` followed by ```clSaveMLTuningCacheQCOM``` for saving the cache to given file. Later we set ```CLML_IS_TUNNING_RUN``` to ```0``` and use ```clLoadMLTuningCacheQCOM``` for reloading the cache with out tuning.

# Drawbacks
[drawbacks]: #drawbacks


OpenCLML is supported by Snapdragon devices only with extension ```cl_qcom_ml_ops```. Seamless copy from OpenCL to CLML is supported for clBuffers now. Using Image objects on TVM may have challenges for direct copy within OpenCL context.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives


OpenCL ML uses Adreno specific proprietary and public optimization paths and outperforms TVM generated OpenCL kernels by a big difference.

# Prior art
[prior-art]: #prior-art

There exists an ongoing development for texture support on Adreno devices https://discuss.tvm.apache.org/t/rfc-texture-memory-support/9467.

# Unresolved questions
[unresolved-questions]: #unresolved-questions


How do we deal with sub graphs with tiny layers? This is the case where not offloading the tiny layer performs better than accelerator.

# Future possibilities
[future-possibilities]: #future-possibilities

Integrating OpenCLML into TVM gives an end-to-end compiler stack for Snapdragon platform with Adreno GPU target. Operator support evolves along with OpenCL ML SDK releases from Qualcomm.
