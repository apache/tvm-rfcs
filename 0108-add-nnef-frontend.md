- Feature Name: `Relay NNEF frontend`
- Start Date: 2024-04-11
- RFC PR: [apache/tvm-rfcs#0108](https://github.com/apache/tvm-rfcs/pull/0108)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary
[summary]: #summary

Add the Khronos Neural Network Exchange Format (NNEF) as a frontend to TVM.

# Motivation
[motivation]: #motivation

NNEF is an open, standardized format for neural network exchange developed by the Khronos Group since 2018 (https://www.khronos.org/nnef). It is aimed at deploying trained neural networks from deep learning frameworks to proprietary inference engines of neural network hardware vendors. Such inference engines often require an offline compilation step for running models more efficiently, hence hardware vendors are are looing into open source compiler stacks to be leveraged. On one hand, hardware vendors may integrate their hardware as a backend into TVM, while at the same time integrating NNEF as a frontend would allow vendors to use TVM as an end-to-end compilation tool starting from a standardized format.

The Khronos Group also maintains a set of tools for handling NNEF models. Since NNEF is mainly a textual format, these include a parser (with C++ and Python interfaces), and conversion tools from other formats. NNEF supports conversion from models of various deep learning frameworks, including Caffe, TensorFlow (also Lite) and all those that support ONNX, such as PyTorch. Creating NNEF models is also possible manually by directly writing the model text file(s) (since NNEF is similar to a scripting language). Manually written models may even be executed or trained in deep learning frameworks (currently support for PyTorch exists).

For example, loading an NNEF model in Python is as simple as follows:

```python
import nnef
graph = nnef.load_graph('example.nnef')
```

The resulting graph object, containing tensors and operators can then be traversed and processed, for example converted into TVM representation, as done in this PR.

The NNEF tools also provide a simple C++ based reference implementation for NNEF models, whose main purpose is testing/debugging conversions, and serving as a reference for other more efficient inference backends. Furthermore, a PyTorch based interpreter is also supported, which is able to execute NNEF models via on/the-fly conversion to PyTorch calls, and can also be used as a (more efficient) reference.


# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

We are going to add support for models in NNEF format. The model may be provided either as an NNEF model folder, or an `nnef.Graph` object 
already loaded into memory.
The conversion is done via the new frontend function
```python
relay.frontend.from_nnef(model, freeze_vars=False)
```
  - model: either a string / PathLike to an NNEF model folder, or an `nnef.Graph` object.
  - freeze_vars: optional bool, which sets whether the parameters should be considered variables or constants for optimization

Example usages (assume we have a directory `inception_v1.nnef` with a complete NNEF Inception graph)
```python
import nnef
from tvm import relay

model_path = 'path/to/model/inception_v1.nnef'
# If modification is needed the graph can be imported with `nnef.load_graph` 
graph = nnef.load_graph(model_path)

mod, params = relay.frontend.from_nnef(graph)
```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

As this RFC only adds a new frontend, no other features should be affected. 

The process of importing an NNEF model consists of:

- Loading an NNEF model into memory, if a model path is provided, using `nnef.load_graph` function to get an `nnef.Graph` object.
After this step the model may be modified with functions provided for NNEF models before final conversion to TVM.
- Converting the operations of the Graph, setting inputs, and reading parameters one by one.


# Drawbacks
[drawbacks]: #drawbacks

Potential increase in time-cost of unit tests.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

The frontend of NNEF is similar to that of ONNX, PyTorch, and TensorFlow, adding it would increase the number of model formats that TVM can process.

# Prior art
[prior-art]: #prior-art

We are aware of the following projects that currently support importing NNEF models:

- https://aimotive.com/aiware
- https://github.com/sonos/tract
- https://github.com/fragata-ai/arhat-nnef
- https://rocm.docs.amd.com/projects/MIVisionX/en/latest/model_compiler/README.html
- https://www.khronos.org/openvx/

# Unresolved questions
[unresolved-questions]: #unresolved-questions

- Whether test cases can make use of pre-written the NNEF models, (text files with NNEF syntax, such as `graph.nnef`) as a starting point. Currently our test cases use separate model folders with prewritten model definitions, and we only generate the inputs for those. The 'tests/python/frontend/nnef/models' folder contains these test cases.
- Installation of NNEF and NNEF-Tools to the TVM CI Docker images. We need the Docker images to contain an install script which uses git to add NNEF to the CI environment, also with lint exceptions to `.nnef` files (mentioned in the previous point). It seems to work when the docker images are rebuilt from source with the install scripts added, but we are not sure if it okay.

# Future possibilities
[future-possibilities]: #future-possibilities

The Khronos Groups is actively working on the next major update to the NNEF format, whose main purpose is to increase model coverage by adding support for dynamic models and custom operators. In the latter case, more involved compilation of models carries even more potential, so we plan to add support for the next generation as well.

Support for some NNEF operators would only be possible through more complex mapping to a sequence of TVM operators, and the less widely used ones were not the focus of this initial release. We may add support to such operators in the future if required.
