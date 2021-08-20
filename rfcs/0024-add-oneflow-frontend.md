- Feature Name: (add oneflow frontend)
- Start Date: (2021-8-20)
- RFC PR: [apache/tvm-rfcs#0024](https://github.com/apache/tvm-rfcs/pull/0024)
- GitHub Issue: [apache/tvm#8804](https://github.com/apache/tvm/issues/8804)

# Summary
[summary]: #summary

To enhance the compatibility of TVM with deep learning frameworks,
we have created a frontend for TVM that targets [oneflow](https://github.com/Oneflow-Inc/oneflow) 

# Motivation
[motivation]: #motivation

OneFlow, an open source deep learning framework with whole new frame design and the world's leading technology for distributed system. Here are advantages of OneFlow:

- Perfectly support container platforms(k8s & docker)
- Handle large models easily
- Almost zero runtime overhead & linear speedup
- Support automatic mixed precision
- ...

We are proud that OneFlow can support basic CNNs as well as very large pre-trained models, such as [GPT3, BERT, etc](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/master/LanguageModeling). They are built using an early version of OneFlow, based on lazy mode.

Currently, oneflow(nightly) supports the conversion of eager models to lazy graphs. We can quickly convert the eager model built by OneFlow to a lazy graph with the following code.

```python
import oneflow as flow


class Graph(flow.nn.Graph):
    def __init__(self, module):
        super().__init__()
        self.m = module

    def build(self, x):
        out = self.m(x)
        return out


# module: eager model built by OneFlow
graph = Graph(module)
```

Because of these features we wrote this `from_oneflow` which is based on lazy mode. 

We also note that many developers have converted their models to ONNX format for compatibility with TVM, and this part of the work is ongoing, as you can see [here](https://github.com/Oneflow-Inc/oneflow_convert_tools).

Based on this background, we proposed this RFC to add a OneFlow frontend for TVM, improving usability for OneFlow users and enhancing the compatibility between OneFlow and TVM.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

We use a simple API to help users convert oneflow to tvm relay.

```python
relay.frontend.from_oneflow(graph, model_dir_path, freeze_params=True, user_input=None)
```

- graph: flow.nn.Graph, contains information about the nodes of the model
- model_dir_path: str, path of parameters
- freeze_params: bool, if this parameter is False, then the user can specify the input of the  graph
- user_input: dict, information about the specified input of the model

> NOTES:
> We prefer to let the user change the model node information by changing the model itself

The following codes will show how to convert a oneflow model to a tvm relay

```python
import tvm
import tvm.relay as relay


# load eager model(assuming that resnet50 has been built)
res50_module = resnet50()
pretrain_models = flow.load(model_path)
res50_module.load_state_dict(pretrain_models)
res50_module.eval().to("cuda")

# load test image
image = load_image(image_path)
image_flow = flow.Tensor(image, device=flow.device("cuda"))

# use Graph convert eager to lazy
res50_graph = Graph(res50_module)
_ = res50_graph._compile(image_flow)

# get tvm relay
mod, params = relay.frontend.from_oneflow(res50_graph, model_path)
```

If user writes a model that contains an op that is not yet supported, the program will report an error.

More demos could be seen at [this](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/tvm_oneflow/oneflow_tvm).

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

Since the purpose of this RFC is only to add a new front-end for converting the OneFlow model to TVM Relay IR, other functions will not be affected.

In this proposed RFC, the whole process of OneFlow frontend conversion can be divided into 2 steps:

1. Parse Graph: Read the node information contained in the graph, extract the size, data type of each node and construct the computational graph by each computational node (in oneflow, these nodes are marked as user_conf). At the same time, we get the data information of the parameter nodes (in oneflow, these nodes are labeled as variable_conf) according to the paths where the parameters are provided by the user.
2. Convert operators: After the exported inference graph is loaded, we will extract its parameters and convert operators one by one. The order of all operator conversions will be based on the flow of the graph.

# Drawbacks
[drawbacks]: #drawbacks

Potential increase in time-cost of unit tests.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

The frontend of OneFlow is similar to ONNX. For now, we consider only supporting conversion via `flow.nn.Graph`, as this API will be rolled out and recommended starting with OneFlow 0.5.0. In the meantime, we will develop the latest OneFlow eager model to ONNX format conversion script based on the existing OneFlow lazy model conversion to ONNX format to make it easy to work with TVM. 

# Prior art
[prior-art]: #prior-art

It's the first time we have added a OneFlow frontend to an ML compiler.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

We will add new unit test cases that rely on OneFlow framework, and this may increase time-cost of unit tests. If there are any problems, please let me know.

# Future possibilities
[future-possibilities]: #future-possibilities

For this RFC, we have made an established plan,

- Support OneFlow Eager to ONNX
- Some operators will be supported in this quarter
