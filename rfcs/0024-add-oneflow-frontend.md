- Feature Name: (`add oneflow frontend`)
- Start Date: (2021-8-20)
- RFC PR: [apache/tvm-rfcs#0000](https://github.com/apache/tvm-rfcs/pull/)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/)

# Summary
[summary]: #summary

To enhance the compatibility of TVM with deep learning frameworks,
we have created a frontend for TVM that targets [oneflow](https://github.com/Oneflow-Inc/oneflow) 

# Motivation
[motivation]: #motivation

# TODO

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

We use a simple API to help users convert oneflow(eager) model to tvm relay

```python
relay.frontend.from_oneflow(graph, model_dir_path, freeze_params=True, user_input=None)
```

- graph: 
- model_dir_path: 
- freeze_params:
- user_input:

The following codes will show how to convert a oneflow model to a tvm relay

```python
import tvm
import tvm.relay as relay

import oneflow as flow


class Graph(flow.nn.Graph):
    def __init__(self, module):
        super().__init__()
        self.m = module

    def build(self, x):
        out = self.m(x)
        return out


# load eager model
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

More demos could be seen at [this](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/tvm_oneflow/oneflow_tvm)

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

# TODO

# Drawbacks
[drawbacks]: #drawbacks

# TODO(hujiakui)

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

# TODO(hujiakui)

# Prior art
[prior-art]: #prior-art

It's the first time we have added a OneFlow frontend to an ML compiler.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

We will add new unit test cases that rely on OneFlow framework, and this may increase time-cost of unit tests. If there are any proslems, please let me know.

# Future possibilities
[future-possibilities]: #future-possibilities

For this RFC, we have made an established plan,
# TODO
