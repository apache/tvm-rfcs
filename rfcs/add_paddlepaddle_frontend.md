- Feature Name: add-paddlepaddle-frontend
- Start Date: 2021-08-05
- RFC PR: https://github.com/apache/tvm-rfcs/pull/19
- GitHub Issue: TODO

# Summary
[summary]: #summary

Add a paddlepaddle frontend, enhancing TVM's compatibility for deep learning frameworks, which supports PaddlePaddle>=2.0

# Motivation
[motivation]: #motivation

PaddlePaddle, an independent R&D deep learning platform in China, has been officially open-sourced to professional communities since 2016. It has been widely adopted by a wide range of sectors including manufacturing, agriculture, enterprise service, and so on while serving more than 2.3 million developers. With such advantages, PaddlePaddle has helped an increasing number of partners commercialize AI.

Currently, PaddlePaddle has built a prosperous technological ecology, there are more than 500 models developed by official organization or outside developers, covering CV/NLP/OCR/Speech, refer to the following links for more details,

- [PaddlePaddle/models](https://github.com/PaddlePaddle/models)
- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas)
- [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)
- [DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech)

As of version 2.0, PaddlePaddle supports imperative programming like PyTorch. Furthermore, a mechanism of `Dynamic to Static` is provided to export PaddlePaddle a model in graph representation, which is more friendly for deployment. The following example code shows how to export a PaddlePaddle model,

```
import paddle
import paddlehub
model = hub.Module(name="resnet50_vd_imagenet_ssld")
input_spec = paddle.static.InputSpec(
    [1, 3, 224, 224], "float32", "image")
paddle.jit.save(model, "model/infer", input_spec=[input_spec])
```

PaddlePaddle's deployment is supported by Paddle Inference/Paddle Lite/OpenVINO/Tengine/Adlik now. We noticed that there are lots of developers converting models to ONNX format for the compatibility with TVM, but only a limited number of models are convertible due to lack of ONNX operators.  
Based on this background, we proposed this RFC PaddlePaddle frontend for TVM, improving usability for PaddlePaddle users and enhancing the compatibility between PaddlePaddle and TVM.


# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

If you dive into the pull request code, there are 2 concepts imported from PaddlePaddle that you may want to know,
- `paddle.jit.load`: Recommended API to load an exported inference model, the type of the return value is `TranslatedLayer`, storing `Program`(similar to computation graph) and parameters;
- `paddle.static.load_inference_model`: API compatible with older version PaddlePaddle models, the type of the return value is `Program`. All the parameters are saved in `Scope` by default, parameters can be extracted from the `paddle.fluid.global_scope()`.

Therefore, this RFC will also add a new API to TVM to support PaddlePaddle models,
```
relay.frontend.from_paddle(program_or_layer, shape_dict=None, scope=None)
```
- `program_or_layer`: the return value of `paddle.static.load_inference_model` or `paddle.jit.load`
- `shape_dict`: optional, input shapes of the model
- `scope`: optional, which is available only if `model` is loaded using `paddle.static.load_inference_model`

The following example code shows how to import a PaddlePaddle model,
```
import paddle
model = paddle.jit.load('model/infer')

shape_dict = {'image': [1, 3, 224, 224]}
mod, params = relay.frontend.from_paddle(model, shape_dict=shape_dict)
```

Errors may happen if there exist some operators in the model that are not supported by this frontend. If so, details will be printed out.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

Since this RFC only aims to add a new frontend for converting PaddlePaddle models to TVM Relay IR, no other features will be affected.

In this proposed RFC, the whole process of PaddlePaddle frontend importing can be divided into 2 steps:
- 1. Reading a PaddlePaddle Model: The frontend supports models in PaddlePaddle's inference model format, which are exported as graph based models by PaddlePaddle's `Dynamic to Static` mechanism. The model contains 2 files storing the model structure and parameters respectively. We use `paddle.jit.load` to load the model files (For the compatibility with versions of PaddlePaddle below 2.0, `paddle.static.load_inference_model` is also supported); 
- 2. Operators Conversion: After the exported inference model is loaded, we will extract its parameters and convert operators one by one. Since all the operators are transversed according to topological ordering, there's no need to worry about the order of converting the operators. 

# Drawbacks
[drawbacks]: #drawbacks

Potential increase in time-cost of unit tests.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

The frontend of PaddlePaddle is similar to ONNX and TensorFlow. We support model loaded by `paddle.jit.load` and `paddle.static.load_inference_model`. We considered supporting `paddle.jit.load` only since this API is recommended as of PaddlePaddle 2.0, but there are lots of users still using older versions. Thus, supporting `paddle.static.load_inference_model` is still necessary.
Currently, we have to convert PaddlePaddle models to ONNX format to make them work with TVM, but only a limited number of models are supported due to the lack of ONNX operators and the operator differences. With a new PaddlePaddle frontend, we can support more operators and provide a better experience for TVM and PaddlePaddle's users.

# Prior art
[prior-art]: #prior-art

It's the first time we have added a PaddlePaddle frontend to an ML compiler.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

We will add new unit test cases that rely on PaddlePaddle framework, and this may increase time-cost of unit tests. If there are any problems, please let me know.

# Future possibilities
[future-possibilities]: #future-possibilities

For this RFC, we have made an established plan,

- About 200 operators will be supported in this quarter, such as deformable_conv/multiclass_nms
- Control flow operators will be supported this year, mainly refer to while_loop/if
- Quantized model will be supported this year, including quantized model obtained from `PaddleDetection`/`PaddleClas`/`PaddleSeg`
