- Feature Name: add-paddlepaddle-frontend
- Start Date: 2021-08-08
- RFC PR: TODO
- GitHub Issue: TODO

# Summary
[summary]: #summary

Add a paddlepaddle frontend, enhance TVM's campatibility of deep learning frameworks, which support PaddlePaddle>=2.0

# Motivation
[motivation]: #motivation

PaddlePaddle,  an independent R&D deep learning platform in China, has been officially open-sourced to professional communities since 2016. It has been widely adopted by a wide range of sectors including manufacturing, agriculture, enterprise service, and so on while serving more than 2.3 million developers. With such advantages, PaddlePaddle has helped an increasing number of partners commercialize AI.

Currently, PaddlePaddle has built a prosperous technological ecology,  there are more than 500 models developed by official organization or outside developers, including CV/NLP/OCR/Speech,  for more details we can refer to the following links,

- [PaddlePaddle/models](https://github.com/PaddlePaddle/models)
- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas)
- [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)
- [DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech)

After upgrading to 2.0, PaddlePaddle supported imperative programming similar with PyTorch, but a mechanism of `Dynamic to Static` is provided, which can export PaddlePaddle model as graph representation and more friendly for deployment, the following example code shows how to export a PaddlePaddle model,

```
import paddle
import paddlehub
model = hub.Module(name="resnet50_vd_imagenet_ssld")
input_spec = paddle.static.InputSpec(
    [1, 3, 224, 224], "float32", "image")
paddle.jit.save(model, "model/infer", input_spec=[input_spec])
```

PaddlePaddle's deployment is supported by Paddle Inference/Paddle Lite/OpenVINO/Tengine/Adlik now. We have noticed there are lots of developers convmodel to ONNX format for TVM's supporting,  but only part of models can be converted due to the lack of ONNX operators.  
Based on this background, we proposed this RFC addle frontend for TVM,  improve usability  and extend more models support for PaddlePaddle's users.


# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

If you dive in the pull request code, there's 2 concepts imported from PaddlePaddle you may want to know,
- `paddle.jit.load`: Recommended API to load exported inference model, the type of return result is `TranslatedLayer`, stores `Program`(similar with computation graph) and parameters;
- `paddle.static.load_inference_model`: API to compatible with old version PaddlePaddle's model, the type of return result is `Program`, and all the parameters save in `Scope`, for the default situation, we can extract the parameters from the `paddle.fluid.global_scope()`.

So, this RFC also will bring a new API for TVM to support PaddlePaddle model,
```
relay.frontend.from_paddle(program_or_layer, shape_dict=None, scope=None)
```
- `program_or_layer`: the return result of `paddle.static.load_inference_model` or `paddle.jit.load`
- `shape_dict`: optional parameter, input shapes of the model
- `scope`: optional parameter, only available if `model` is loaded by `paddle.static.load_inference_model`

The following example code shows how to import a PaddlePaddle model,
```
import paddle
model = paddle.jit.load('model/infer')

shape_dict = {'image': [1, 3, 224, 224]}
mod, params = relay.frontend.from_paddle(model, shape_dict=shape_dict)
```

Error may happend if there are some operators is not supported by this frontend, and the details will print out.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

Since this RFC is to add a new frontend, PaddlePaddle model will be converted to TVM Relay IR, so all the other features will not be effected.

For this proposed RFC, the whole process of PaddlePaddle frontend importing can be divided into 2 steps:
- 1. Reading PaddlePaddle Model: The frontend supports PaddlePaddle's inference model format which is exported as graph based model by PaddlePaddle's `Dynamic to Static` mechanism, The model contains 2 files that store the model structure and parameters separately, We use `paddle.jit.load` to load the model files(For the compatibility of previous version of PaddlePaddle, `paddle.static.load_inference_model` also supported); 
- 2. Operator Conversion: After the exported inference model is loaded, we will extract its parameters and convert operators one by one. Since all the operators can be iterated out by toposort, there's no need to worry about the order of converting operators. 

# Drawbacks
[drawbacks]: #drawbacks

This may bring more time cost of unit test running.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

The frontend of PaddlePaddle is similar with ONNX or TensorFlow. We support model loaded by `paddle.jit.load` and `paddle.static.load_inference_model`. Also we haved considered only support `paddle.jit.load` since this API is recommended after PaddlePaddle 2.0, but there are lots of users still use `paddle.static.load_inference_model`.
Currently, we have to convert PaddlePaddle model to ONNX format to make it work with TVM, but only part of models are supported due to the lack of ONNX operators and the operator difference. With a new PaddlePaddle frontend, we can support more operators and provide a better experience for TVM and PaddlePaddle's users.

# Prior art
[prior-art]: #prior-art

It's the first time we add a PaddlePaddle frontend to a ML compilers.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

We will add new unit test which will rely on PaddlePaddle framework, also the test will bring more cost time, if there's any problem, please let me know.

# Future possibilities
[future-possibilities]: #future-possibilities

For this RFC, we have make a established plan,

- About 200 operators will be supported in this quarter, such as deformable_conv/multiclass_nms
- Control flow operators will be supported in this year,  mainly about while_loop/if/
- Quantized model will be supported in this year, include quantized model from `PaddleDetection`/`PaddleClas`/`PaddleSeg`
