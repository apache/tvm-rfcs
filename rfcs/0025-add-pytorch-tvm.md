- Feature Name: PyTorchTVM
- Start Date: 2021-08-24
- RFC PR: [apache/tvm-rfcs#0025](https://github.com/apache/tvm-rfcs/pull/25)
- GitHub Issue: TODO

# Summary
[summary]: #summary

This RFC add a `PyTorchTVM` module to support: offload subgraphs of TorchScript to TVM, and then embed those TVM-accelerated subgraphs back to TorchScript for runtime execution.

To help boost model performance and enhance TVM adoption for machine learning practitioners who often use PyTorch, `PyTorchTVM` is proposed for seamless integration for TVM in TorchScript, and its workflow is demonstrated as follows:
1. Convert a TorchScript module (or submodule) to TVM graph (Relay)
2. Optimize and compile the TVM graph with auto-tuning
3. Export and embed the optimized TVM module as a PyTorch custom op
4. The embedded custom op works smoothly with TorchScript (both `torch.jit.trace` and `torch.jit.script`), without tangible difference with normal PyTorch models, i.e. it can be saved to disk, loaded back and served online with no change in the overall workflow



# Motivation
[motivation]: #motivation

PyTorch enjoys increasing popularity among machine learning research community as well as in industrial production environment. However, it is still a missing piece as a generic, comprehensive and effective toolchain to accelerate real-world models and workloads in PyTorch, which raises primary concern in performance-critical production environments.

Below are the two classic acceleration workflows as the status quo:
- PyTorch -> ONNX -> TensorRT/TVM
- PyTorch -> TorchScript -> TensorRT/TVM

However, both workflows introduce one level of indirection, which means flaws of either levels are inherited in the pipeline. For example:
- ONNX offers no support for models with dynamic control flow, so the first workflow is unable to support models with dynamic control flow
- The coverage of TensorRT is often limited to a range of standard neural networks, so both of the workflows, if offloaded to TensorRT, are hard to be effective on real-world models.

Furthermore, both of the existing workflows don't provide any benefit of an interface that is practical enough for researchers to widely adopt and reuse. For example, it requires deep knowledge of TVM runtime modules to load the exported binary artifacts back to python and use it together with PyTorch.

So we hope to use TVM to accelerate PyTorch model inference.


# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation


As an example, an end-to-end ResNet-based image classifier contains 3 major parts in its pipeline:
1. A data loader that reads and decodes images (in png/jpeg/...）to PyTorch Tensors
2. A sequence of image transformation that normalizes the input images, including resize, crop, type conversions, etc
3. Finally, a ResNet that maps a batch of input images to their class labels accordingly
Below is a snippet that illustrates the workflow of this pipeline:
``` python
class Predictor(nn.Module):

    def __init__(self, tvm_module=None):
        super().__init__()
        self.resnet18 = resnet18(pretrained=True, progress=False).eval()
        self.transforms = nn.Sequential(
            T.Resize([256, ]),  # We use single int value inside a list due to torchscript type restrictions
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.half),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )

    def forward(self, image_path: List[str]) -> torch.Tensor:
        with torch.no_grad():
            images: List[torch.Tensor] = []
            for path in image_path:
                img = read_image(path)
                images.append(img)
            x = torch.stack(images).cuda().half()
            x = self.transforms(x)
            y_pred = self.resnet18(x)
            return y_pred.argmax(dim=1)
```

With PyTorchTVM, we are able to compile the ResNet with TVM and embed it back to PyTorch seamlessly with a few lines of code:

``` python
from tvm.contrib.pt_op import PyTorchTVMModule, compile

print("compile...")
option = {
    "input_infos": [
        ("x", (1, 3, 224, 224)),
    ],
    "default_dtype": "float16",
    "export_dir": "pytorch_compiled",
    "num_outputs": 1,
    "tuning_n_trials": 0,  # set zero to skip tuning
    "tuning_log_file": "tuning.log",
}
x = torch.randn(1, 3, 224, 224).cuda().half()
resnet_jit = torch.jit.trace(model.resnet18, x)
resnet_tvm = compile(resnet_jit, option)
```

The TVM-accelerated `resnet_tvm` module can be used directly in PyTorch, or integrated into TorchScript with `torch.jit.script` along with all other PyTorch-native operations.

``` python
resnet_tvm = torch.jit.script(resnet_tvm)
print(resnet_tvm.graph)


class PredictorTVM(nn.Module):

    def __init__(self):
        super().__init__()
        self.resnet18 = resnet_tvm
        self.transforms = nn.Sequential(
            T.Resize([256, ]),  # We use single int value inside a list due to torchscript type restrictions
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.half),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )

    def forward(self, image_path: List[str]) -> torch.Tensor:
        with torch.no_grad():
            images: List[torch.Tensor] = []
            for path in image_path:
                img = read_image(path)
                images.append(img)
            x = torch.stack(images).cuda().half()
            x = self.transforms(x)
            # y_pred = self.resnet18(x)
            y_pred = self.resnet18([x])[0]
            return y_pred.argmax(dim=1)


print("run tvm...")
model_tvm = PredictorTVM().cuda().half()
for i in range(20):
    t = time.time()
    model_tvm([image_path])
    torch.cuda.synchronize()
    print(time.time() - t)

torch.jit.script(model_tvm).save("model_tvm.pt")
```

Note that the script above provides a seamless serializable solution that allows TVM acceleration to be embedded into TorchScript and thus served in online production without extra effort.


# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

We have opened an initial PR: https://github.com/apache/tvm/pull/8777

The essential cpp code is as follows:

``` c++
// This is just a wrapper class of tvm graph runtime module
class TvmGraphModulePack {
 ...
 private:
  tvm::runtime::Module module_;
  ...
};

// This is the base of our custom classes, 
// we define some common helper function in this class
class BaseTvmClass : public torch::jit::CustomClassHolder {
  ...
  // Converts a list of input tensor shapes to a std::string
  static std::string TvmShapeRepr(const c10::List<c10::List<int64_t>>& shapes);
  // Gets shape list from input tensors
  static c10::List<c10::List<int64_t>> GetShapes(const c10::List<at::Tensor>& inputs);
  ...
};

// The custom class that embeds TVM Graph runtime Module in torchscript. 
// There is also a TvmVMRuntimeClass that supports VM Runtime Module which is not shown here
class TvmGraphRuntimeClass : public BaseTvmClass {
 public:
  TvmGraphRuntimeClass(const int64_t num_inputs, const int64_t num_outputs,
                       const std::string& device)
      : BaseTvmClass(num_inputs, num_outputs, device) {}
  
  // Load a TVM Graph Runtime Module into tvm_modules_.
  void LoadTvmModule(const c10::List<c10::List<int64_t>>& shapes, const std::string& lib_path,
                     const std::string& graph_path, const std::string& params_path) {
    ...
    auto shape_repr = TvmShapeRepr(GetShapes(inputs));
    const auto it =
        tvm_modules_.emplace(shape_repr, TvmGraphModulePack(path, device_type_, device_id_)).first;
    ...
  }

  virtual c10::List<at::Tensor> forward(const c10::List<at::Tensor>& inputs) override {
    CHECK_EQ(inputs.size(), num_inputs_);
    auto shape_repr = TvmShapeRepr(GetShapes(inputs));
    auto iter = tvm_modules_.find(shape_repr);
    ...
  }
  
 private:
  // key of this map is the shape repr string of inputs
  std::map<std::string, TvmGraphModulePack> tvm_modules_;
};


// registry
static auto __tvm_class_graph_runtime_registry =
    torch::jit::class_<TvmGraphRuntimeClass>("tvm_class", "TvmGraphModule")
        .def(torch::init<const int64_t, const int64_t, const std::string&>())
        .def("load_tvm_module", &TvmGraphRuntimeClass::LoadTvmModule)
        .def("forward", &TvmGraphRuntimeClass::forward)
        .def("to", &TvmGraphRuntimeClass::to)
        .def_pickle(
            ...
            });
```

And we wrap the custom class in Python:

``` python
class GraphModule(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, device=None):
        ...
        self.engine = torch.classes.tvm_class.TvmGraphModule(num_inputs, num_outputs, self.device)
        
    def init(self, input_shapes, lib_path, graph_path, params_path):
        self.engine.load_tvm_module(input_shapes, lib_path, graph_path, params_path)

    def forward(self, inputs: List[torch.Tensor]):
        return self.engine.forward(inputs)
        
    ...
```

# Drawbacks
[drawbacks]: #drawbacks


There are some limitations now:

1. Dynamic shape support

    Currently we support multiple input_shapes with a bucket policy, which is hacky. A more formal implementation will be in our future work.

2. Zero overhead output

    Now we only have `set_input_zero_copy`, but our `set_output` has a memcpy.

3. Performance of TVM without tuning

    Without autotuning, the performance of TVM is most likely worse compared with native pytorch. To give users immediate feedback, maybe we can make tvm use cudnn/cublas/cutlass as a default implementation.


# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

PyTorch is rapidly being adopted because of its user-fiendly API and dynamic capability. With `PytorchTVM`, users can accelerate PyTorch model inference without compromising the full functionality of PyTorch.

# Prior art
[prior-art]: #prior-art

Our implementation is inspired by this RFC that embeds TVM into TensorFlow: https://discuss.tvm.apache.org/t/rfc-add-tensorflow-custom-op-to-embed-tvm-runtime-in-tensorflow-graph-and-session/4601


# Unresolved questions
[unresolved-questions]: #unresolved-questions

See Section Drawbacks


# Future possibilities
[future-possibilities]: #future-possibilities

* Resolve the 3 drawbacks listed above.
* Cover more dynamic networks in TVM, and finally convert and accelerate most PyTorch model.
