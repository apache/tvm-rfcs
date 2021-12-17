- Feature Name: (fill me in with a unique identifier, `my_awesome_feature`)
- Start Date: (fill me in with today's date, YYYY-MM-DD)
- RFC PR: [apache/tvm-rfcs#0000](https://github.com/apache/tvm-rfcs/pull/0000)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)
- GitHub pre-RFC PR: [apache/tvm-PR-9730](https://github.com/apache/tvm/pull/9730)
- GitHub pre-RFC discussion: [BYOC-Marvell](https://discuss.tvm.apache.org/t/pre-rfc-byoc-marvell-ml-ai-accelerator-integration/11691)

# Summary
[summary]: #summary

Integrate Marvell’s ML/AI accelerator with TVM BYOC framework in order to bring the TVM ecosystem to Marvell customers.

# Motivation
[motivation]: #motivation

Marvell MLIP is an ML/AI inference accelerator and is embedded on our ARM Neoverse N2-based OCTEON 10 processor.
  We are building an easy-to-use, open, software suite for our customers by integrating and utilizing TVM so that
  we can bring TVM capability and experience to our customers.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Based on what Marvell ML/AI inference accelerator does the best, a given pre-trained network model
will be applied to a TVM-Mrvl-BYOC AOT compilation and code-gen flow as illustrated in steps below.

STEP (1) Run TVM-Mrvl-BYOC AOT ML Frontend Compilation and Mrvl-BYOC code-gen. The steps involved in this are:

* Load pre-trained network into TVM IR graph

* Do Marvell-specific layout conversions to transform IR graph in order to meet requirements of the accelerator

* Do Marvell-specific composite-merging/fusing to transform IR graph in order to utilize available HW capability
  in the accelerator

* Do additional Marvell-specific transform pass(es) to further optimize IR graph

* Partition IR graph into one or more for-accelerator Mrvl subgraphs and/or one or more for-TVM-target non-Mrvl
  (e.g., ARMv9) subgraphs
    * These subgraphs cover the whole pre-trained network
    * For-accelerator Mrvl subgraph here means & contains connected, composite-fused Call nodes (let's call this sub-graph A)
      as in the given IR graph. A composite-merged Call node can be, for instance, fused from this sequence of IR call nodes:
      conv2d + add + batch_norm + tuple.getitem(0) + relu
    * For the first Marvell-BYOC revision, at most one for-accelerator Mrvl subgraph and at most one for-TVM-target
      non-Mrvl subgraph (let's call this sub-graph B) can be identified; plus, the for-accelerator Mrvl subgraph can
      only use input tensor(s) of given pre-trained network as its subgraph’s input tensors

* Do code-gen step for each for-accelerator Mrvl subgraph:
    * Marvell-BYOC-specific attributes are introduced for each composite-merged/fused Call node so that a Nodes-JSON
      file and a Constants-JSON file are produced for the Mrvl subgraph

STEP (2) Run Mrvl-ML/AI Backend Compiler to generate model binary for each Mrvl subgraph

* The Mrvl-ML/AI backend compiler will be distributed as an executable in the OCTEON SDK; and it can be used to read
  in Nodes-JSON and Constants-JSON files of each Mrvl subgraph as input meta-data in order to generate final instructions,
  in model binary file

* Note: Mrvl-ML/AI backend compiler, which does accelerator-specific optimization and code generation, is not included
  to upstream

STEP (3a) or (3b) Run inference on the software Simulator or on the Mrvl ML/AI HW accelerator for the Mrvl subgraph

* The Mrvl Software Simulator of the Mrvl ML/AI HW accelerator will be distributed as an executable in a Mrvl-ML/AI tar
  ball; and it can be used to read in input file(s) and the model binary to run inference for the Mrvl subgraph

* Note: Mrvl ML/AI accelerator can run inference in either float16 mode or int8 quantization mode. For this RFC, we will
  focus only on float16 inference run

STEP (4) Use TVM-llvm Compiler & Runtime to run inference

* Perform integration steps between sub-graph(s) in order to run inference for the given pre-trained network -
  note: runtime binary for each for-TVM-target non-Mrvl subgraph can be generated, for instance, using the regular TVM
  LLVM build

* For the first Marvell-BYOC revision, at most one integration step from a for-accelerator Mrvl subgraph to
  a TVM-target non-Mrvl subgraph is implemented

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

## Illustration using a MNIST model

Let's use a Keras MNIST fashion model below as an example (partial & pseudo code for illustration).
```
  Get Input-Fashion-Image-Tensor-nchw - input_shape: [1, 1, 28, 28]

  keras.Input(shape=input_shape)
  keras.layers.Conv2D(64, kernel_size=(2, 2), activation="relu")
  keras.layers.MaxPooling2D(pool_size=(2, 2))
  keras.layers.Conv2D(32, kernel_size=(2, 2), activation="relu")
  keras.layers.MaxPooling2D(pool_size=(2, 2))
  keras.layers.Dropout(0.3)
  keras.layers.Reshape()
  keras.layers.Dense(256, activation="relu")
  keras.layers.Dense(10)

  Generate Output-Tensor - output_shape: [1, 10]

  top_label_id = numpy.argmax(Output-Tensor)
  # fashion label map
  fashion_label_dictionary = {
      0: "T-shirt/top",
      1: "Trouser",
      2: "Pullover",
      3: "Dress",
      4: "Coat",
      5: "Sandal",
      6: "Shirt",
      7: "Sneaker",
      8: "Bag",
      9: "Ankle boot",
  }
  print(f"Fashion item identified as: {fashion_label_dictionary[top_label_id]}")
```

We can train the above MNIST fashion model using the following train_images dataset and save
  the pre-trained model in ONNX (say, mnist_fashion.onnx). Then, we can run BYOC Marvell flow by giving any
  image of the orig_test_images[i] dataset to get its inference fashion label and item name in top_label_id and
  fashion_label_dictionary[top_label_id], respectively. In addition, we can also use the corresponding
  golden label, golden_output_labels[i], to validate the inference result.

```
(train_images, train_labels), (
    orig_test_images,
    golden_output_labels,
) = keras.datasets.fashion_mnist.load_data()
```

As illustrated in the tests/python/contrib/test_mrvl/test_mrvl_codegen.py and infrastructure.py files as well as
  in pseudo code below, we can call onnx.load() and relay.frontend.from_onnx() to generate TVM mod and params. Then,
  they are used as function arguments to call the aot_build_and_json_code() API in order to generate Nodes-JSON file
  (nodes_json_filename) and Constants-JSON file (consts_json_filename).

* Notes: please refer to the python/tvm/relay/op/contrib/mrvl.py file for more details.

* In the mrvl.py file: the partition_for_mrvl() function is the main entry point for the BYOC Marvell flow.

* We use relay.build(mod_mrvl_subgraph).get_params() and relay.build(mod_mrvl_subgraph).get_external_graph_json()
    to trigger Marvell-specific GetExternalJSON() and JSON load/save functions (as defined in the
    src/relay/backend/contrib/mrvl/graph_executor_codegen_mrvl.cc file) in order to generate
    Marvell-specific byoc_const_params and byoc_external_graph_json objects.

* In the mrvl.py file: the dump_json_meta_data_files() function takes in Marvell-specific byoc_external_graph_json
    and byoc_const_params objects to generate and return two Marvell-specific Nodes-JSON file and Constants-JSON file,
    respectively.

```
    # load pre-trained model
    mnist_fashion_onnx_model = onnx.load("mnist_fashion.onnx")
    mod, params = relay.frontend.from_onnx(
        mnist_fashion_onnx_model, dtype="float32", freeze_params=False
    )


    # from test_mrvl_codegen.py: to generate sub graphs and JSON files
    (
        nodes_json_filename,
        consts_json_filename,
        mod_mrvl_subgraph,
        mod_non_mrvl_subgraph,
        mrvl_layers_in_mrvl_subgraph,
        mrvl_layers_in_non_mrvl_subgraph,
    ) = aot_build_and_json_codegen(
        model_name="mnist_fashion",
        working_dir="mnist",
        mod,
        params,
    )


    # from infrastructure.py: pedueo code defined by the above aot_build_and_json_codegen() function
    (
        mod_mrvl_subgraph,
        mod_non_mrvl_subgraph,
        orig_params,
        opt_level,
        disabled_pass,
        orig_mod,
        mrvl_layers_in_mrvl_subgraph,
    ) = mrvl.partition_for_mrvl(
        mod,
        params=params,
        tvm_custom_dict={},
        gen_non_mrvl_subgraph=gen_non_mrvl_subgraph,
        flow_pass=1,
    )

    build_target, device_id = "llvm", 0
    mod_name = relay.backend.utils.mangle_module_name("")
    byoc_executor = relay.build(mod_mrvl_subgraph, target=build_target, mod_name=mod_name)
    byoc_const_params = byoc_executor.get_params()
    byoc_external_graph_json = byoc_executor.get_external_graph_json()

    nodes_json_filename, consts_json_filename = mrvl.dump_json_meta_data_files(
        byoc_external_graph_json,
        byoc_const_params,
        filename_prefix=f"{working_dir}{model_name}-tvm-mrvl-byoc-ir",
    )
```

The mod_mrvl_subgraph object and the mod_non_mrvl_subgraph object returned from the aot_build_and_json_code()
  call are IR graphs of one for-accelerator Mrvl subgraph and one TVM-target non-Mrvl subgraph, respectively.

Different strategy can be used to cut the MNIST model into different sets of at most one Mrvl subgraph and at
  most one non-Mrvl subgraph. Below we will illustrate one such alternative (i.e., the default strategy) so
  that, for this specific sample MNIST model, the entire network model is turned into one Mrvl subgraph and
  no non-Mrvl subgraph.

* Below is the original IR graph - i.e., right after from_onnx() call

```
    #[version = "0.0.5"]
    def @main(%permute_input: Tensor[(1, 1, 28, 28), float32]) -> Tensor[(1, 10), float32] {
      %0 = nn.conv2d(%permute_input, meta[relay.Constant][0] /* ty=Tensor[(64, 1, 2, 2), float32] */,
          padding=[0, 0, 1, 1], channels=64, kernel_size=[2, 2], /* en_id=418 */) /* ty=Tensor[(1, 64, 28, 28), float32] */;
      %1 = nn.bias_add(%0, meta[relay.Constant][1] /* ty=Tensor[(64), float32] */,
          /* en_id=419 */) /* ty=Tensor[(1, 64, 28, 28), float32] */;
      %2 = nn.relu(%1, /* en_id=420 */) /* ty=Tensor[(1, 64, 28, 28), float32] */;
      %3 = nn.max_pool2d(%2, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0],
          /* en_id=449 */) /* ty=Tensor[(1, 64, 14, 14), float32] */;
      %4 = nn.conv2d(%3, meta[relay.Constant][2] /* ty=Tensor[(32, 64, 2, 2), float32] */,
          padding=[0, 0, 1, 1], channels=32, kernel_size=[2, 2], /* en_id=472 */) /* ty=Tensor[(1, 32, 14, 14), float32] */;
      %5 = nn.bias_add(%4, meta[relay.Constant][3] /* ty=Tensor[(32), float32] */,
          /* en_id=473 */) /* ty=Tensor[(1, 32, 14, 14), float32] */;
      %6 = nn.relu(%5, /* en_id=474 */) /* ty=Tensor[(1, 32, 14, 14), float32] */;
      %7 = nn.max_pool2d(%6, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0],
          /* en_id=515 */) /* ty=Tensor[(1, 32, 7, 7), float32] */;
      %8 = transpose(%7, axes=[0, 2, 3, 1], /* en_id=516 */) /* ty=Tensor[(1, 7, 7, 32), float32] */;
      %9 = nn.batch_flatten(%8, /* en_id=538 */) /* ty=Tensor[(1, 1568), float32] */;
      %10 = transpose(meta[relay.Constant][4] /* ty=Tensor[(1568, 256), float32] */, axes=[1, 0],
          /* en_id=599 */) /* ty=Tensor[(256, 1568), float32] */;
      %11 = nn.dense(%9, %10, units=None, out_dtype="float32", /* en_id=600 */) /* ty=Tensor[(1, 256), float32] */;
      %12 = add(%11, meta[relay.Constant][5] /* ty=Tensor[(256), float32] */,
          /* en_id=601 */) /* ty=Tensor[(1, 256), float32] */;
      %13 = nn.relu(%12, /* en_id=602 */) /* ty=Tensor[(1, 256), float32] */;
      %14 = transpose(meta[relay.Constant][6] /* ty=Tensor[(256, 10), float32] */, axes=[1, 0],
          /* en_id=675 */) /* ty=Tensor[(10, 256), float32] */;
      %15 = nn.dense(%13, %14, units=None, out_dtype="float32", /* en_id=676 */) /* ty=Tensor[(1, 10), float32] */;
      add(%15, meta[relay.Constant][7] /* ty=Tensor[(10), float32] */, /* en_id=677 */) /* ty=Tensor[(1, 10), float32] */
}

```

* We can get to the following one Mrvl subgraph by applying the default strategy.
    * in the mrvl.py file: the compute_two_subgraphs() function of the class MrvlIRGraphUtils is used
      to create mod_mrvl_subgraph and mod_non_mrvl_subgraph for

```
    def @main(%permute_input: Tensor[(1, 1, 28, 28), float32]) -> Tensor[(1, 10), float32] {
      %0 = @tvmgen_mrvl_main_0(%permute_input, /* en_id=4136 */) /* ty=Tensor[(1, 28, 28, 1), float32] */;
      %1 = @tvmgen_mrvl_main_1(%0, /* en_id=4137 */) /* ty=Tensor[(1, 28, 28, 64), float32] */;
      %2 = @tvmgen_mrvl_main_2(%1, /* en_id=4138 */) /* ty=Tensor[(1, 14, 14, 64), float32] */;
      %3 = @tvmgen_mrvl_main_3(%2, /* en_id=4139 */) /* ty=Tensor[(1, 14, 14, 32), float32] */;
      %4 = @tvmgen_mrvl_main_4(%3, /* en_id=4140 */) /* ty=Tensor[(1, 7, 7, 32), float32] */;
      %5 = @tvmgen_mrvl_main_5(%4, /* en_id=4141 */) /* ty=Tensor[(1, 1568), float32] */;
      %6 = @tvmgen_mrvl_main_6(%5, /* en_id=4142 */) /* ty=Tensor[(1, 256), float32] */;
      @tvmgen_mrvl_main_7(%6, /* en_id=4143 */) /* ty=Tensor[(1, 10), float32] */
    }
```

* In the above Mrvl subgraph, it is formed by "not-yet optimized Marvell (backend) layers". For example,
    tvmgen_mrvl_main_0 to tvmgen_mrvl_main_7 are composited/fused Marvell layers.
    * In the mrvl.mrvl_pattern_table() function, fusing patterns have been defined in order to composite
      original IR nodes into Marvell backend layers.
    * For example, the following 3 IR call nodes (nn.conv2d + nn.bias_add + nn.relu) in the original IR graph
      are composited into one Marvell layer: tvmgen_mrvl_main_1, conceptually speaking.
```
      # from original IR graphs
      %4 = nn.conv2d(%3, meta[relay.Constant][2] /* ty=Tensor[(32, 64, 2, 2), float32] */,
          padding=[0, 0, 1, 1], channels=32, kernel_size=[2, 2], /* en_id=472 */) /* ty=Tensor[(1, 32, 14, 14), float32] */;
      %5 = nn.bias_add(%4, meta[relay.Constant][3] /* ty=Tensor[(32), float32] */,
          /* en_id=473 */) /* ty=Tensor[(1, 32, 14, 14), float32] */;
      %6 = nn.relu(%5, /* en_id=474 */) /* ty=Tensor[(1, 32, 14, 14), float32] */;


      # from Mrvl subgraph
      %3 = @tvmgen_mrvl_main_3(%2, /* en_id=4139 */) /* ty=Tensor[(1, 14, 14, 32), float32] */;
      def @tvmgen_mrvl_main_3(%mrvl_3_i0: Tensor[(1, 14, 14, 64), float32], Inline=1, Compiler="mrvl",
          global_symbol="tvmgen_mrvl_main_3", Primitive=1) -> Tensor[(1, 14, 14, 32), float32] {

        %13 = fn (%FunctionVar_0_0: Tensor[(1, 14, 14, 64), float32], PartitionedFromPattern="nn.conv2d_add_nn.relu_",
            Composite="mrvl.conv2d_nhwc2nhwc") -> Tensor[(1, 14, 14, 32), float32] {
          %11 = nn.conv2d(%FunctionVar_0_0, meta[relay.Constant][2] /* ty=Tensor[(32, 2, 2, 64), float32] */,
              padding=[0, 0, 1, 1], channels=32, kernel_size=[2, 2], data_layout="NHWC", kernel_layout="OHWI",
              out_layout="NHWC", /* en_id=781 */) /* ty=Tensor[(1, 14, 14, 32), float32] */;
          %12 = add(%11, meta[relay.Constant][3] /* ty=Tensor[(1, 1, 1, 32), float32] */,
              /* en_id=789 */) /* ty=Tensor[(1, 14, 14, 32), float32] */;
          nn.relu(%12, /* en_id=793 */) /* ty=Tensor[(1, 14, 14, 32), float32] */
        };

        %13(%mrvl_3_i0, /* en_id=3343 */) /* ty=Tensor[(1, 14, 14, 32), float32] */
      }
```

* Because Marvell backend layer uses NHWC format (for instance, for Conv2D, Pool2D, and Sum2D),
    the relay.transform.ConvertLayout() pass is applied in the mrvl.py file. As a result, NHWC format is used
    for Marvell layer: tvmgen_mrvl_main_1 to tvmgen_mrvl_main_4. In addition, the first tvmgen_mrvl_main_0 layer
    is corresponding to a layout_transform() operation, which takes the original input tensor in src_layout="NCHW"
    and convert the input to a dst_layout="NHWC" tensor.

```
      relay.transform.ConvertLayout(
          {"nn.conv2d": ["NHWC", "OHWI"], "nn.max_pool2d": ["NHWC"]}
      ),

      %0 = @tvmgen_mrvl_main_0(%permute_input, /* en_id=4136 */) /* ty=Tensor[(1, 28, 28, 1), float32] */;
      %1 = @tvmgen_mrvl_main_1(%0, /* en_id=4137 */) /* ty=Tensor[(1, 28, 28, 64), float32] */;
      %2 = @tvmgen_mrvl_main_2(%1, /* en_id=4138 */) /* ty=Tensor[(1, 14, 14, 64), float32] */;
      %3 = @tvmgen_mrvl_main_3(%2, /* en_id=4139 */) /* ty=Tensor[(1, 14, 14, 32), float32] */;
      %4 = @tvmgen_mrvl_main_4(%3, /* en_id=4140 */) /* ty=Tensor[(1, 7, 7, 32), float32] */;

      def @tvmgen_mrvl_main_0(%mrvl_0_i0: Tensor[(1, 1, 28, 28), float32], Inline=1, Compiler="mrvl",
          global_symbol="tvmgen_mrvl_main_0", Primitive=1) -> Tensor[(1, 28, 28, 1), float32] {
        layout_transform(%mrvl_0_i0, src_layout="NCHW", dst_layout="NHWC",
            /* en_id=3334 */) /* ty=Tensor[(1, 28, 28, 1), float32] */
      }
```

* Currently, in order for the following Marvell classes/functions to identify a Mrvl subgraphs and a non-Mrvl
  subgraph from the layout-converted, composited/fused IR graph, we are utilizing the unique en_id attribute
  stored for the Class CallNode and the class Tuple (include/tvm/relay/expr.h).
    * in mrvl.py: class MrvlIRGraphUtils.RestOfMrvlLayers(ExprMutator) is used to convert the non-Mrvl subgraph,
      which can have composited Marvell layer(s) back to their original IR nodes (e.g., to use original tensor
      layout and with no compositions)
    * in mrvl.py: class MrvlIRGraphUtils.RestMrvlLayersGetInputs(ExprVisitor) is used to reconstruct the input
      tensor for the non-Mrvl subgraph so that it become a IR graph, which is recognized by the TVM LLVM build.
    * in mrvl.py: the revert_mrvl_mod_to_orig() function is defined to convert the initial non-Mrvl subgraph back
      to a IR subgraph using original layouts with no Marvell-specific compositions (e.g., similar to what was
      given by the frontend)

```
def revert_mrvl_mod_to_orig(mod_mrvl_subgraph, mrvl_layers_in_mrvl_subgraph, debug=False):
    """

    def run_opt_pass(mod, passes):
        passes = passes if isinstance(passes, list) else [passes]
        seq = tvm.transform.Sequential(passes)
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
        return mod

    mod_new = tvm.IRModule(mod_mrvl.functions, mod_mrvl.type_definitions)
    mod_new["main"] = MrvlSubgraphToRevert(mrvl_layers_in_mrvl_subgraph, mod_mrvl).visit(mod_mrvl["main"])
    mod_new = relay.transform.RemoveUnusedFunctions()(mod_new)
    mod_new = relay.transform.InferType()(mod_new)
    mod_new = run_opt_pass(mod_new, relay.transform.DefuseOps())
    mod_new = run_opt_pass(mod_new, relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "OIHW"], "nn.max_pool2d": ["NCHW"]}))
    mod_new = run_opt_pass(mod_new, relay.transform.SimplifyExpr())
    mod_new = run_opt_pass(mod_new, relay.transform._ffi_api.DropNoopTranspose())
    mod_new = run_opt_pass(mod_new, relay.transform.InferType())
    return mod_new
```

* Marvell-specific graph executor codegen, We have defined call backs and extension functions in the following files:
    * Some common classes have been moved from the original src/relay/backend/graph_executor_codegen.cc file to the
      new src/relay/backend/graph_executor_codegen.h file so that they can be shared by Marvell-specific functions
      and derived classes defined in the new src/relay/backend/contrib/mrvl/graph_executor_codegen.cc file

    * new definitions are listed below:
```
    /////////////
    // in the new src/relay/backend/graph_executor_codegen.h file
    /*! \brief Node types */
    enum GraphNodeType {
      kGraphNop,
      kGraphInputNode,
      kGraphOpNode,
      kGraphInputNodeExt,
      kGraphOpNodeExt,
    };

    
    class ExternalJsonWriterCB {
     public:
      template <class T>
      void RegisterCB(T* const object,
                      void (T::*const mf)(dmlc::JSONWriter*, Array<tvm::runtime::Module>,
                                          std::vector<GraphObjectPtr>, std::vector<GraphNodeRef>)) {
        using namespace std::placeholders;
        callback_ = std::bind(mf, object, _1, _2, _3, _4);
        hasCallback_ = true;
      }
      void RegisterCB(void (*const fun)(dmlc::JSONWriter*, Array<tvm::runtime::Module>,
                                        std::vector<GraphObjectPtr>, std::vector<GraphNodeRef>)) {
        callback_ = fun;
        hasCallback_ = true;
      }
      void Exe(dmlc::JSONWriter* external_writer, Array<tvm::runtime::Module> mod,
               std::vector<GraphObjectPtr> nodes, std::vector<GraphNodeRef> heads) {
        ICHECK(hasCallback_) << "ERROR: no registered callback";
        callback_(external_writer, mod, nodes, heads);
      }
      inline bool HasCallback() { return hasCallback_; }

     private:
      std::function<void(dmlc::JSONWriter*, Array<tvm::runtime::Module>, std::vector<GraphObjectPtr>,
                         std::vector<GraphNodeRef>)>
          callback_;
      bool hasCallback_{false};
    };

    /////////////
    // in the new src/relay/backend/graph_executor_codegen.cc file
    class GraphExecutorCodegen : public backend::MemoizedExprTranslator<std::vector<GraphNodeRef>> {
     public:
      GraphExecutorCodegen(runtime::Module* mod, const TargetMap& targets)
          : mod_(mod), targets_(targets) {
        // we need the following variable to be a static member of the class so we can access
        //   its setting in the following static GetExternalJsonWriter() function; but this static
        //   member can actually be used as a local Callback setting for "per" GraphExecutorCodegen
        //   instantiation during each TVM build-codegen flow
        external_json_writer_ = std::make_shared<ExternalJsonWriterCB>();
        ICHECK(external_json_writer_);
      }
      static ExternalJsonWriterCB* GetExternalJsonWriter() { return external_json_writer_.get(); }
      ....
      LoweredOutput Codegen(IRModule mod, relay::Function func, String mod_name) {
        ....

        // if it has been registered for this GraphExecutorCodegen object, call the external JSON writer
        if (external_json_writer_->HasCallback()) {
          std::ostringstream external_os;
          dmlc::JSONWriter external_writer(&external_os);
          external_json_writer_->Exe(&external_writer, ret.external_mods, nodes_, heads_);
          ret.external_graph_json = external_os.str();
        }

        return ret;
      }
    };

    extern "C" ExternalJsonWriterCB* GetExternalJsonWriter() {
      return GraphExecutorCodegen::GetExternalJsonWriter();
    }

    /////////////
    // in the new src/relay/backend/contrib/mrvl/graph_executor_codegen.cc file
    // Marvell-specific extentions
    class GraphInputNodeMrvlExt : public GraphInputNode {
        ...
        GraphNodeType Type() const override { return kGraphInputNodeExt; }
        void Save(dmlc::JSONWriter* writer) const override { /* extensions */ }
    }

    class GraphOpNodeMrvlExt : public GraphOpNode {
        ...
        GraphNodeType Type() const override { return kGraphOpNodeExt; }
        void Load(dmlc::JSONReader* reader) override;
        void LoadAttrs(dmlc::JSONReader* reader);
        std::pair<std::string, GraphAttrs> GetLoadedGraphAttrs();
    }

    class MrvlExtJson {
     public:
      MrvlExtJson() {
        ICHECK(!GetExternalJsonWriter()->HasCallback()) << "ERROR: has registered callback";
        GetExternalJsonWriter()->RegisterCB(this, &MrvlExtJson::GetExternalJSON);
      }
      virtual ~MrvlExtJson() {}
      void GetExternalJSON(dmlc::JSONWriter* writer, Array<tvm::runtime::Module> external_mods,
                           std::vector<GraphObjectPtr> nodes, std::vector<GraphNodeRef> heads);
      void LoadExternalJsonAttrs(std::unordered_map<std::string, GraphAttrs>* external_attrs_map,
                                 const Array<tvm::runtime::Module>& external_mods);
    };
```

* the need to link between pre-trained model and final Marvell backend layer - for instance, through tvm_custom
    * We did not include prototype code in PR-9730 but intend to provide our sample changes in another RFC and PR.


# Drawbacks
[drawbacks]: #drawbacks

* We haven't identified any major *not* do items. Several other designs are by choices - that is we understand that
  there are benefits for doing or benefits for not-doing.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

* We follow the TVM BYOC framework to enable BYOC Marvell flow without impacting any TVM core features.


# Unresolved questions
[unresolved-questions]: #unresolved-questions

* We are following the existing TVM BYOC framework and example files.
    * for example: to do IR compositions, to define own IR passes, to mix implementations in Python/C++, and etc.

* We have extended graph_executor_codegen.cc and JSON loader/saver in order to read and write out Marvell specific
  attributes

* Currently, we haven't spend enough time to under how tvm/rust/cargo requirements and steps. Therefore, we are
  bypassing the tvm/Jenkinsfile's tests/scripts/task_rust.sh step. We will need help to re-enable the step.

* We like to duplicate the Jenkins environment in order to run tvm/Jenkinsfile as is, but, we ran into many issues.
  Currently, we have a tvm-like Jenksinsfile environment to only run a subset of test suites using a modified
  Jenkinsfile.

* We have identified a need to allow a call-back function to be registered when generating Mrvl-BYOC-specific
  Nodes-JSON file. We are trying to follow TVM Python/CPP-CB style as much as possible. But, since our callback
  function tvm/src/relay/backend/contrib/mrvl/graph_executor_codegen_mrvl.cc::GetExternalJSON() function is using
  non-simple argument types, we need help from TVM community to provide suggestions/guidelines in order to make
  new CB code better to meet TVM community requirements here.

* For one Mrvl-BYOC relay transformation pass, we have identified a need to inject a (global) expr node ID for the
  RelayExprNode class and its derived classes: Tuple and CallNode, so that during the transformation pass, we can
  uniquely identify each Tuple or CallNode object. Again, we need help from TVM community to provide
  suggestions/guidelines here in order to know whether this is one of the best ways to achieve the Mrvl-BYOC need.

* We also identified a need to maintain linkages between (operator-)information described in the original, given
  pre-trained network model and the code-gen JSON files so that the compiler backend will be able to report user-level
  (e.g., meaningful-to-user) messages regarding the given pre-trained network. For instance, in the
  tvm/python/tvm/relay/frontend/onnx.py and common.py files, we can see user-level information being captured using
  “tvm_custom” related code as in original onnx.py file for the given pre-trained network; but, in common.py, the code
  later drops the linkage, via attrs.pop(“tvm_custom”), and does not pass the linkage onto the initial relay IR graph.
  We have a draft solution to maintain linkages between the given pre-trained network model and its relay IR graph
  (using expr node ID and tvm custom ID, plus, a few utility functions), but would like to know whether the TVM
  community has any better or work-in-progress resolution.

* When using TVM RPC code to exercise and run inference on a remote-hosted Mrvl ML/AI HW accelerator for the Mrvl
  subgraph, we ran into one minor issue and have made local TVM RPC enhancement so that, when a TVM RPC client sends
  a file to the remote server, the TVM RPC client can know where the remote server saves the file on the remote machine.
  Since this is not directly related to this Mrvl-BYOC PR, we will find time to contribute this enhance back in another
  TVM PR soon.

* In order for us to generate the constants-JSON file, we must “NOT” remove external params, which were stored in
  metadata module, in the BuildRelay() function defined in the tvm/src/relay/backend/build_module.cc file. Currently,
  we are using the CPP directive: #ifndef TVM_USE_MRVL to achieve the not-removal requirement for the Mrvl-BYOC flow
  when config.cmake has USE_MRVL ON. We are not sure whether there are side effects due to not removing external params
  in the BuildRelay() function. Are there any other (better) resolution regarding this matter?
  * We also wonder whether this tests/python/relay/test_external_codegen.py test suite's test case,
    test_load_params_with_constants_in_ext_codegen(), needs to be pytest.mark.skipif(True if USE_MRVL is ON)?

# Future possibilities
[future-possibilities]: #future-possibilities

* For this BYOC-Marvell RFC, we are focusing on relay compilation and codegen to generate a Nodes-JSON file and a
  Constants-JSON file. The next thing is to expand to include Marvell driver and runtime supports.

* For the first Marvell-BYOC revision, solution for at most one integration step from a for-accelerator Mrvl subgraph
  to a TVM-target non-Mrvl subgraph is provided for a pre-trained network model. Plus the Mrvl subgraph can only use
  input tensor(s) of given pre-trained model as its subgraph’s input tensors.  How to efficiently handle the integration
  of and data communication between multiple Mrvl subgraphs and multiple non-Mrvl subgraphs at inference runtime will
  be needed.

* Mrvl ML/AI accelerator can run inference in either float16 mode or int8 quantization mode. We are working on a Mrvl
  Bring-You-Own-Quantization-Int8 flow under the tvm/python/tvm/relay/quantize/contrib/mrvl folder. When we have a solid
  POC codebase, we will start to communicate with the TVM Community via another pre-RFC/RFC/PR.
