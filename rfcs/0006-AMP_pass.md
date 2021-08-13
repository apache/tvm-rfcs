- Feature Name: Automatic Mixed Precision Pass and support
- Start Date: 2021-06-08 
- RFC PR: https://github.com/apache/tvm-rfcs/pull/6
- GitHub Issue: https://github.com/apache/tvm/issues/8296

# Summary
[summary]: #summary

Many pieces of hardware support arithmetic not only on IEEE 32 bit floating point numbers, but also IEEE 16 bit floating point numbers. 
These 16 bit operations typically have higher theoretical throughput and involve less use of memory bandwidth.
As a result, we can see significant increases in speed from changing 32 bit floating point operations into 16 bit analogs for many models.
Surprisingly, this change has little effect on the results of some models, even without retraining, though some care must had when changing a select few
operations. For example, some 16 bit floating point operations such as `exp` and `log` are considered generally unsafe in the 16 bit floating point space
due to loss of [numerical precision](https://on-demand.gputechconf.com/gtcdc/2019/pdf/dc91247-automatic-mixed-precision-in-tensorflow.pdf). 
In general for a function `f`, if `|f(x)| >> |x|` for expected ranges of input we probably want to stick to 32 bit floating point versions.
As a result, within models, 16 bit floating point is often interspersed with 32 bit floating point operations for unsafe operations. The usage of 
differing precision for floating point in a model is often called "Mixed Precision."

This RFC describes a plan to support automatic mixed floating point precision models within TVM. Specifically, we focus on the conversion 
of an existing, trained 32 bit floating point model, into a mixed precision model without retraining. Note, we do not focus on the conversion
of models already operating in a mixed precision space though much of the work being done will help guarantee support for these models.

In particular we focus discussion on the following areas:
- Creating a pass to automatically transform a 32 bit floating point Relay model into a 16 bit analog
- The changes in the intermediate representation of TVM that must be made to ensure wide operator support for FP16
- Issues in some codegen pathways that must be address to ensure wide support for FP16.

For the initial work, IEEE's 16 bit floating point will be targeted though future support for bfloat16 will be held in mind. 

# Motivation
[motivation]: #motivation

Many machine learning models can move large portions of their computational graphs into the FP16 space 
without significant loss of accuracy. For many pieces of hardware this also comes with a boost in speed. For example, 
PyTorch utilized FP16 in mixed precision training and saw significant [increases in training speed](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/). 

We should expect similar increases in speed for inference. 

This speed increase without significant accuracy loss is highly desirable for many users.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

The mixed precision pass operates on Relay models and their operations.

We define an operator as in "mixed precision" space if it's inputs are in reduced precision form (e.g. FP16).

Operations are partitioned into category lists denoted "ALLOW", "DENY", and "FOLLOW" which represents the benefit 
of using a reduced floating point version of the operation. "ALLOW" operations are compute intensive
and almost always see hardware memory and latency savings by utilizing a reduced floating point form.
Examples of these operations are matrix multiplication and convolutions. "FOLLOW" operations see little to 
no savings in using reduced floating point forms -- at least not enough to justify the overhead of 
casting values back and forth from FP32. "DENY" operations meanwhile are operations we do not want to 
use reduced floating point forms on, usually due to numerical precision reasons.

We always want to move "ALLOW" operations into mixed precision space by casting their inputs, 
are fine with transforming "FOLLOW" operations into mixed precision space space if their inputs are already
in reduced form, and want to explicitly cast back into full floating point space for "DENY" operations. 
Each operation will be placed into one of these lists via a function which take in Relay `CallNodes`
and returns either "ALLOW", "DENY", or "FOLLOW. For example, we might have a function which colors only 
a convolution as "ALLOW" if it has a large enough kernel and "FOLLOW" otherwise. 

The final consideration is using higher bit accumulators. For example, for a global average pool, we might
have 16 bit floating point inputs, but accumulate the result in a 32 bit floating point buffer in order to 
maintain numerical information. As a result, we must have a way to communicate whether an operator should 
accumulate results in a higher bit buffer. An example of hardware with native support for this sort of operation 
are the Tensor Cores in Nvidia's Turing architecture. For NVidia's Tensor Cores for example have many operations 
accumulate in FP32 but have an output datatype of FP16. 

The interface to control the conversion of an operator for the mixed precision pass is therefore as follows:
   - Write a function in python which given a Relay CallNode, decides whether it should be in the "ALLOW", 
        "FOLLOW", or "DENY" lists of operations. Furthermore, the function should decide the accumulation 
        and output datatypes of the operation, though these are only used if the operator will be in mixed
        precision space.
  ```python
  def mixed_precision_func(call_node: "relay.Call", mixed_precision_dtype: str) -> Tuple[int, str, str]:
    """
    Parameters
    ----------
    call_node:
        A Relay Call node which is currently being examined by the mixed precision pass.

    mixed_precision_dtype:
        The datatype of the mixed precision pass (i.e. usually float16).

    Returns
    -------
    result : Tuple[int, str, str]
        A tuple where the first element (int) represents a code describing the operation as belonging to "ALLOW", "DENY", or "FOLLOW" lists.
        The second element describes the accumulation datatype of the operation (i.e. usually float32 or mixed_precision_dtype). The third 
        element describes the output datatype of the operation (i.e. usually mixed_precision_dtype). 
    """
  ```
   - Register the function as an operator attribute with a provided function:
  ```python
  def register_mixed_precision_conversion(op_name, func=None, level=10):
      """Register mixed precision conversion function for an op

      Given an op the function should return information on how the value should be
      converted. Specifically the function should take a call node and the target
      mixed precision datatype (e.g. FP16) and return the conversion category
      (see python/tvm/relay/transform/mixed_precision.py) as well as the accumulation
      and output datatype of the operation in the mixed precision dtype space.

      Parameters
      ----------
      op_name : str
          The name of the operator

      func: function (call_node: relay.Call, target_dtype: string)
      -> [conversion category, accumulation dtype, output dtype]: [int, string, string]
          A function which given a call_node and target_dtype (e.g. FP16) returns the
          conversion category and associated accumulation/output of the operation
          when transformed into the mixed precision dtype space.

      level : int
          The priority level
      """
  ```
   - An example of creating a function which operates on a Conv2D operator and registering it is as follows:
  ```python
  import math 

  MIXED_PRECISION_ALWAYS = 0
  MIXED_PRECISION_FOLLOW = 1
  MIXED_PRECISION_NEVER = 2

  def conv2d_mixed_precision_func(call_node: "relay.Call", mixed_precision_dtype: str) -> Tuple[int, str, str]:
    """Note this won't work for dynamic shaped inputs."""
    accumulation_dtype = "float32"
    output_dtype = mixed_precision_dtype
    
    input_shape_elements = math.prod(call_node.op.data.shape)

    # Always convert to mixed precision if the input is big enough, else move to follow list
    if input_shape_elements > 100:
      return (MIXED_PRECISION_ALWAYS, accumulation_dtype, output_dtype)
    return (MIXED_PRECISION_FOLLOW, accumulation_dtype, output_dtype)

  # Register conversion function for conv2d
  register_mixed_precision_conversion("nn.conv2d", conv2d_mixed_precision_func)
  ```
   - After registering the appropriate operators for the model. We then invoke the mixed precision pass. Note we want to rerun some 
         other graph optimizations afterwards:
  ```python
  def convert_to_fp16(mod, params, fast_math=True):
    mod = tvm.IRModule.from_expr(mod["main"])

    # Run safe operations to simplify graph
    mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
    mod = tvm.relay.transform.FoldConstant()(mod)

    # Run main mixed precision pass
    mod = InferType()(mod)
    mod = ToMixedPrecision()(mod)

    # Run more passes to clean up new graph
    mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
    mod = tvm.relay.transform.FoldConstant()(mod)
    mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
    mod = tvm.relay.transform.FoldConstant()(mod)
    mod = tvm.relay.transform.FastMath()(mod) if fast_math else mod

    return mod, params
  ```
    - We then have a Relay model in mixed precision form!

With this interface, every single Relay operator within a model will belong to "ALLOW", "FOLLOW", or "DENY" lists and is 
accordingly transformed into a mixed precision form. By default, unregistered operators will always be assumed to be in the 
"FOLLOW" list of operations and accumulate and output results as the mixed precision dtype. A default registry of functions will also be provided and be based
on TensorFlow's [similar feature](github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/core/grappler/optimizers/auto_mixed_precision_lists.h).

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

See [previous discussion thread](https://discuss.tvm.apache.org/t/rfc-relay-fp32-fp16-model-support/9994).

As some have noticed the design can be simplified to a single pass where casting is determined by
running type inference on mutated nodes. With a post-order traversal we can then check if we need to 
cast arguments/propagate color.

Part of the associated RFC issue will also be used dedicated to creating a tutorial on how to control
the conversion of ops via the Python interface. Furthermore, some work will be done in benchmarking
the performance gains from the pass.

## Pass implementation

## Other changes needed in TVM

## Codegen issues 

## Plan for benchmarking 

# Drawbacks
[drawbacks]: #drawbacks

If this is not useful, we are just adding an additional pass which will do nothing. Furthermore we 
will have to make sure it works well on a wide range of models or people will be very mad at TVM.
This necessitates good default definitions for automatic mixed precision conversion for operators
which we will have to maintain. Furthermore, additional work needs to be done in order to ensure 
good coverage of support for this pass.

Furthermore, this pass might not be useful if mixed precision training becomes super popular in the 
future in which case most models might be in a reduced precision floating point form already.

It also might not be useful if integer quantization becomes super popular, though it may be possible
to mix integer quantization and mixed floating precision techniques. Despite this, automatic mixed 
precision has an advantage of having a lesser accuracy loss compared to integer quantization, especially
when models are not retrained. This makes it very useful as a simple flag that can be turned on for 
every trained model to essentially get a free speed increases.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

- Why is this design the best in the space of possible designs?

Other alternatives require a lot more work and changes and could probably considered future goals of TVM.
This include automatic mixed precision training. It also operates on the Relay level which is the right
place to make these changes and seems to mostly work mostly well out of the box excepting a few issues.

- What other designs have been considered and what is the rationale for not choosing them?

We can support automatic mixed precision retraining though that is a much, much larger future goal. It's
good to have this in the meantime.

- What is the impact of not doing this?

TVM is not the best tool for making models go fast as we leave a lot of free speedup on the table.

# Prior art
[prior-art]: #prior-art

Many of the ideas are taken from Tensorflow's [automatic mixed precision training framework](https://on-demand.gputechconf.com/gtcdc/2019/pdf/dc91247-automatic-mixed-precision-in-tensorflow.pdf)
and the initial "ALLOW", "FOLLOW", and "DENY" lists are based [similarly](github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/core/grappler/optimizers/auto_mixed_precision_lists.h). 

# Unresolved questions
[unresolved-questions]: #unresolved-questions

- What parts of the design do you expect to resolve through the RFC process before this gets merged?

Feedback on the user interface in the pass will be appreciated. The creation of default conversion methods for operators
is another topic discussion.

- What parts of the design do you expect to resolve through the implementation of this feature before stabilization?

There are likely many misc. TVM changes that must be made in order to better support FP16 execution as discussed above.
We will deal with these issues as we encounter them, though believe that in general these are issues not specific 
to the pass in general but rather FP16 support throughout all of TVM.

- What related issues do you consider out of scope for this RFC that could be addressed in the future 
  independently of the solution that comes out of this RFC?

Making accumulation datatypes a standard idea for all operations within TVM. Furthermore, having good coverage
for the conversion of existing mixed precision models.

# Future possibilities
[future-possibilities]: #future-possibilities

Really this can be used for any floating point datatype. A custom FP24 for FPGA? 
BFloat16? Some other weird dtype? We have an easy way to convert models 
toward utilizing exotic types with FP32 when appropriate under this framework.
