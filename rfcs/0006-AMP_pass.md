- Feature Name: Automatic Mixed Precision Pass
- Start Date: 2021-06-08 
- RFC PR: https://github.com/apache/tvm-rfcs/pull/6
- GitHub Issue: https://github.com/apache/tvm/issues/8296

# Summary
[summary]: #summary

Many pieces of hardware support arithmetic not only on IEEE 32 bit floating point numbers, but also IEEE 16 bit floating point numbers. 
These 16 bit operations typically have higher theoretical throughput and involve less use of memory bandwidth.
As a result, we can see significant increases in speed from changing 32 bit floating point operations into 16 bit analogs for many models.
Surprisingly, this change has little affect on the results of some models, though some care must had when changing a select few
operations. Some 16 bit floating point operations such as `exp` and `log` for example are considered unsafe to use 16 bit analogs 
due to loss of [numerical precision](https://on-demand.gputechconf.com/gtcdc/2019/pdf/dc91247-automatic-mixed-precision-in-tensorflow.pdf). 
In general for a function `f`, if `|f(x)| >> |x|` for expected ranges of input we probably do not want to use the 16 bit floating point versions.

This RFC describes a relay pass which automatically converts a 32 bit floating point model into a reduced bit 
floating point analog. For the initial work, IEEE's 16 bit floating point will be targeted though future support
for bfloat16 will be held in mind. Additionally, we discuss some additional work that must be done to support 16 bit floating point 
on some common targets.

# Motivation
[motivation]: #motivation

Many machine learning models can move significant portions of their computational graphs into the FP16 space 
without significant loss of accuracy. For many pieces of hardware this also comes with a boost in speed. For example, 
Pytorch saw utilizing FP16 in mixed precision training saw significant [increases in convergence speed](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/). 

We should expect similar increases for inference. This speed increase without significant accuracy loss is highly desirable
for many users.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Operations are partitioned into categories denoted "ALLOW", "DENY", and "FOLLOW" which represents the benefit 
of using a reduced floating point version of the operation. "ALLOW" operations are compute intensive
and almost always see hardware memory and latency savings by utilizing a reduced floating point form.
Examples of these operations are matrix multiplication and convolutions. "FOLLOW" operations see little to 
no savings in using reduced floating point forms -- at least not enough to justify the overhead of 
casting values back and forth from FP32. "DENY" operations meanwhile are operations we do not want to 
use reduced floating point forms on, usually due to numerical precision reasons.

In general we always want to insert casts into reduced floating point space for "ALLOW" operations, 
are fine with transforming "FOLLOW" operations into reduced floating point space if their inputs are already
in that form, and want to explicitly cast back into full floating point space for "DENY" operations. 
Each operation will be placed into one of these lists via a "coloring" function which take in Relay `CallNodes`
and returns a color. For example, we might have a function which colors only a convolution as "ALLOW" if it 
has a large enough kernel and "FOLLOW" otherwise. For the default implementation we will keep things simple
however and do something like place all convolutions in the "ALLOW" list, all element-wise operations in 
the "FOLLOW" list, and so on. Still, the code will be designed to be easily extensible via overwriting 
this "coloring" function.

The final variable we must keep in mind is the fact that some hardware platforms can operate on reduced
floating point types. However, while they for example may take two FP16 operands they may accumulate the 
result in a 32 bit buffer. An example of this are the Tensor Cores in Nvidia's Turing architecture. 
The final knob we give is a control over how operations accumulate their result. For this, we have 
a function, which maps operation types like `conv2d` to an accumulation datatype as well as an output 
datatype. The output datatype is the type other operations down the line will likely ingest from the previous
calculation while the accumulation datatype describes the size of buffer where the results are initially
stored. For NVidia's tensor cores for example many operations accumulate in FP32 but have an output datatype
of FP16. The default implementation will follow this guideline closely and will by default have all 
operations output FP16 and accumulate in FP32 only if TVM supports mixed datatypes for that particular
operation.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

See [previous discussion thread](https://discuss.tvm.apache.org/t/rfc-relay-fp32-fp16-model-support/9994).

As some have noticed the design can be simplified to a single pass where casting is determined by
running type inference on mutated nodes. With a post-order traversal we can then check if we need to 
cast arguments/propagate color.

Part of the associated RFC issue will also be used dedicated to creating a tutorial on how to control
the conversion of ops via the Python interface. Furthermore, some work will be done in benchmarking
the performance gains from the pass.

# Drawbacks
[drawbacks]: #drawbacks

If this is not useful, we are just adding an additional pass which will do nothing. Furthermore we 
will have to make sure it works on a wide range of models or people will be very mad at TVM.

This might not be useful if mixed precision training becomes super popular in the future in which 
case most models might be in a reduced precision floating point form already.

It also might not be useful if integer quantization becomes super popular, though it may be possible
to mix integer quantization and mixed floating precision techniques. Floating point does have 
several advantages still over integer quantization including simplicity and the fact that some 
operators like `sin` and `erf` are still designed in hardware with floating point in mind.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

- Why is this design the best in the space of possible designs?

Other alternatives require a lot more work and changes and could probably considered future goals of TVM.
This include automatic mixed precision training.

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

We still need to make sure that the current design and knobs exposed provide extensibility to every hardware platform out there.

- What parts of the design do you expect to resolve through the implementation of this feature before stabilization?

Probably a lot of edge cases of operations within TVM.

- What related issues do you consider out of scope for this RFC that could be addressed in the future 
  independently of the solution that comes out of this RFC?

Making accumulation datatypes a standard idea for all operations within TVM.

# Future possibilities
[future-possibilities]: #future-possibilities

Really this can be used for any floating point datatype. A custom FP24 for FPGA? 
BFloat16? Some other weird floating point type? We have an easy way to convert 
toward utilizing these weird floating point types with FP32 when appropriate
under this framework.
