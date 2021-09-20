- Feature Name: unified-target-device-planning
- Start Date: 2021-09-20
- RFC PR: [apache/tvm-rfcs#0000](https://github.com/apache/tvm-rfcs/pull/0000)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary
[summary]: #summary

TVM currently has separate `Device` and a `Target` abstractions:

- `Device` (aka `DLDevice`, provided by `dlpack`) is intended to be a runtime
  abstraction describing a tensor's storage location. (It used to be called a
  'context', and some references to that term remain in the codebase.) It is
  simply a pair of a `DLDeviceType` enum (eg `kDLCPU`, `kDLCUDA`, etc) and an
  opaque integer representing a 'device identifier'. For the most part TVM
  ignores or defaults the device identifier to zero so only the device type is
  significant. Traditionally TVM used a  single 'default' device for all calls
  to (fused) primitive operators.

- `Target` is intended to be a compile-time abstraction holding all the compiler
  options which influence code-generation. For example `llvm
  -mcpu=skylaxe-avx512`. Traditionally TVM only needed one target to be
  specified so as to guide the compilation of all primitive operators. This is
  because traditionally all tensor dataflow between primitive operators was
  handled by some form of interpreter directly compiled into the runtime (the
  'graph executor', the 'interpreter', or the 'VM'), and traditionally only one
  'default' device was supported for the whole model.

Two recent generalizations to TVM complicate this traditional view:
- TVM now supports 'hetrogenous' execution, in which each primitive operator
  call may be executed and stored on a different device. The desired device
  is indicated by an `on_device` 'annotation', which is just a call to a
  built-in operator with an additional `device_type` attribute. A
  'device planning' analysis/pass associates a device with every primitive
  operator call consistent with those annotations. Since different devices may
  have very different compilation options, we also need a way to recover the
  appropriate `Target` to handle each call. This is currently handled by:
   - Building a `TargetMap` from `DLDeviceType` to `Target`, based on a list
     of provided targets passed into the build API(s).
   - Consulting that table for each call in `LowerTEPass`, using the
     `DLDeviceType` associated to the call by device planning.
- TVM also now supports Ahead-of-time (AOT) compilation for the entire model. In
  this world we can no longer assume everything other than calls to primitive
  operators will be handled by the 'host' interpreter, and we must instead
  clearly distinguish the `Target` for each primitive call from the `Target`(s)
  handling the residual dataflow. This is currently handled by passing both a
  'host target' and traditional 'target' as a pair throughout the compiler, and
  using a 'target annotation' pass to discover which code fragments should be
  compiled for which targets.

This has left us with a few problems:
1. We have independent 'device planning' and 'target annotation' passes, even
   though:
   - they are very similar (take some annotations or labels, find a
     consistent assignment for all sub-expressions w.r.t. those annotations,
     then find the transitions between devices/targets in the AST).
   - they must be coherent, yet currently are handled completely independently
     and probably can't be safetly used together.
2. We require all targets to have a unique `DLDeviceType`. However modern
   platforms can have multiple 'CPU'-like devices (eg Arm 'big.LITTLE') and
   multiple 'GPU'-like devices (eg an actual GPU and an Arm EthosU tensor
   accelerator). Using device type alone to bridge the device and target worlds
   is too rigid.
3. We have multiple conventions for representing available targets in the
   codebase:
    - a list of targets indexed by device type, using the invalid zero
      device type to designate the 'default'.
    - a 'host' and 'device' target pair, with calls to ensure they are
      consistent with each other.
    - a target containing a 'host target' field.
   This makes the code particulaly difficult to maintain.

It seems likely we'll need to support a first-class notion of 'memory scope'
as a refinement of 'device', but we don't want to do that until the above
issues are under control.

In this RFC we propose:
1. We bring `Device` and `Target` together into a new structure:
   ```
   class TargetDevice {
    public:
     Target target;
     Device device;
   }
   ```
   (Eventually this would be extended to include a memory scope.)
2. We introduce a `TargetDevice` registry which maps globally unique
   'target device labels' to 'TargetDevice' objects. The registry may be
   extended 'at compile time' by contrib code, and at runtime via `tvmc`
   command line options. Conventions would be supported so that, e.g.
   `my_accelerator:0` and `my_accelerator:1` (same architecture, but device ids
   0 and 1 respectively) could be referenced from "on_device" calls without 
   having to construct and register for the same `Target` twice.
3. We change "on_device" to use target device labels instead of device types.
4. We allow primitive operators to include `TargetDevice` annotations, for
   example to specify they are available on only specific devices/targets.
5. We unify 'device planning' and 'target annotation' into a single
   'TargetDevice planning' pass. This pass understands:
    - "on_device"
    - Annotations on primitive ops
    - The special handling for other built-ins, such as shape functions.
6. We remove all uses of target maps. For example, in `LowerTEPass` we
   recover the target device label for each primitive operator call and use
   the global label-to-TargetDevice map to recover the Target necessary to
   complete compilation of the primitive.

# Motivation
[motivation]: #motivation

Why are we doing this? What use cases does it support? What is the expected outcome?

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Explain the proposal as if it was already included in the language and you were teaching it to a TVM user. 

That generally means:

- Introducing new named concepts.
- Explaining what the feature enables (hint: think in terms of examples).
- If applicable, provide sample error messages, deprecation warnings, or migration guidance.

For internal RFCs (e.g. for compiler internals), this section should focus on how core contributors s
hould think about the change, and give examples of its concrete impact. 

For policy RFCs, this section should provide an example-driven introduction to the policy, 
  and explain its impact in concrete terms.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

This is the technical portion of the RFC. Explain the design in sufficient detail that:

- Its interaction with other features is clear.
- It is reasonably clear how the feature would be implemented.
- Corner cases are dissected by example.

The section should return to the examples given in the previous section, 
and explain more fully how the detailed proposal makes those examples work.

# Drawbacks
[drawbacks]: #drawbacks

Why should we *not* do this?

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

- Why is this design the best in the space of possible designs?
- What other designs have been considered and what is the rationale for not choosing them?
- What is the impact of not doing this?

# Prior art
[prior-art]: #prior-art

Discuss prior art, both the good and the bad, in relation to this proposal.
A few examples of what this can include are:

- Does this feature exist in other ML compilers or languages and discuss the experince their community has had?
- For community proposals: Is this done by some other community and what were their experiences with it?
- For other teams: What lessons can we learn from what other communities have done here?
- Papers: Are there any published papers or great posts that discuss this? 
  If you have some relevant papers to refer to, this can serve as a more detailed theoretical background.

If there is no prior art, that is fine - your ideas are interesting to us whether they are 
  brand new or if it is an adaptation from other languages.

Note that while precedent set by other languages is some motivation, it does not on its own motivate an RFC.
Please also take into consideration that TVM intentionally diverges from other compilers.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

- What parts of the design do you expect to resolve through the RFC process before this gets merged?
- What parts of the design do you expect to resolve through the implementation of this feature before stabilization?
- What related issues do you consider out of scope for this RFC that could be addressed in the future 
  independently of the solution that comes out of this RFC?

# Future possibilities
[future-possibilities]: #future-possibilities

Think about what the natural extension and evolution of your proposal would
be and how it would affect the language and project as a whole in a holistic
way. Try to use this section as a tool to more fully consider all possible
interactions with the project and language in your proposal.
Also consider how this all fits into the roadmap for the project
and of the relevant sub-team.

This is also a good place to "dump ideas", if they are out of scope for the
RFC you are writing but otherwise related.

If you have tried and cannot think of any future possibilities,
you may simply state that you cannot think of anything.

Note that having something written down in the future-possibilities section
is not a reason to accept the current or a future RFC; such notes should be
in the section on motivation or rationale in this or subsequent RFCs.
The section merely provides additional information.
