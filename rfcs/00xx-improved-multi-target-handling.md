- Feature Name: improved-multi-target-handling
- Start Date: 2021-09-20
- RFC PR: [apache/tvm-rfcs#0000](https://github.com/apache/tvm-rfcs/pull/0000)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary
[summary]: #summary

Traditionally TVM managed two execution environments:
1. A device, such as a GPU, which would execute the 'inner' parts of fused primitive tensor operators.
2. A host, such as a CPU, which would a) execute any residual 'outer' parts of primitive tensor operators, and
   b) coordinate the control and data flow between those operators. The later is managed by an executor (graph,
   interpreter or VM) compiled directly into the TVM runtime.

The compilation options for host and device are grouped into a pair of `Target` objects, typically called `target` and
`target_host`, which hold all the compiler flags and settings needed to influence code-generation. For example
`cuda` or `llvm -mcpu=skylaxe-avx512`. Device and host may be the same target.

At runtime, tensors are managed by the `dlpack` library which provides a `Device` abstraction. That is a pair
of a `DLDeviceType` enum (eg `kDLCPU`, `kDLCUDA`, etc) and an opaque integer representing a 'device
identifier'. (However since TVM mostly ignores or defaults the device identifier to zero in effect TVM uses
`DLDeviceType` alone to identify devices.) Thus at runtime we also need a pair of `Device` objects
corresponding to our two `Target` objects.

(Note that the codebase still refers to 'device' as 'context' in a few places.)

Two more recent generalizations to TVM complicate this story:

- TVM now supports 'hetrogenous' execution, in which each primitive operator call may be executed and
  stored on a different device. The desired device is indicated by an `on_device` 'annotation', which is just
  a call to a built-in operator with an additional `device_type` attribute. A 'device planning' analysis/pass
  associates a device with every primitive operator call consistent with those annotations.

  Each call to a primitive operator for a particular `Device` signals we need to compile ('lower') that
  primitive for the device, which requires a matching `Target`. This is currently handled by:
   - Building a `TargetMap` from `DLDeviceType` to `Target`, based on a list of provided targets passed into
     the build API(s).
   - Consulting that table for each call in `LowerTEPass`, using the `DLDeviceType` associated to the call by
     device planning.
   - Using an entry for the invalid 'zero' device type to indicate the 'default' device.

  However TVM is being targetted to architectures with multiple CPUs (eg Arm 'Big.LITTLE') and
  multiple devices (eg a GPU as well as an accelerator such as Arm 'Ethos-U'). So we can no longer
  assume a `DLDeviceType` uniquely identifies a device and it's appropriate `Target`. The `TargetMap`
  convention also interacts poorly with the `target` and `target_host` convention, and the codebase
  has gotten messy at those points.

- TVM also now supports Ahead-of-time (AOT) compilation for the entire model rather than just the primitive
  operators. This means we need to be explicit about the `Target` responsible for executing every Relay
  sub-expression. Generally this is assumed to be the host target, however support for AOT has also required
  support for Bring-your-own-compiler (BYOC) for 'embedded' targets. This has resulted in a compilation
  flow very similar to device planning whereby `Target` annotations are used to decide which 'compiler' is
  to be used for each Relay sub-expression.

  However this machinely works independently of the above device planning, and it's not clear how they
  would ever interact. The gap between `Target` and `Device` make reconciliation difficult.

In this RFC we propose:
1. Use a combination of `Target` and `Device` as the unit of planning in 'device planning':
   ```
   class TargetDevice {
    public:
     Target target;
     Device device;
   }
   ```
   (Eventually this would be extended to include a memory scope.)
2. Allow `TargetDevice` objects to be registered under a globally unique `TargetDeviceLabel` (ie a
   string). Registration may be 'static' (ie built into the TVM compiler) or 'dynamic' (ie injected for a
   particular run of the compiler, eg on the `tvmc` command line).
3. We change the "on_device" and "device_copy" call attributes to use `TargetDeviceLabel`s instead
   of integers (ie device types).
4. We allow primitive operators to include (sets of) `TargetDeviceLabel`s, for example to specify they are
   available only on specific devices/targets.
5. We remove all uses of target maps. For example, in `LowerTEPass` we
   recover the `TargetDeviceLabel` for each primitive operator call and use the global `TargetDevice` registry
   to recover the `Target` necessary to complete compilation of the primitive, and the `Device`s needed to effect
   any tensor copies.
6. We gather the various `Target` and `Device` defaults into a single `CompileOptions` class:
     - The default `TargetDeviceLabel` for primitive operators.
     - The default `TargetDeviceLabel` for non primitive operators, such as
       Relay control flow and shape computation.
7. We remove the various copies of target/target_host reconciliation, TargetMap
   construction and 'default/fallback' device calculation from the codebase in favor
   of the centralized `CompileOptions` class.
8. We attach the `CompileOptions` class to an `IRModule` attribute.

We stop short of actually changing the current BYOC TargetAnnotation machinery. But our intent is to
at least remove as many accidental differences to make the next step clear.

-------- rest still in template form --------

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
