- Feature Name: improved-multi-target-handling
- Start Date: 2021-09-20
- RFC PR: [apache/tvm-rfcs#0000](https://github.com/apache/tvm-rfcs/pull/0000)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary
[summary]: #summary

TVM supports 'hetrogeneous' execution, whereby primitive operators may be (sequentially) evaluated on more than
one device (GPU, CPU, accelerator, etc). For the non-BYOC flow this works as follows:
1. Relay programs may contain "on_device" annotations which specify that a sub-expressions's result should
   reside on a device with a given `DLDeviceType` (kDLCPU, kDLCUDA, etc).
2. The device planning pass uses those annotations to decide on the unique device for every Relay sub-expression,
   including every primitive operator call. Sub-expressions which are unconstrained are assigned to the 'default'
   device. The pass then inserts "device_copy" operators whenever tensors need to cross device boundaries.
3. The user/driver must also supply a list of `Target` objects. The compiler uses that list to build a `TargetMap`
   from `DLDeviceType` to `Target` for all of those objects.
4. Each call to a primitive operator for a particular `DLDeviceType` signals we need to compile ('lower') that
   primitive for that device. The `Target` to use for that compilation is found from the `TargetMap`.

This approach has 5 problems:
1. TVM is being targeted to environments with multiple CPUs (eg Arm 'Big.LITTLE') and multiple tensor-friendly
   devices (eg a GPU as well as an accelerator such as Arm 'Ethos-U'). This means a `DLDeviceType` no longer
   uniquely determines a `Target`.
2. Though TVM's `Device` abstraction (an alias for `dlpack`'s `DLDevice`) is a pair of a `DLDeviceType` and an
   arbitrary 'device id', TVM does not consistently plumb the device id through annotations, passes and operators.
   Thus currently we cannot use 'device id' to distinguish, eg, two CPUs in the same system.
3. The codebase still uses an older `target` and `target_host` convention for distinguishing the main `Target` for
   primitive operators from the `Target` for residual tensor computation, shape computation, and (for AOT) the
   overall Relay control-flow.
4. `Target`s are often manufactured on-the-fly (eg to represent the default 'CPU' target on which shape computations
   should be hosted). However there's no guarantee those default `Target`s will match up with the user-supplied
   `Target`s, thus it's possible to end up with `"llvm"` and `"llvm -m ..."` `Targets` coexisting. Now that
   `IRModule` uses `Target` objects themselves to distinguish which `PrimFunc`s are intended for which targets,
   it is particularly important to ensure there's a single source of truth for available `Target`s.
5. TVM also supports a 'BYOC' extension mechanism. This allows "target.<target name>" annotations to be placed on
   primitive operations to indicate they should possibly be compiled with the matching BYOC toolchain. A target
   annotation pass uses those annotations to decide on a target name for every Relay sub-expression. A partition graph
   pass then inserts function call boundaries whenever execution needs to cross target boundaries. However this
   machinery is separate from and incompatible with the "on_device" mechanism, and 'target names' are a separate
   concept from `Target` objects.

In this RFC we tackle problems 1-4. We won't directly take on 5 since it involves more moving parts, but our hope
is for this RFC to clear the way to taking on 5 in the future.

Our proposal is:
1. Extend `Target` to have a `DLDeviceType` attribute.
2. Allow `Target` objects to be registered under a globally unique target label. Registration may be 'static' (ie
   built into the TVM compiler via another REGISTER macro) and 'dynamic' (ie injected for a particular run of the
   compiler, eg as part of `tvmc` command line processing). (This machinery should be reconciled with the existing
   CUDA-specific target registration map.)
3. Change the "on_device" call attributes to use a string instead of an integers (ie `DLDeviceType`). The string
   can be of the form `<target label>` or `<target label>:<device id>`. The former simply implies a device id of 0.
4. Rework device planning to use a pair of `Target` and 'device id' instead of `DLDeviceType`:
   ```
   class TargetDevice {
    public:
     Target target;
     int device_id;
   }
   ```
   (We could also use a `Device` and accept the redundant `DLDeviceType` specification.) It is trivial
   to go from an "on_device" label to a `TargetDevice` and back using the global `Target` registry.
5. Remove all uses of `TargetMap`. For example, in `LowerTEPass` we simply use the `TargetDevice` associated with
   every primitive operator call already found by device planning.
6. Bind two `TargetDevice`s as attributes on every `IRModule`:
    - The default for primitive operators not otherwise constrained by "on_device" annotations.
    - The default for non primitive operators, such as Relay control flow and shape computation.
7. We remove the various copies of target/target_host reconciliation, `TargetMap`
   construction and 'default/fallback' device calculation from the codebase.

This proposal tackles the original problems:
1. There's now no ambiguity about `Targets` since we propagate them from the global registry directly.
2. We support device ids.
3. We always know the `Target` for every sub-expression and don't need to pass around the `target` and
   `target host` separately.
4. `Targets` are never created on the fly, they are first registered then propagated.
5. The global registration implied by the existing BYOC target names is now more similar to how the mainline
   `Target`s are handled.


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
