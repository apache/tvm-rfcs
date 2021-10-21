- Feature Name: unified-target-device-and-memory-scope-planning
- Start Date: 2021-09-20
- RFC PR: [apache/tvm-rfcs#0038](https://github.com/apache/tvm-rfcs/pull/0038)
- GitHub Issue: [apache/tvm#9327](https://github.com/apache/tvm/issues/9327)

# Summary
[summary]: #summary

TVM supports 'hetrogeneous' execution, whereby primitive operators may be (sequentially) evaluated
on more than one device (GPU, CPU, accelerator, etc). For the non-BYOC flow this works as follows:
1. Relay programs may contain `on_device` annotations which specify that a sub-expression's result
   should reside on a device with a given `DLDeviceType` (`kDLCPU`, `kDLCUDA`, etc).
2. The `PlanDevices` pass uses those annotations to decide the unique device for every Relay
   sub-expression, including every primitive operator call. Sub-expressions which are unconstrained
   are assigned to the 'default' device. The pass then inserts `device_copy` operators whenever data
   needs to cross device boundaries.
3. The user must also supply a list of `Target` objects. The compiler uses that list to build
   a `TargetMap` from `DLDeviceType` to `Target`.
4. Each call to a primitive operator for a particular `DLDeviceType` signals we need to compile
   ('lower') that primitive for that device. The `Target` to use for that compilation is found from
   the `TargetMap` by the `LowerTEPass`.

For the BYOC flow things are quite different:
1. Operators may be annotated with an `FTVMAnnotateTarget` function for a particular
   `target.<name>`. Here `<name>` serves only to distinguish possible BYOC toolchain names and is
   currently not connected to the `Target` machinery in any way. The function should return true if
   the given expression could be compiled for toolchain `<name>`. (However there are currently no
   examples of this annotation in-tree.)
2. The `MergeComposite` pass can be used to assign a `"Composite"` attribute to Relay functions
   which have been hoisted out of a larger expression based on a fusion pattern. The attribute can
   have any value of the form `"some.arbitrary.prefix.<name>"`. Again, this indicates the function
   could be compiled for toolchain `<name>`. (The EthosU compilation flow illustrates this approach
   in-tree.)
3. The `AnnotateTarget` pass looks for the annotations from (1) and (2) to decide the unique
   toolchain name for every Relay sub-expression which should go via a BYOC path. The transitions in
   to and out of those sub-expressions are marked with `compiler_begin` and `compiler_end`
   annotations.
4. The `PartitionGraph` pass hoists sub-expressions delimited by `compiler_begin` and `compiler_end`
   annotations into new top-level `Function`s with a `"Compiler"` attribute bound to the toolchain
   `<name>`.
5. The rest of the compilation flow treats `"Compiler"` annotated functions specially.

We have 6 problems:
1. TVM is being targeted to environments with multiple CPUs (eg Arm 'Big.LITTLE') and multiple
   tensor-friendly devices (eg a GPU as well as an accelerator such as Arm 'Ethos-U'). This means a
   `DLDeviceType` no longer uniquely determines a `Target`.
2. Though TVM's `Device` abstraction (an alias for `dlpack`'s `DLDevice`) is a pair of a
   `DLDeviceType` and an arbitrary 'device id', TVM does not consistently plumb the device id
   through annotations, passes and operators.  Thus currently we cannot use 'device id' to
   distinguish, eg, two CPUs in the same system.
3. Upcoming work requires us to distinguish and propagate memory scopes for data at the Relay
   level. (See also [RFC #9](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0009_Unified_Static_Memory_Planning.md)
   which has a similar need for memory scope propagation at the TIR level). This is an identical
   problem to propagating devices, and it seems most natural to simply combine targets, devices and
   memory scopes into a single 'target of device planing' rather than implementing a whole new pass.
4. Device planning currently has no machinery to hoist adjacent expressions which share the same device
   into their own Relay `Function`. For all our executors except VM that's unnecessary anyway since
   all Relay expressions left over after lowering are interpreted by the runtime. However for AOT we
   have to compile *all* Relay code for a particular target. Note the BOYC machinery does support this,
   but for the purposes of redirecting the compilation flow entirely. We need a middle ground.
5. The BYOC flow is not connected to the `Target` machinery in any way.
6. The BYOC annotate/partition flow is very similar to the device annotate/rewrite flow. For comparison:

   | Feature               | Device Planning            | BYOC                                            |
   | --------------------- | -------------------------- | ----------------------------------------------- |
   | Source of annotations | `on_device`, `device_copy` | `FTVMAnnotateTarget`, `MergeComposite`+patterns |
   | Target of planning    | DLDeviceType               | Toolchain name                                  |
   | Propagation           | Unification based          | Ad-hoc                                          |
   | Relay support         | Full                       | First-order, no ADTs                            |
   | Delimiting            | insert `device_copy`       | insert `compiler_begin`, `compiler_end`         |
   | Multiple per expr     | No                         | Yes (though always picks first)                 |
   | Hoists into functions | No                         | Yes                                             |
   | Customized heuristics | No                         | No                                              |

   Taking the 'upper bound' of the two implementations seems ideal, especially to address issues 4 (limitation
   of device planning) and 5 (limitation of BYOC) above.

Our proposal is:
1. We introduce a new FFI-friendly class to represent a *S*torage or *E*xecution *Scope*:

   ```
   class SEScope {
     DLDeviceType device_type;
     int virtual_device_id;
     Target target;
     String memory_scope;
   }
   ```

   We allow each of these fields to be independently 'constrained' (ie have a specific value) or
   'unconstrained' (no specific value for the field is known yet). In particular, it is valid for
   an `SEScope` to contain only a `device_type`. However if the `target` field is defined then
   `device_type` must equal `target->kind->device_type`.

2. At this stage we leave the `memory_scope` field uninterpreted. For example, we don't attempt to
   represent that, eg, `"global"` on a `kDLCPU` is the same memory area as `"host"` on a `kDLCUDA` and thus no
   `device_copy` operation is required between those scopes. We'll pick this issue up again after
   [RFC #9](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0009_Unified_Static_Memory_Planning.md)
   has landed.

3. The `on_device` and `device_copy` call attributes use `SEScope`s instead of integers. However the Python
   bindings for these 'operators' continue to accept a `Device` for convenience. The machinery in `LowerTEPass`
   which resolves `DLDeviceTypes` to `Targets` is moved up in the compilation flow and becomes part of
   `PlanDevices`. In particular, any `SEScope` encountered during device planning is 'canonicalized' to fill
   in a `Target` by the same lookup as we do today. This means we continue to support the easy shorthand of
   referring to devices by the `DLDeviceType` alone. However, advanced users can supply a `SEScope` to these
   operators which contains the exact `Target` to use.

4. We rework device planning to be in terms of `SEScope`s instead of `DLDeviceTypes`. Two `SEScope`s
   become special:
    - We need a default scope for all primitive operators which are not otherwise
      constrained to a particular scope.
    - We need a scope for 'host-only' operations and data, such as for shapes and shape functions.
      (Currently this is hardcoded to `kDLCPU`).

5. We extend `PlanDevices` to be able to a) run *after* lowering and b) refine existing constraints.  It will
   look inside calls to `PrimFunc`s and follow the chain:

   ```
   tir::PrimFunc.buffer_map -> tir::Buffer.data -> tir::Var.type_annotation -> PointerType.storage_scope -> String
   ```

   to discover the memory scope for each Relay argument. That scope will enter `SEScope`s and flow through the
   existing unification machinery. The existing sub-pass in `PlanDevices` will insert `device_copy` calls
   wherever sub-expressions disagree on their memory scope.

   (An additional pass is planned to heuristically move `device_copy`s around, and eliminate redundant
    copies, however that's outside the scope of this RFC.)

6. We rework `PartitionGraph` to `PartitionBySEScope` to work on `SEScope` annotations instead of
   `compiler_begin` and `compiler_end` annotations. Algorithmically it's not a big change -- maximal
   sub-expressions which share the same `SEScope` (or a projection thereof, eg just the `target`) are hoisted
   into global `Function`s. The function's `"result_se_scope"` attribute describes both the scope holding the
   function's result *and* the `Target` for which the function is to be compiled.

7. We allow `MergeComposite` to be used to insert `on_device` annotations, call it `MergeAndAnnotate`.

8. (?) We rework `AnnotateTarget` to just look for `FTVMAnnotateTarget` operator attributes, call it
   `AnnotateSEScopes`. When the function fires an `on_device` annotation is inserted. However since
   there are no examples of these attributes being used in-tree perhaps this is dead code?

9. (?) We rework `PlanDevices` to support collecting multiple candidate `SEScopes`, mimicking the
   current behavior in `AnnotateTarget`. However,  since the current behavior simply picks the
   first toolchain name, and we don't currently have any passes which attempt to solve the
   (very hard) device selection problem, this work may be best deferred till we understand more.

10. We retire the BYOC `MergeComposite`/`AnnotateTarget`/`PartitionGraph` flow in favor of the
    `MergeAndAnnotate`/`AnnotateSEScopes`/`PlanDevices`/`PartitionBySEScope` flow. BYOC hooks which
    are currently keyed by toolchain name can instead be keyed by `Target`.

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
