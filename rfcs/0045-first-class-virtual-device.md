- Feature Name: Add virtual device information as a first class field on Relay expressions.
- Start Date: 2021-11-20
- RFC PR: [apache/tvm-rfcs#1111](https://github.com/apache/tvm-rfcs/pull/0045)
- GitHub Issue: [apache/tvm#1111](https://github.com/apache/tvm/issues/9665)

# Summary
[summary]: #summary

I propose adding a new field to Relay expressions, virtual_device_. This field will contain virtual device information [currently called SEScope].

# Motivation
[motivation]: #motivation

Currently, the virtual device information (called SEScope today, but we will rename it soon) is stored in Function attributes and in on_device Relay ops. This op is a wrapper op that contains the virtual device information for an expression.

Here's an example of how the virtual device information is stored in the program today (example from test_pass_plan_devices.py):

(note that SEScope is just the virtual device, but we have not renamed it yet).

```
"""
#[version = "0.0.5"]
def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
          %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32],
           param_se_scopes=[meta[SEScope][0], meta[SEScope][0], meta[SEScope][1], meta[SEScope][1]],
           result_se_scope=meta[SEScope][1]) {
  %0 = add(%a, %b);
  %1 = on_device(%0, se_scope=meta[SEScope][0], is_fixed=True);
  %2 = device_copy(%1, src_se_scope=meta[SEScope][0], dst_se_scope=meta[SEScope][1]);
  %3 = add(%c, %d);
  subtract(%2, %3)
}
"""
```

Using this method to store the virtual device information has proven to be very fragile.

Normal visitors that don't care about virtual devices need to peek inside on_device ops.

Additionally, we need DeviceAware visitors to be able to know the virtual device of sub-expressions. Notice in the example above that on_device doesn't wrap every expression. Let's say we want to know the virtual device of %3 while visiting it. We can't look it up directly since the information is not stored on the node. So how do we get the information? Well, instead of a normal visitor, we need to use a DeviceAware visitor, which keeps track of the current virtual device when it visits sub-expressions. We can then get the virtual device from the DeviceAware visitor itself.

Making virtual devices first class eliminates the need for this complexity, and will allow us to implement more features in device and memory planning in the future.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Users can introduce new information about the virtual device through the on_device op. This behavior is the same as it was before.

However, let's say you want to write a pass that uses the virtual devices after device planning. Now, you'll be able to use the virtual device directly in your pass, just like you can with the checked_type_ field.

For example, in this visitor, we can just look at the virtual device directly. 

```
  Expr VisitExpr_(const LetNode* let_node) final {
    Expr expr = GetRef<Expr>(let_node);
    // Iterate through chained lets, provided they all agree on their device type.
    SEScope scope = expr->virtual_device_;
    ...
 ```

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

The AST change will occur in include/tvm/ir/expr.h:

```
class RelayExprNode : public BaseExprNode {
 public:
  /*!
   * \brief Stores the result of type inference(type checking).
   *
   * \note This can be undefined before type inference.
   *       This value is discarded during serialization.
   */
  mutable Type checked_type_ = Type(nullptr);
  /*!
   * \return The checked_type
   */
  inline const Type& checked_type() const;
  /*!
   * \brief Check if the inferred(checked) type of the Expr
   *  is backed by a TTypeNode and return it.
   *
   * \note This function will thrown an error if the node type
   *       of this Expr is not TTypeNode.
   *
   * \return The corresponding TTypeNode pointer.
   * \tparam The specific TypeNode we look for.
   */
  template <typename TTypeNode>
  inline const TTypeNode* type_as() const;

  /*!
   * \brief The virtual device (SEScope) for this node (the result of device planning).
   *
   * \note Unfortunately, the type of virtual_device_ needs to be ObjectRef to avoid a circular import.
   *       We can forward-declare the SEScope type for the getter function, but not for the field
   *       itself.
   */
  mutable ObjectRef virtual_device_;

  /*!
   * \return The virtual device (currently called SEScope, this will be changing soon.)
   */
  SEScope virtual_device() const;

  static constexpr const char* _type_key = "RelayExpr";
  static constexpr const uint32_t _type_child_slots = 22;
  TVM_DECLARE_BASE_OBJECT_INFO(RelayExprNode, BaseExprNode);
};
```

Additionally, I will add virtual_device_ to the WithFields methods.

# Drawbacks
[drawbacks]: #drawbacks

One challenge with making virtual devices first class is that passes in TVM do not propagate all fields when they visit expressions. You can see this today with spans-- most visitors do not preserve spans. When we introduce the virtual device field, we will need to ensure that it is propagated correctly throughout the Relay program. To do this, I introduced WithFields (code: https://github.com/apache/tvm/blob/main/src/relay/ir/expr.cc#L79-L99), a COW constructor that copies extra fields, including spans.  I will extend WithFields to also copy virtual devices. Then, we can use WithFields to ensure that the virtual device field is correctly propagated. 

Additionally, passes that move or introduce expressions will need to cooperate with device annotation (i.e., figure out the correct device themselves and insert it in any expressions they create), or device planning will need to be run again after the pass (similar to the type inference pass). In the future, we'd like to introduce a 'lite' version of the device planning pass that "fills in the gaps" left by these passes. It will propagate virtual device information strictly upwards and will have much less overhead than the full device planning pass.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

The current implementation is the main alternative to making virtual devices first class. We actually considered making virtual devices first class while designing the current implementation, but we decided against it because of challenges propagating the virtual devices correctly, and we weren't sure how fundamental the virtual devices were.

Another option we considered was adding the virtual device information as an attribute to each node instead of adding it as a first-class field on each node. However, not all Relay nodes have attributes, so we would have to add attributes for all the Relay nodes as well. This is also an invasive change, and we'd have to make sure attributes are propagated everywhere correctly. So this option also comes with similar challenges. And, since we'll need virtual device on every node anyways, it makes sense to just add a field for it.

# Prior art
[prior-art]: #prior-art

Pytorch tensor expressions have a device field stored directly on them.

Additionally, checked_type_ is a first class field on Relay nodes, so there is precedent for storing information directly on nodes.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

The name of the virtual device is up for discussion. We've considered renaming SEScope to VirtualDevice, and it seems like that is the name we will eventually go with. If we go with a different name for SEScope, the name of virtual_device_ may be different.

I also need to confirm that it is feasible to propagate all spans and virtual devices through most passes. I will do this by picking a pass and making sure I can get it "span clean", meaning that it preserves all spans and propagates them correctly.

# Future possibilities
[future-possibilities]: #future-possibilities

I'll also need to change the machinery of the device planner after this field is introduced. The basic idea is that we won't have any on_device ops after device planning anymore, all the virtual device information will be stored directly in the virtual_device_ field. I won't go into much detail about this here for the sake of limiting the scope of this discussion. Also, we will probably introduce a 'lite' version of device planning that can "fill in the gaps" in device annotations that may be left by some passes.