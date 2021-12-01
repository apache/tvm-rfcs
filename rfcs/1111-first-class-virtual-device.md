- Feature Name: Add virtual device information as a first class field on Relay expressions.
- Start Date: 2021-11-20
- RFC PR: [apache/tvm-rfcs#1111](https://github.com/apache/tvm-rfcs/pull/1111)
- GitHub Issue: [apache/tvm#1111](https://github.com/apache/tvm/issues/1111)

# Summary
[summary]: #summary

I propose adding a new field to Relay expressions, virtual_device_. This field will contain virtual device information [currently called SEScope].

# Motivation
[motivation]: #motivation

 Currently, the virtual device information (called SEScope today, but we will rename it soon) is stored in on_device Relay ops. This op is a wrapper op that contains the virtual device information for an expression. This pattern method was used to encode device information in Relay before @mbs-octoml's recent work on device planning. In his work, he continued to use on_device to encode the virtual device information.

At the time, we considered adding virtual_device_ as a first-class field in Relay expressions. However, we decided not to purse that method because not all fields of expressions are preserved during passes, so it would be a lot of work ensure that a new field is propagated correctly. Additionally, we weren't sure how fundamental the virtual devices would be.

Now, we believe that virtual devices are very fundamental. Pretty much every expression in Relay needs to have one. Using on_device to store them is clunky. Normal visitors that don't care about virtual devices need to peek inside on_device ops. Additionally, to change the virtual device of an expression, we have to backtrack to the last on_device and then rewrite it. First class virtual devices eliminate a lot of this complexity, and allow us to implement more features in device and memory planning in the future.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Users be able to introduce new information about the virtual device through the on_device op. This behavior is the same as it was before.

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

The main drawback to this implementation is that most passes in TVM do not propagate all fields when they visit expressions. You can see this today with spans-- most visitors do not preserve spans. When we introduce the virtual device field, we will need to ensure that they are propagated correctly throughout the Relay program. To do this, I introduced WithFields, a COW constructor that copies extra fields, including spans. I will extend WithFields to also copy virtual devices. The main challenge will be making sure the WithFields methods are used everywhere. In the long run, though, I think this is a positive change. It comes with the side effect that spans will also be propagated throughout Relay more completely. 

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

The current design, which propagates virtual device information through on_device ops is not very robust.

Another option we considered was adding the virtual device information as an attribute to each node instead of adding it as a first-class field on each node. However, not all Relay nodes have attributes, so we would have to add attributes for all the Relay nodes as well, which is also an invasive change. Additionally, we believe that the virtual device information is important enough to add it as a first class field. 

# Prior art
[prior-art]: #prior-art

checked_type_ is a first class field on Relay nodes, so there is precedent for storing information directly on nodes.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

The name of the virtual device is up for discussion. We've considered renaming SEScope to VirtualDevice, and it seems like that is the name we will eventually go with. If we go with a different name for SEScope, the name of virtual_device_ may be different.
I also need to confirm that it is feasible to propagate all spans and virtual devices through most passes. I will do this by picking a pass and making sure I can get it "span clean", meaning that it preserves all spans and propagates them correctly.

# Future possibilities
[future-possibilities]: #future-possibilities

I'll also need to change the machinery of the device planner after this field is introduced. The basic idea is that we won't have any on_device ops after device planning anymore, all the virtual device information will be stored directly in the virtual_device_ field. I won't go into much detail about this here for the sake of limiting the scope of this discussion. 
