- Feature Name: TUNIP: TVMScript Unified Printer
- Start Date: 05/25/2022
- RFC PR: [apache/tvm-rfcs#74](https://github.com/apache/tvm-rfcs/pull/74)
- GitHub Issue: [apache/tvm#11912](https://github.com/apache/tvm/issues/11912)
- Co-Authors: Lite Ye ([**@yelite**](https://github.com/yelite)), Greg Bonik
  ([**@gbonik**](https://github.com/gbonik)) Yong Wu
  ([**@yongwww**](https://github.com/yongwww)), Yuchen Jin
  ([**@YuchenJin**](https://github.com/YuchenJin))

# Summary
[summary]: #summary

This RFC proposes to modularize and infrastructuralize the existing TVMScript
printer, to develop unified printing mechanism across TVM stack, where TIR,
Relax and any future vendor-specific IR are all treated equally as dialects and
could be printed together without potential conflict in engineering.

# Motivation
[motivation]: #motivation

TVMScript, as a roundtrippable python-based text format, is
the central piece of TVM performance productivity. As the frontend of TVM, it
enables end users to directly construct the TVM IR, either TIR or Relax, in a
pragmatic approach. From Relax to MetaSchedule and TIR, TVMScript enables
inspectability and reproducibility at any level of compilation and
optimization. Furthermore, based on TVMScript, developers are empowered to
intercept, manipulate and customize the compiler behavior in a principled way.

While TVMScript is gaining traction and buy-in from the open source community,
the TVMScript printer suffers from multiple profound design issues:
- Not supporting IR fragment printing requires users to jump in-between
  TVMScript syntax and TIRText syntax 
- The lack of modularity leads to practical inability to scale up to and
  maintain multiple IRs without engineering conflicts 
- Enhancing co-existence of multi-level IRs often leads to re-engineering of
  existing features.

**Goal.** This RFC introduces Tvmscript UNIfied Printer (TUNIP), a systematic
redesign to
address those engineering, usability and scalability issues above. The goal of
this re-design includes:

**Goal 1 [Unified Representation].** Become the unified roundtrippable
representation of TIR and Relax, allowing systematic mixing of IRs or IR
fragments (Relax + TIR) in the same IRModule in the target language (for
example, python, C++). 

Currently TVMScript priner is designed specifically for TIR, and printing
multiple dialects together was not a design goal at that time. Therefore,
supporting Relax requires ad-hoc hack around the system (for
instance, [relax#149](https://github.com/tlc-pack/relax/pull/149) added support
of printing `T.cast` and `T.max` in an ad-hoc way, without reusing the printing
code for TIR). The unified printer in this RFC addresses this issue by having a
unified approach for printing IR tree to TVMScript. Engineers will be able to
implement a fully-fledged printer for Relax, TIR and any potential IR in the
future with minimal effort.

The folder structure that we want to pursue is:
```bash
include/tvm/script/printer/
└── ... # Public headers for the core infra
src/script/printer/
├── core # Core infra, which is IR-agnostic
│   ├── ir_docsifier.cc
│   └── ...
├── tir # TIR dialect 
│   ├── expr.cc
│   ├── stmt.cc
│   └── ...
└── relax # Hypothetical Relax dialect (not part of our RFC)
    └── ...
```

**Goal 2 [Third-Party IRs in Multi-Stage Compilation].** Modularize and
infrastructuralize the printer to support more future IRs or third-party IRs at
any level with maintainability, for example, IRs at lower-level than TIR, or
Relax VM executable.

The current TVMScript printer is tightly coupled with TIR by being a subclass
of TIR-specific functors
([link](https://github.com/apache/tvm/blob/main/src/printer/tvmscript_printer.cc#L129)).
This design isn’t scalable when we want to support more IRs. More importantly,
it’s impossible for the current approach to support third-party IR bteing
registered in a dynamic library.

**Goal 3 [Reproducibility and Error Reporting].** Expand reproducibility and
flexible rendering of diagnostic messages during any level of IR
transformation.

For example, the following snippet runs and produces an error.

```py
import tvm

@T.prim_func
def func_a(A: T.Buffer[(1,), "int32"]):
    A[0] = 0

@T.prim_func
def func_b(A: T.Buffer[(8,), "int32"]):
    A[0] = 0

tvm.ir.assert_structural_equal(func_a, func_b)
```

The current error message indicates what the difference was, but not where it
occurred.  This can sometimes be inferred from a stack trace, but becomes
increasingly difficult with larger IR graphs.

```
ValueError: StructuralEqual check failed, caused by lhs:
1
and rhs:
8
```

TUNIP should enable individual utilities and IR passes to have error messages
directing the user to exact locations in the IR representation.

```
ValueError: StructuralEqual check failed, first delta highlighted below

@T.prim_func
def func_a(A: T.Buffer[(1,), "int32"]) -> None:
                       ^^^^
    A[0] = 0

@T.prim_func
def func_b(A: T.Buffer[(8,), "int32"]) -> None:
                       ^^^^
    A[0] = 0
```


# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

This section introduces the design philosophy of the printer, and demonstrates
the proposed user-facing APIs where users means IR developers.

## Two-Stage Translation

Traditionally in TVM stack, printing is a single-stage process. The printer
assumes certain syntax of the target language, and therefore, so far there are
3 different printers all for TIR: ReprPrinter, TIRTextPrinter,
TVMScriptPrinter.

We extend the idea of the existing Doc class at
[src/printer/doc.h#L67](https://github.com/apache/tvm/blob/main/src/printer/doc.h#L67)
to allow better consistency and scalability. An IR, which could
be TIR, Relax or any other ones developed by third-party vendors, is first
translated to an intermediate Doc node tree, and then the Doc tree is mapped to a target
language, for example, Python, C++ IRBuilder API, or Rust.

**Stage 1 [TVM IR => Doc]**. On the first stage, the printer needs to take care
of translating a TVM IR to Doc tree. As an example, `tir.For` is translated to
`ForDoc` without having to worry about the underlying language. Note that some
complicated nodes in TVM IR, for example, `PrimFunc`, could be translated to
multiple IR elements, including `FunctionDoc` and a few `StmtDoc`.

During the translation from IR to Doc tree, it is possible that some statement
influences the syntax of its children or vice verse, especially for syntactic
sugars and declaring undefined variables in IR fragment printing. Therefore, a
generic data structure `Frame` is introduced to allow retrieval and
manipulation the relevant context information.

**Stage 2. [Doc => target language]**. On the second stage, Doc tree is then
honestly translated to the target language in text format. For example, when
the target language is python, `ForDoc` is translated to python’s for loop
syntax:

```python
for ... in ...:
  ...
```

When the target language becomes python IRBuilder, `ForDoc` is translated to:

```cpp
with T.For(...):
  ...
```

For generality, the Doc tree is designed to select minimal elements that exist
in languages used in developing TVM. A full spec of the Doc could be found in
the next section.

## Distributed Registration

As a major engineering challenge for TVMScript to scale to multiple IRs, the
existing printing logic has to be engineered, maintained and re-engineered in a
single file, which has brought significant confusion for developing multi-level
IRs for TVM Unity.

Inspired by the pass infrastructure, as well as the ReprPrinter in TVM, we
propose to develop the infrastructure to enable distributed registration, and
further allows printer for different levels of IR to be registered in separate
translation units, and in the meantime keeps the capability to be mixed
together at various level, for example, Relax uses TIR expression in its
function bodies, and TIR calls back to Relax function.

## Diagnostics and Reproducibility

Existing error reporting mechanisms have not taken IR structure and
reproducibility into consideration. Usually it reports a single line error
message without providing necessary context of how the IR looks like during
compilation. For example, when comparing whether two TIRs are structurally
equivalent, the system may report:

```cpp
ValueError: StructuralEqual check failed, caused by lhs:
{slow_memory_3_var: buffer(slow_memory_3_buffer_var, 0x501bf80), fast_memory_2_var: buffer(fast_memory_2_buffer_var, 0x501bd80), placeholder_3: buffer(placeholder_5, 0x50138a0), placeholder_2: buffer(placeholder_4, 0x5012b60), T_subtract: buffer(T_subtract_1, 0x5014390)}
and rhs:
{}
```

which lacks necessary information for users to understand where the mismatch
is.

As a recent effort, structural error reporting in TIR scheduling provides
relevant and reproducible context, as demonstrated below:

```cpp
@tvm.script.ir_module
class Module:
    @tir.prim_func
    def main(a: tir.handle, b: tir.handle) -> None:
        A = tir.match_buffer(a, [128, 128, 128, 128], dtype="float32")
        B = tir.match_buffer(b, [128, 128, 128, 128], dtype="float32")
        # body
        # with tir.block("root")
        for i, j, k, l in tir.grid(128, 128, 128, 8):
            tir.Block#0
            with tir.block("B"):
            ^^^^^^^^^^^^^^^^^^^^
                vi, vj, vk = tir.axis.remap("SSS", [i, j, k])
                vl = tir.axis.spatial(128, l * 16)
                tir.reads([A[vi, vj, vk, vl]])
                tir.writes([B[vi, vj, vk, vl]])
                B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * tir.float32(2)

Error: ...
```

However, the underlying mechanism supports only S-TIR and error reporting on
`tir.ForNode` and `tir.BlockNode`, and is less extensible for generic cases.

To generalize this UX across the TVM stack, during the first stage in
translation, the following steps is additionally executed:

- Each Doc node is optionally attached to a node in TVM IR
- After the 1st stage is finished, collect all IR nodes that gets attached to
  Doc into a map, whose key is IR node and value is a list of Doc nodes.
- For each IR node that has diagnostic message, trace back through its parent
  until it reaches to an IR node in the map collected in previous step. Then it
  can produce a map from Doc node to diagnostic message.
- In the 2nd stage, diagnostic message will be printed as doc is being printed
  into target language

# Reference-level explanation

## Doc Spec

The design of the Doc is to have a unified representation of TVMScript in
different languages. The overall structure is simplied from Python ast, and
their meaning is straightforward.

```py
Doc(Optional<ObjectRef> source) # Base class for doc

# Expression
ExprDoc() # Base class for expression
LiteralDoc(Union[IntImm, FloatImm, String, nullptr_t] value) 
IdDoc(String name)
AttrAccessDoc(ExprDoc value, String attr)
IndexDoc(ExprDoc value, Array<Union<ExprDoc, SliceDoc>> indices) 
CallDoc(ExprDoc callee, Array<ExprDoc> args, Array<String> kwargs_keys, Array<ExprDoc> kwargs_values)
OperationDoc(OperationKind kind, Array<ExprDoc> operands)
LambdaDoc(Array<IdDoc> args, ExprDoc body)
TupleDoc(Array<ExprDoc> elements)
ListDoc(Array<ExprDoc> elements)
DictDoc(Array<ExprDoc> keys, Array<ExprDoc> values)

# Statements
StmtDoc(Array<String> comments) # Base class
AssignDoc(ExprDoc lhs, Optional<ExprDoc> rhs, Optional<ExprDoc> annotation)
IfDoc(ExprDoc predicate, Array<StmtDoc> then_branch, Array<StmtDoc> else_branch)
WhileDoc(ExprDoc predicate, Array<StmtDoc> body)
ForDoc(ExprDoc lhs, ExprDoc rhs, Array<StmtDoc> body)
ScopeDoc(Optional<ExprDoc> lhs, ExprDoc rhs, Array<StmtDoc> body)
ExprStmtDoc(ExprDoc expr)

# Special Docs
SliceDoc(Optional<ExprDoc> start, Optional<ExprDoc> stop)
FunctionDoc(IdDoc name, Array<AssignDoc> args, Array<ExprDoc> decorators, ExprDoc return_type, Array<StmtDoc> body))
ClassDoc(IdDoc name, Array<ExprDoc> decorators, Array<AssignDoc> aliases, Array<FunctionDoc> functions)
```

## IRDocsifier Spec

IRDocsifier is responsible for transforming IR node tree into Doc tree. Its API
looks like

```cpp
class IRDocsifierNode : public Object {
 public:
  // ir_prefix maintains a map from dispatch_token to ir prefix
  // so that the print function can construct an expression with
  // the current ir prefix, like `T.xxx` in TIR and `R.xxx` in Relax 
  Map<String, String> ir_prefix;
  // TranslationTable maintains a map from IR node to Doc
  // It will be updated when new variable gets into the scope, 
  // like when print PrimFunc or BlockRealize
  // It will be looked up when printing variable nodes like tir::Var and tir::Buffer
  TranslationTable translation_table;
  Array<Frame> frames;
  Array<String> dispatch_tokens;

  /*!
   * \brief Transform the input object into TDoc
   */
  template <class TDoc>
  TDoc AsDoc(const ObjectRef& obj);

  /*!
   * \brief Push a new dispatch token into the stack
   * \details The top dispatch token decides which dispatch table to use
   *          when printing Object. This method returns a RAII guard which
   *          pops the token when going out of the scope.
   */
  WithCtx WithDispatchToken(const String& token);

  /*!
   * \brief Push a new frame the stack
   * \details Frame contains the contextual information that's needed during printing,
   *          for example, variables in the scope. This method returns a RAII guard which
   *          pops the frame and call the cleanup method of frame when going out of the scope.
   */
  WithCtx WithFrame(const Frame& frame);

  /*!
   * \brief Get the top frame with type FrameType
   */
  template <typename FrameType>
  Optional<FrameType> GetFrame() const;
}
```

To register print function to the `IRDocsifier`, one should use the
`TVM_STATIC_IR_FUNCTOR` macro and the `set_dispatch` method of the
`ObjectFunctor`

- Registration of printing methods for IR nodes

```cpp
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<PrimType>("tir", [](PrimType ty, IRDocsifier p) -> Doc {
      using runtime::DLDataType2String;
      return TIR(p)->Attr(DLDataType2String(ty->dtype));
    });

// Explanation:
// 1. Here we register the print function of the PrimType node in TIR
// 2. The first arg to the `set_dispatch` function is the dispatch token
//    It's optional and represents the name of IR
// 3. The first argument to the print function is the node to be printed
// 4. The second argument is instance of `IRDocsifier`, which can be used
//    to recursively translate the child nodes.
// 5. The print method returns a subclass of Doc

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<Range>([](Range e, IRDocsifier p) {
  return SliceDoc(p->AsExprDoc(e->min), p->AsExprDoc(e->min + e->extent));
});

// The first arg to the `set_dispatch` can be omitted, and 
// the print function will be registered the default layer.
// It will be called by default and can be overriden by registering
// another print function under an IR name. 

// This function will be called instead of the previous one, 
// if Printer is printing relax.
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<Range>("relax", [](Range e, IRDocsifier p) {
  ...
});
```

- Dispatch

```cpp
auto tir_dispatch_ctx = ir_docsifier->WithDispatchToken("tir");
Doc doc = ir_docsifier->AsDoc<Doc>(node);

// Here we setup the ir_docsifier to call print functions under 
// the 'tir' dispatch token, and then call the AsDoc method to 
// translate `node`, as an ObjectRef, into `Doc`, by using the 
// print functions registered in the dispatch table.

template <class TDoc>
TDoc AsDoc(const ObjectRef& obj) const {
  return Downcast<TDoc>(AsDocImpl(obj));
}
```

## Frame Spec

Frame provides the contextual information during printing. Most commonly, frame
contains variable defined in the current scope (like tir function, tir block,
tir loop). A subclass of Frame can be created to store more specific
information. For instance, `tir::ForLoopFrame` should contain the information about
the TIR for loop in order to print iter var remapping when printing
BlockRealize.

```cpp
class FrameNode : public Object {
 public:
  Array<ObjectRef> objs;
  TranslationTableNode* translation_table;

  /*!
   * \brief Set the name of a variable IR node
   */
	virtual IdDoc DefByName(const ObjectRef& obj, const String& name);
  /*!
   * \brief Set the doc of a variable IR node
   * \details This is useful when the variable is implicitly defined in the TVMScript.
   *          For example, when defining a `tir::Buffer buf`, buf->data is also a tir::Var,
   *          which should be printed as `buf.data`, rather than an identifier
   *          in the TVMScript.
   */
  virtual ExprDoc DefByDoc(const ObjectRef& obj, const ExprDoc& doc);
}
```

## Upgrade Plan

`IRModule.script()` is the current way to print TIR into TVMScript. It calls
the `script.AsTVMScript` function registered at
`scr/printer/tvmscript_printer.cc`. We plan to split the whole upgrading process 
into 5 steps.

1. Without breaking change to existing functionality, upstream system
   components piece by piece with small PRs under a tracking issue.
   This new system mainly locates in `src/script`, which does not affect
   the functionality of the existing TVMScript printer.
2. Expose the unified printer as a global TVM function `script.printer.Script`, which is parallel
to the existing printer.
3. Add a boolean flag `use_legacy_printer` to the Python `IRModule.script`,
   which defaults to True. `IRModule.script` calls `script.printer.Print` if `use_legacy_printer`
   is explicitly turned off.
4. After stabilizing the new infra, change the default value `use_legacy_printer` to `True`.
5. Finally, deprecate the `use_legacy_printer` flag and clean up legacy code. 

# Drawbacks
[drawbacks]: #drawbacks

N/A

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

Compared to the existing way of printing TVMScript in single stage, introducing
two-stage printing will certainly increase the amount of code that needs to be
written. However, we believe two-stage printing is the right choice because it
reduces the complexity in the printing logic of each IR dialect by removing
unneccessary details about the target language syntax and string operations.
Therefore, it's more scalable if we want to support printing multiple kinds of
IR (TIR, Relax, and any potential third-party IRs in the future).

For example, printing buffer region (like `A[1:10, 2]`) in the current printer looks like

```cpp
Doc TVMScriptPrinter::PrintBufferRegion(const BufferRegionNode* op) {
  Doc doc;
  if (op->region.size() == 0) {
    doc << Print(op->buffer) << "[()]";
  } else {
    doc << Print(op->buffer) << "[";
    for (size_t i = 0; i < op->region.size(); ++i) {
      if (i != 0) doc << ", ";
      const auto& range = op->region[i];
      if (!is_one(range->extent)) {
        doc << Print(range->min) << " : " << Print(ana_.Simplify(range->min + range->extent));
      } else {
        doc << Print(range->min);
      }
    }
    doc << "]";
  }
  return doc;
}
```

while in the unified printer with two-stage printing

```cpp
ExprDoc PrintBufferRegion(tir::BufferRegion buffer_region, IRDocsifier p) {
  Array<Doc> indices;

  for (const Range& range : buffer_region->region) {
    if (tir::is_one(range->extent)) {
      indices.push_back(p->AsExprDoc(range->min));
    } else {
      indices.push_back(p->AsExprDoc(range));
    }
  }

  return p->AsExprDoc(buffer_region->buffer)->Index(indices);
}
```

The latter one is much simpler because it's free from the noisy code on how to
print the script in valid index syntax in Python. 

Assume the printer needs to support `k` IRs, and it takes `m` time to develop
the logic around IR semantics and `n` time to develop the logic around target
language syntax. It will take `k*(m+n)` time if we use single-stage printing
and `km + n` time if we adopt two-stage printing. We believe the cost of
extending the Doc class will be paid off as soon as `k` is larger than one,
based on our PoC on using two-stage printing for TIR.

Additionally, with two-stage printing we can change the output language from
Python to other languages easily. Although we will still focus on TVMScript in
Python in the foreseeable future, having such flexibilty is a nice additional
benefit.

# Prior art
[prior-art]: #prior-art

RFC for TVMScript: https://discuss.tvm.apache.org/t/rfc-hybrid-script-support-for-tir/7516

# Unresolved questions
[unresolved-questions]: #unresolved-questions

N/A

# Future possibilities
[future-possibilities]: #future-possibilities

With the unified TVMScript printer, we have one of the building blocks towards
a more open architecture, where the community can author their own IR and plug
into the TVM stack, interacting with other components and layers. 

As a mirror of this RFC, we will send out another RFC on the unified TVMScript parser,
to support parsing TVMScript into different kinds of IR.
