- Feature Name: TIR Language Specification
- Start Date: May 31, 2023
- RFC PR: [apache/tvm-rfcs#0101](https://github.com/apache/tvm-rfcs/pull/101)
- GitHub Issue: N/A

# Summary

This RFC proposes including a language specification for TensorIR (TIR) in the TVM documentation. TIR tests would be expected to conform to the language specification; instances of compiled code behaving contrarily to the specification would constitute bugs. The specification would be valuable as a learning resource for new users and guidance for developers.

# Motivation

TensorIR is TVM’s lower-level intermediate representation for operations on tensors, allowing users to specify arithmetic operations on values that can be loaded from and stored into buffers. Many of TVM’s most valuable features for improving performance rely on TIR: in particular, scheduling transformations are defined on TIR programs and, similarly, TVM’s autotuning functionality (e.g., MetaScheduler) also operates on TIR. Additionally, TIR is the simplest way to define a new tensor operator in TVM, which is often a necessary step for supporting the latest deep learning models.

Yet, despite the importance of TIR for the language, there is relatively little documentation on the language itself, which makes it difficult for new users to pick up the language. Some existing resources include the recent ASPLOS paper on TIR by Feng _et al._ and Tianqi Chen’s Machine Learing Compilation course (see [Prior Art](#prior-art)). While these sources are valuable and provide an overview of TIR’s usage and utility, they do not (and are not intended to) provide a comprehensive review of the language and its features. The lack of a full language reference can be frustrating for new users, expert users, and compiler implementers alike:

* New users may not fully understand the behavior of constructs that they encounter in example code, creating an impediment to their understanding of the language.
* Expert users may encounter edge cases where the intended functionality is unclear. This can lead to unease as to whether a program’s behavior in such a situation is a “hack” that relies on compiler implementation details or indeed intended behavior.
* Compiler maintainers may encounter situations where it is unclear whether implementing a pass or making some change could alter the behavior a user would expect. Similarly, it may not always be clear whether some surprising behavior is a “bug” or, in fact, intended. A clear specification would eliminate the ambiguity in such situations.

The goal of writing a language specification would be to address these problems by writing a document that achieves the following goal: **Someone who is not otherwise familiar with TIR can refer only to the specification and be able to describe what any given TIR program is intended to do (without running it)**, assuming all validity conditions are met (something that will be discussed below). That is, with a language specification, it should be unambiguous what any given TIR program is intended to do (or, otherwise, whether the program is doing something that should be considered unsupported).

Additionally, by creating a single document that describes the core language mechanics, the specification would have the further benefit of making it simpler to propose revisions to the language design, since they could be described in terms of the specification and what revisisions would need to be made to it. The implications of a change to the language on different language features would be easier to enumerate, which could highlight from an early stage any difficulties that might arise in implementing such a change.

# Guide-Level Explanation

The draft specification is [included in full as part of this RFC](assets/0101/spec.md) and is meant to adopted as part of TVM's documentation. The document is written as a technical reference, intended to describe the behavior of each language construct in a precise manner. By contrast, the document is not meant as a tutorial, so it does not generally describe very much of the intent behind the design. A language specification may the right document for describing that, though if there are some places where a brief word on intent would be helpful, that could be considered for the sake of readability.

Additionally, as the specification itself emphasizes, the specification is concerned only with the _functional behavior_ of TIR programs. That is, it describes only the _visible behavior_ that results from executing a program. The main implication is that the specification is not intended to make guarantees about performance and which constructs are likely to result in better performance. The reasons for this approach are twofold: First, it is difficult to make any a priori guarantees about performance (most of the time, we use heuristics) and, second, this simplifies the specification and thus gives the compiler implementation more freedom to make changes related to performance.

# Reference-Level Explanation

The text of the draft specification, given in [`spec.md`](assets/0101/spec.md), should be understood to be the reference-level explanation of this RFC and should be reviewed accordingly.

The policy proposals in this RFC are as follows:
1. The text of the specification, once fully settled in the RFC discussion, should be added into the TVM documentation.
2. The specification is to be taken as authoritative on the intended visible effects of running a TIR program, unless the TIR program contains constructs that the specification explicitly chooses not to specify. If there is no reasonable interpretation of the specification for a given TIR program, that should be considered a flaw in the specification and prompt a proposal to revise the specification. If the observed behavior of the TIR program once actually run contradicts the specification, this should be understood to be a compiler bug (unless the community decides that the relevant portions of the specification are mistaken).
3. Any substantial change to TIR's design that affects the visible semantics must also describe how the specification is to be updated as a result of the changes; the implementation should include updates to the specification.
4. Any substantial change to the TIR specification must go through the RFC process (this includes those that accompany revisions to TIR in item 3; presumably, any substantial revision to TIR would already be proposed via an RFC).

## Design Choices

The draft as written it makes the decision to specify only a subset of programs that are possible to write in TIR, designating certain behaviors (some of which may be in use) as deliberately unspecified. This addresses an ambiguous aspect of TIR’s design: In particular, TIR’s AST exposes many things that are typically thought of as compiler internals—even though they are “compiler internals” in some sense, however, expert users can and do manually set them to take advantage of properties of the compiler implementation or of specific devices. For simplicity (see [Rationale and Alternatives](#rationale-and-alternatives) for further discussion), this specification avoids many of these expert techniques and instead aims to specify TIR at a high level of abstraction to avoid dealing with certain low-level details.

To summarize, TIR has historically made the distinction between a “front-end user” and a compiler implementer rather ambiguous (which is unusual among most programming languages). This specification instead aims to provide a distinction between “high-level” behavior that is guaranteed at TIR’s front-end versus lower-level details that are left to the compiler implementation.

### What Should the “Front End” Be?

As a working definition, this specification considers the following to be expected “front-end” users (as opposed to expert users who would be expected to make use of compiler internals):

* Users trying to straightforwardly implement a tensor operator to be optimized by the autoscheduler.
* Users of TE, which lowers into TIR.
* In the case of the ongoing Unity work, users who want to express tensor operators in TIR so they can be invoked directly in Relax.

### Behaviors That Are Not Supported in the Draft Specification

The draft specification abstracts over certain implementation details regarding how buffers are laid out in memory as well as hardware-specific details. The following are the additional restrictions the draft specification imposes on input programs:

* Buffers may not alias each other (except in the case of `match_buffers` for `Block` nodes, which is specifically addressed in the draft).
* Each allocation creates exactly one buffer (by contrast, lower levels of compilation may use a single allocation to represent multiple buffers by changing the strides and element offset appropriately).
* Users must not manually specify the strides or element offsets for buffers.
* No buffer allocated in the body of a `PrimFunc` may be exposed externally.
* The program may not contain device-specific builtins or builtins that engage in pointer manipulation.

TIR programs that violate these rules are considered to be unspecified, which means that the specification makes no guarantees about their semantics. Later stages of compilation may violate these assumptions, but they are meant to apply to input programs. (It is not desirable for the TIR compiler to entirely reject input programs that do these things, but such programs should be the domain of expert users who are familiar with the compiler implementation and are prepared to update their code if the implementation details change.)

# Drawbacks

There are some potential drawbacks to adopting a language specification:

1. Learning and understanding the specification can be an undertaking for existing or aspiring developers, particularly if some of the details prove to be tricky in practice.
2. The document itself is large and complex, which can become a burden if it needs to be revised to account for updates to TIR. In particular, this may slow down the process of implementing changes in TIR.
3. There is the potential for portions of the specification to fail to live up to the goal, such as by being factually wrong (both in terms of what the implementation really does and what the community intends and expects of TIR) or ambiguous.
4. There is the possibility for the specification to create distracting or tedious arguments over very minor points of the language that might never otherwise arise.
5. The great amount of effort required to write and maintain the specification would not be well spent if the specification is referenced only rarely or is frequently disregarded even when brought up.
6. The specification is concerned only with the _effects_ of code and not whether it's well-optimized (how quickly it gets those things done). This might frustrate readers who are looking for a guide on writing performant TIR code.

Points 2-4 would be relatively minor concerns provided that the specification turns out to be useful enough to justify the effort in maintaining it and hammering out ambiguities or contradictions with the implementation. Point 5 could be addressed by making an effort to make the specification easily visible and include references to it in appropriate contexts. Point 6 is inherent to writing a specification of this kind and this limitation should be clearly stated to set expectations. However, point 1 could potentially prove to be more serious, since the specification is useful only if community members read it and understand it well enough to be able to make revisions to it when necessary—this is why ensuring the community is aware of the specification and accepts its rationale is important.

# Rationale and Alternatives

The design choices described above were taken as a result of discussions with longtime TIR developers, who found that the implementation of some language features varied wildly across devices (especially regarding buffers: some devices feature multiple physical indices for memory or limited ability to dynamically allocate memory) and sometimes did not agree amongst themselves as to how certain constructs would behave in certain specific situations. To avoid dramatically increasing the complexity of the specification, the specification instead aims to take a higher level of abstraction in order to be simpler, more readable, and more stable over time. Additionally, starting with a high-level description leaves room for defining and describing lower-level invariants later.

A disadvantage of this choice is that it leaves some expert techniques unspecified. While it would be useful to document and describe these features used by expert users, it poses some problems for the specification: accounting for all these low-level implementation details across all devices TIR targets would greatly expand the scope of the specification and, more importantly, including this behavior in the specification would essentially “commit” the compiler maintainers to supporting it, which could restrict their ability to change other compiler implementation details. It would not be very helpful for the specification to be so tied to the compiler implementation that it essentially repeats what the implementation says; moreover, if changing small details of the implementation requires frequent revisions to the specification that break backwards compatibility, then the document is not very useful as a reference either.

# Prior Art

There is no existing specification for TIR and relatively little documentation on TIR from within the TVM project. However, there are other references for TIR that could be compared with the proposed specification:
1. [TensorIR: An Abstraction for Automatic Tensorized Program Optimization](https://arxiv.org/abs/2207.04296) (Feng _et al._, 2023). This is an ASPLOS publication detailing some of the optimization mechanisms implemented by TIR, going into detail on the block mechanism. While it explains some details of TIR's implementation, it is not a comprehensive language specification.
2. [The Machine Learning Compilation online course](https://mlc.ai/summer22/). This course includes some lessons illustrating how to impement tensor operators in TIR. It instructs students on basic constructs in TIR and how to use TIR to optimize their operator implementations. It is very helpful to have a resource oriented towards beginners, though it covers only a portion of TIR's features and does not discuss TIR at the same level of specificity as the proposed specification.

For related projects within the TVM community, we might consider the fact that Relay is described in formal detail in its [ArXiv paper](https://arxiv.org/abs/1904.08368) (which proved useful for discussion while it was being adopted as the front-end IR for TVM) and the [draft specification for Relax](https://github.com/apache/tvm/pull/14148) in the Unity project (full disclosure: That is also my project. There is also the difference that Relax's design is still in flux while TIR's is settled).

As prior art, we may also consider specifications for other languages. The [ISO C](https://www.iso-9899.info/wiki/The_Standard) and [C++](https://isocpp.org/std/the-standard) standards are massive documents that cover the details of those languages extremely comprehensively in order to address the many practical issues that arise in ensuring compatibility between multiple independent implementations of compilers that target many distinct architectures. Compare also the [Java language specification](https://docs.oracle.com/javase/specs/). By contrast, the [Python language reference](https://docs.python.org/3/reference/) is a much more compact document meant for a wider audience.

# Unresolved Questions

In the text of the specification, there are some unresolved questions:

- Vectorized loops: In the compiler implementation, vectorized loops are lowered into a sequence of vectorized operations using the `VectorizeLoop` pass, whose implementation is generally very complicated. It would be undesirable for the specification to simply describe such a large and complicated pass in text form, since it would likely also be more difficult to understand in that form compared to the source code. Instead, it would be preferable to have a simple description of the high-level intent of the vectorized loop. I am not certain that the current description in the text (describing them as having the same semantics as a parallel loop) is entirely accurate given how the loop vectorization pass really works. I am also not sure whether it should be part of the specification that the ordering of effects is preserved; perhaps the specification should not mention this fact (even though it does happen to be how `VectorizeLoop` behaves) and simply treat it exactly like a parallel loop.
- I am not entirely certain if this line about `Block` nodes can be reconciled with the principle about users not setting `elem_offset` or `strides` directly: "However, the buffers in `alloc_buffers` are not permitted to have unbound `Var`s in their `shape`, `stride`, or `elem_offset` fields, so `alloc_buffers` does not act as a binding site for those variables (any variables in those fields should already be bound)." How do users ensure that those buffers already have their fields set?
- The semantics of `BlockRealize` and `BufferRealize` would be good to review closely, as I was not entirely certain of how those worked. Additionally, it would be good to add some description of how some of the extra information is used by the compiler (even if it is not part of the semantics _per se_).
- There are likely many additional implicit validity conditions for the different `PrimExpr` and `Stmt` nodes that are not noted in the type-checking rules in the current draft. Any omissions would be good to note.
- Another tricky area in the current draft is dealing with varialbes that could be contained inside `Buffer` nodes. `elem_offset`, `strides`, and `shape` could all contain variables and constructs featuring `Buffer`s should be careful to specify when these are assigned or not.
- It would be good to include descriptions of the TE-specific nodes if they are similar enough to the other TIR constructs (or if they can be neatly mapped onto other TIR constructs). Perhaps that should be in a document related to TE instead.

Separately from these textual issues, the main issues relate to procedure around the specification and spreading word about the specification:
1. How do we define what constituates a "substantial" change to TIR or to the specfication? For example, adding a new datatype into TIR would require updating the specification by adding it to the list of datatypes and inserting it where relevant in the discussions of other datatypes, but it would likely not affect much else in the specification--should that still require a full RFC? What would distinguish a change requiring discussion from a simple "bugfix" that could simply be PR'd?
2. As a corollary, is there any process less formal than an RFC for updating the specification that might be suitable? An advantage of using an RFC for changes is that it would ensure visibility of changes and prevent the changes from being accepted without consensus, but a drawback is that this could delay the acceptance of changes.
3. Would including links or references to the specification in TIR header files increase its visibility? This would be easy to do, but it's doubtful that many people would see it there. There should certainly be references to the specification in visible places, but I'm not sure exactly which those are. We might consider TVM's `README.md` or makiing it a category in the docs website, but I think having some references to it from within the codebase would encourage contributors to look at the specification.

Additionally, in the [pre-RFC thread](https://discuss.tvm.apache.org/t/pre-rfc-language-specification-for-tir/14844), there was a suggestion that the specification could be used to guide the implementation of a reference interpreter, whose purpose would be to implement the specification in as simple a manner as possible (aiming for correctness over efficiency), which could then be used as an "executable specification" to compare against the compiler. We might consider whether a reference interpreter should be included as part of this RFC, though it comes with its own tradeoffs. Disadvantages include the fact that a reference interpreter would be more code to write and maintain in addition to the substantial TIR codebase and that the reference interpreter might impose choices on sections of specification that permit many interpretations (e.g., evaluation order for parallel loops), which might mislead users as to the intent of the specification. However, the advantages could be considerable: it would be a useful tool for debugging potential compiler bugs and having the reference interpreter would also ensure that the specification is, in fact, implementable. The inclusion of the reference interpreter (now or at a later point) would be a useful subject to establish in the RFC discussion.

# Future possibilities

If adopted, a high-level specification for TIR could serve as the first stage for potential other lines of work, like formalizing the notion of “dialects” in TIR that are used for different stages of compilation. Having specifications for each dialect would allow for clearly communicating the invariants expected at different stages of compilation, which would be helpful for future compiler development. (Such a project would, though, be of a larger scope and likely require reasoning about some of the low-level details that are elided in this draft.)

# Special Thanks

While I wrote most of the text of the draft specification, none of it would have been possible  without the efforts of Wuwei Lin (@vinx13), Eric Lunderberg (@Lunderberg), and Junru Shao (@junrushao). I am grateful to them for patiently answering my questions about the language and giving their input on many language features. Additionally, I thank Christian Convey (@cconvey), Denise Kutnick (@denise), and Prakalp Srivastava (@psrivas2) for many insightful comments during the process of writing the initial draft. Thanks is due also to the many community members who reviewed and commented on earlier drafts.
