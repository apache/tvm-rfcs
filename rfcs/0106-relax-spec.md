- Feature Name: Relax Language Specification
- Start Date: January 22, 2024
- RFC PR: [apache/tvm-rfcs#0106](https://github.com/apache/tvm-rfcs/pull/0106)
- GitHub Issue: N/A

# Summary

This RFC proposes a language specification for the Relax IR in the TVM documentation, similar to [the proposed TIR specification](https://github.com/apache/tvm-rfcs/pull/101). The specification would describe the intended behavior of language constructs and provide a reference for identifying bugs (encountered behavior contrary to what the specification describes would be a bug) and would thus serve as a resource for both new users and language maintainers.

# Motivation

Now that TVM's main branch includes the features that were proposed as part of TVM Unity (see, in part, [https://github.com/apache/tvm-rfcs/pull/89](#89)), there is particular reason to include documentation for the new capabilities. One of the additions in TVM Unity is the Relax IR, which was proposed as a successor to Relay that is better able to accommodate dynamically shaped tensors and has numerous features intended to facilitate the use of external subroutines.

Despite having been added into TVM's main branch, there is no top-level documentation on Relax at all other than those included inline as code comments, creating particular urgency to document something as complex and extensive as an entire language. Lacking a language specification is not only an issue for users of the language (who may not be certain of the intent of different constructs and have little recourse besides trial-and-error experimentation or looking at test cases) but also for developers, who otherwise would have no organized way to assess the impact a change to the language would have on other components.

A language specification would be useful as a language reference to all those who use it and would also serve as top-level guide to the language for maintainers, who would have a single, centralized resource for judging how changes they are considering to the language affect the previous intended behavior (leading to discussion on whether the specification should be updated). In this way, the specification would also make it easier to discuss changes to the language, since these changes can be articulated in terms of how the specification would have to change to accommodate them.

The specification is not meant to be a replacement for all other forms of documentation or shorter tutorials, though it can still be useful as a learning resource. Rather, it is meant to be an authoritative document on questions of language mechanics: what the specification states is what the language is intended to do. Ambiguities or omissions in the specification should be taken as an indication that the specification should be revised to account for them. In particular, the specification is intended to be a document that accounts for all the intended visible behavior of the language and (as in the TIR specification RFC), has the goal of allowing **anyone who is not otherwise familiar with Relax to use the document to explain precisely what a given Relax program is intended to do**. The specification, as a result, may be rather dense and technical, but it has the goal of being a comprehensive reference.

# Guide-level Explanation

This RFC includes [a draft of the Relax specification](assets/0106/spec.md). The draft specification is meant to be adopted as a part of TVM's documentation as this RFC. The document describes the intended behavior of each Relax construct, occasionally with some comments as to differences between Relay and Relax, and is the bulk of the proposal. As with the proposed TIR specification, the focus of this specification is functional behavior, meaning that it describes the visible effects of running Relax programs; the specification is generally not intended to make guarantees about how the language implementation accomplishes these tasks, partly to give more freedom to language implementations and to make the specification a high-level document that would not need to be updated for every small change to the compiler implementation.

The RFC, in particular, proposes to adopt the draft specification as an initial specification and proposes to adopt a policy stating that changes to Relax's front-end interfaces (such as changes to the AST and introduction of new language mechanics) to also update the specification. The pull request guidelines should make reference to the specification, reminding contributors that changes that affect Relax's core language features should update the specification. Additionally, language behavior that contradicts the specification should be defined as a bug and the bug report guidelines should make reference to the specification. (The bug could be in the specification rather than the implementation, so either the specification or the implementation should be corrected.)

# Reference-Level Explanation

In addition to the specification draft included in [`spec.md`](assets/0106/spec.md), this RFC recommends adopting the following policies:

* The contents of `spec.md` will be added to the online TVM documentation as a Relax language specification.
* The online Relax documentation should point to the language specification as the authoritative document describing the language semantics.
* The bug report and pull request templates should recommend referring to the specification if their subject matter deals with Relax. In particular, the guidelines should state in the case of bug reports that a language behavior is a compiler bug if the actual behavior differs from that described in the specification and that there is an issue with the specification itself if it is unclear what behavior the specification would recommend in a given circumstance. In the case of pull requests, the templates should state that changes that affect the semantics described in the specification should also update the specification.
* Reviewers of pull requests or RFCs that affect Relax's visible semantics should consider whether the changes affect the Relax specification and whether the pull request should therefore also update the Relax semantics.

# Drawbacks

While documentation is generally an asset, the specification can pose problems if it becomes out of date compared to the specification, Additionally, having to update the specification upon making major changes to the language also creates an additional burden on the community. I would contend that the benefits of having a specification outweigh these disadvantages, since the specification can serve to guide the development of Relax and allow for more informed discussion of changes to the language, while requiring the specification to be updated alongside changes to the language also encourages contributors to consider how their changes affect the overall design of the language.

# Rationale and Alternatives

Alternatives to having a specification would be to continue without an officially adopted specification or relying on, for example, a series of tutorials to explain the langauge semantics. These possibilities would still leave many aspects of the language ambiguous and would cause issues for development if, for example, one contributor makes a change to the language without considering interactions with other features. Having an explicit specification preempts misunderstandings that could otherwise result and provides more detail than less formal documents like tutorials would.

# Prior Art

There is no preexisting Relax specification or, in fact, any language documentation for Relax other than the comments in the codebase. As in the [TIR language specification RFC](https://github.com/apache/tvm-rfcs/pull/101), we can consider the [Relay ArXiv paper](https://arxiv.org/abs/1904.08368) to be a less formal specification for Relay and also compare specifications for other languages, like the [ISO C](https://www.iso-9899.info/wiki/The_Standard) and [C++ standards](https://isocpp.org/std/the-standard) or the [Java](https://docs.oracle.com/javase/specs/) and [Python](https://docs.python.org/3/reference/) language references.

The circumstances of Relax differ considerably from the other languages listed, as Relax is in many regards a work in progress and will likely gain new features as the TVM community considers more deep learning applications. Being a language that is in some aspects still being designed makes it all the more essential that the different language features be documented accurately in a manner that allows community members to assess the impact of possible changes.

# Unresolved Questions

The principal questions for this RFC are procedural ones, e.g., whether there should be a particular process for identifying the specification. Since Relax is changing frequently, I would not recommend requiring an RFC for any change to the specification, though particularly major changes to the language might merit an RFC and should be accompanied by updates to the specification regardless—hence, the proposal does not introduce any requirements for updating the specification beyond those already expected for a pull request. We may also consider whether there should be any members of the community tasked with ensuring updates to the specification do not introduce inconsistencies (I would volunteer for this role, but ideally all regular contributors to Relax should feel comfortable reading and potentially updating the specification).

# Future Possibilities

If the specification's adoption is successful, we may consider expanding the scope of the specification to also include the TVMScript parser for Relax, particularly as it becomes more stable. As with the proposed TIR specification, we could also consider having a reference interpreter akin to Relay's Python compiler that is a very straightforward implementation of the semantics if it might be helpful for debugging, though at present the Relax VM is the authoritative implementation and it does not sound like there is a compelling reason to introduce another method of execution.

# Special Thanks

I am grateful to all who have reviewed and commented on previous drafts of the Relax specification, which were posted as pull requests in the past, on the [TVM Unity branch](https://github.com/apache/tvm/pull/14148) and in the [TLC-Pack Relax repo](https://github.com/tlc-pack/relax/pull/273). Yuchen Jin, Prakalp Srivastava, Denise Kutnick, Junru Shao, Sunghyun Park, and Yong Wu offered much feedback on early drafts, including some predating the aforementioned PRs, for which I am especially grateful.