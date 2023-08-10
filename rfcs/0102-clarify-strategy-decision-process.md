Authors: @tqchen

- Feature Name: [Process RFC] Clarify Community Strategy Decision Process
- Start Date: 2023-08-03
- RFC PR: [apache/tvm-rfcs#0102](https://github.com/apache/tvm-rfcs/pull/0102)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

## Summary

Machine Learning Compilation (MLC) is an emerging field in fast development.
With the tremendous help from the whole community, it’s exciting to see that TVM delivers significant needs from and to
developers and thus has become widely popular in both academia and industry.

As the community pushes for different goals that help each other, naturally, there
are strategy decision points about overall directions and new modules adoptions.
These decisions are not fine-grained code-level changes but are important for a
community to be viable in the long term.
The process of bringing those changes is less clarified to the community, and hurdles can be high.
We have made attempts in the past to bring more verbose processes, but this has proven to be less successful.
One observation is that it is hard for broader volunteer developers and community members to follow complicated processes.
Additionally, different members can have different interpretations of how to do things,
leading to stagnation and lack of participation from volunteer members.

We are in a different world now in the case of ML/AI ecosystem, and it is critical for
the community to be able to make collective decisions together and empower the community.
Following the practices of existing ASF projects (e.g. hadoop), we propose to use a simple process for strategic decisions.

## Proposal: Strategy Decision Process

We propose the following clarification of the strategy decision process:
It takes lazy 2/3 majority (at least 3 votes and twice as many +1 votes as -1 votes)
of binding decisions to make the following strategic decisions in the TVM community:

- Adoption of a guidance-level community strategy to enable new directions or overall project evolution.
- Establishment of a new module in the project.
- Adoption of a new codebase: When the codebase for an existing, released product is to be replaced with an alternative codebase.
  If such a vote fails to gain approval, the existing code base will continue. This also covers the creation of new sub-projects within the project.

All these decisions are made after community conversations that get captured as part of the summary.
