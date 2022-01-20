- Feature Name: microTVM Roadmap
- Start Date: 2022-01-18
- RFC PR: [apache/tvm-rfcs#0053](https://github.com/apache/tvm-rfcs/pull/0053)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)
- Co-Authors: Denise Kutnick ([**@denise-k**](https://github.com/denise-k)), Andrew Reusch 
  ([**@areusch**](https://github.com/areusch)), Chris Sidebottom ([**@Mousius**](https://github.com/Mousius))

# [RFC] microTVM Roadmap

## Background and Context

**Roadmap Name:** microTVM Roadmap

**Roadmap Maintainers: `@areusch, @denise-k, @mousius`**

**Roadmap Description:** This roadmap tracks efforts within microTVM, as well as its feature dependencies within TVM (e.g. autotuning).

## Scope

**How are the tasks tracked in this roadmap grouped together? How can we think about this grouping distinct from those made in other roadmaps?**

Tasks in the microTVM Roadmap will generally be grouped together in a similar manner as previous engineering roadmaps used within the microTVM community ([2021](https://discuss.tvm.apache.org/t/tvm-microtvm-m2-roadmap/8821), [2020](https://discuss.tvm.apache.org/t/rfc-tvm-standalone-tvm-roadmap/6987)), and will be categorized using the themes introduced below. 

Large projects (e.g. AutoScheduler support for microTVM, microTVM backends/integrations, etc.) may be split into multiple tangible milestones, and smaller tasks (e.g. writing a schedule template) may be grouped together into a larger task.

**Is the proposed roadmap intended to represent a perpetually ongoing set of efforts, or is there an end goal which will close/finalize the roadmap?**

The microTVM Roadmap is intended to represent an ongoing set of efforts.

**Does the proposed roadmap have any scope overlaps with any existing roadmaps? If so, please list them.**

The microTVM Roadmap is intended to capture all efforts within microTVM and related dependencies. Any dependencies of microTVM will be captured in the appropriate roadmap, and linked into the microTVM Roadmap.

## Themes

**List 4-6 proposed “themes” of the roadmap, intended to convey the purpose of the roadmap and the types of tasks that should be added.**

- **For each theme, include a set of definitions specific to the proposed roadmap.**
- **For each theme, include a set of success criteria specific to the proposed roadmap.**

### Architecture

#### Definitions

One of microTVM's competitive advantages over other frameworks is that it's not owned by a singular ML framework nor hardware vendor. microTVM intends to support inference on devices from any hardware vendor so long as the workload can be modeled in TIR.

In order to enable this, microTVM uses common representations (e.g. Relay, TIR) and frameworks (e.g. Project API, C runtime, Embedded C API, RPC server) which allow everyone to participate and contribute to microTVM.

#### Success Criteria

- Maintain and enhance a common set of APIs and frameworks for microTVM users.
- Create and maintain architectural features to make it easier for hardware vendors to add new backends/integrations to microTVM.

### Documentation

#### Definitions

Robust documentation is a critical prerequisite of enabling users and developers of microTVM of all skill levels. It also facilitates the growth of the microTVM community and the maturity of the technology.

#### Success Criteria

- User and developer documentation for APIs and entry points (e.g. TVMC) into microTVM
- User documentation for each backend/enablement within microTVM
- End-to-end tutorials for both users and developers of microTVM

### Platform Enablement

#### Definitions

One of microTVM’s biggest goals is to enable as many unique hardware architectures as possible. Tasks categorized as Platform Enablement may either be milestones towards enabling an individual hardware target, or features to improve the enablement process across all of microTVM.

#### Success Criteria

- Ease of platform enablement via common APIs and frameworks.
- Support multiple CPU architectures within microTVM
- Support multiple firmware platforms within microTVM
- Demonstrate heterogeneous computing capabilities with microTVM.

### Packaging

#### Definitions

The embedded systems domain has many unique constraints around software packaging which should be addressed in microTVM.

#### Success Criteria

- Packaging for a small memory footprint
- Improve microTVM's dependency management 
- Easily installable packaging (e.g. pip)

### Performance

#### Definitions

microTVM aims to maximize performance in the embedded systems domain. To achieve this, much of microTVM’s performance infrastructure is inherited from TVM. For instance, microTVM inherits TVM’s ability to autotune computational graphs with schedule templates.

#### Success Criteria

- State of the art performance in the embedded systems domain.
- Continuously integration of new autotuning capabilities from TVM into microTVM.
- Robust schedule template availability across supported architectures.
