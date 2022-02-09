- Feature Name: TVM CI & Testing Roadmap
- Start Date: 2022-01-19
- RFC PR: [apache/tvm-rfcs#0054](https://github.com/apache/tvm-rfcs/pull/0054)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)
- Co-Authors: Denise Kutnick ([**@denise-k**](https://github.com/denise-k)), Andrew Reusch 
  ([**@areusch**](https://github.com/areusch)), Leandro Nunes ([**@leandron**](https://github.com/leandron))

# [RFC] TVM Continuous Integration & Testing Roadmap

## Background and Context

**Roadmap Name:** TVM Continuous Integration & Testing Roadmap

**Roadmap Maintainers: `@areusch, @denise-k, @leandron` (add more here)**

**Roadmap Description:** This roadmap tracks all efforts related to Continuous Integration and Testing within TVM.

## Scope

**How are the tasks tracked in this roadmap grouped together? How can we think about this grouping distinct from those made in other roadmaps?**

Tasks in this roadmap may be grouped together using **vertical** or **horizontal** tasks.
* **Horizontal Tasks:** CI/Testing tasks which have effects on all of TVM are considered horizontally-grouped tasks. For instance, if a task existed to [parameterize all of TVM's unit tests](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0007-parametrized-unit-tests.md), this would be considered a **horizontal task**.
* **Vertical Tasks:** CI/Testing tasks which are concentrated within a single subproject of TVM are considered vertically-grouped tasks. For instance, adding [Vulkan unit tests to TVM's CI](https://github.com/apache/tvm/pull/9093) is considered a **vertical task**. Multiple vertical tasks in the same domain may be grouped together into a larger roadmap item as needed. For example, the aforementioned Vulkan example could be grouped into a roadmap item tracking all Vulkan-related CI patches.

**Is the proposed roadmap intended to represent a perpetually ongoing set of efforts, or is there an end goal which will close/finalize the roadmap?**

The CI & Testing roadmap represents a perpetually ongoing set of efforts within TVM.

**Does the proposed roadmap have any scope overlaps with any existing roadmaps? If so, please list them.**

This roadmap is intended to provide a unified view of all CI and Testing efforts horizontally across TVM. Other roadmaps may vertically track certain areas of TVM (e.g. MicroTVM) and may share CI/Testing tasks with this roadmap.

## Themes

**List 4-6 proposed “themes” of the roadmap, intended to convey the purpose of the roadmap and the types of tasks that should be added.**

- **For each theme, include a set of definitions specific to the proposed roadmap.**
- **For each theme, include a set of success criteria specific to the proposed roadmap.**

### Coverage

#### Definition

Test coverage can be defined in multiple ways:

- Coverage of the matrix of end-to-end flows on TVM (from all different types of frontends to all different types of hardware)
- Operator Coverage (for all the different parameters that can be configured for an operator)
- API coverage (for user-facing APIs, internal APIs, and FFI)

#### Goals and Success Criteria

Maximize coverage such that any issues introduced to the code are caught preemptively, rather than having users encounter and report bugs that could've been caught with proper test coverage.

### Ease of Use

#### Definition

Ease of use applies to both local testing and the CI environment.

#### Goals and Success Criteria

- Being able to easily run tests on a local machine
- Being able to replicate the behavior/dependencies of CI on a local machine
- Being able to maintain and update dependencies on CI machines in a sustainable way

### Latency

#### Definition

CI latency is the combination of build time, run time and wait time for individual CI jobs. This includes:

- How long one must wait for CI jobs to start
- How long one must wait for TVM and various dependencies to build
- How long one must wait for nodes to become available
- How long one must wait to get test results once a CI job has started

#### Goals & Success Criteria

The goal of CI latency work is to continuously reduce the overall latency, and to make sure that latency stays low as new features and tests are added.

The success of CI latency work can be measured by % reductions in such latency metrics over time.

### Reporting

#### Definition

CI reporting can be defined in two parts:

- **Infrastructure**: any software, services, or hardware used to generate reporting artifacts
- **Artifacts**: any raw data, metrics, and charts generated as an output from CI runs

#### Goals & Success Criteria

Build up reporting infrastructure to generate useful artifacts for TVM users and developers.

### Stability

#### Definition

CI stability is defined by the overall availability and uptime of CI machines.

#### Goals & Success Criteria

- 100% uptime on baseline CI nodes
- Fallback mechanism for CI node failure
- Documented postmortem protocol for CI node failure

### Security

#### Definition

Security in the context of TVM CI & Testing can be defined in two different ways:

- **TVM Security**: This is the security of TVM's codebase as a whole.
- **CI Security**: This is the security of TVM's CI pipelines and machines.

This roadmap will cover **CI Security**. **TVM Security** will be covered separately in another roadmap.

#### Goals & Success Criteria

- System-level security (e.g. operating system updates) should be regularly maintained on TVM's CI & Build machines, ensuring CI Security.
- TVM's external dependencies should be updated on CI & Build machines in a timely manner, so that any security patches within those dependencies are installed, ensuring CI Security.
- TVM's CI & Testing pipelines should automatically monitor the TVM codebase and alert users if any security issues are found.
