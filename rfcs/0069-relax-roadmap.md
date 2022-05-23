- Feature Name: Relay Next Roadmap
- Start Date: 2022-05-06
- RFC PR: [apache/tvm-rfcs#0053](https://github.com/apache/tvm-rfcs/pull/0069)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)
- Co-Authors: Denise Kutnick ([**@denise-k**](https://github.com/denise-k)), Yuchen Jin
  ([**@YuchenJin**](https://github.com/YuchenJin))

# [RFC] Relay Next Roadmap

# Background and Context

**Roadmap Name: Relay Next (Relax) Roadmap**

**Roadmap Maintainers: `denise-k, hypercubestart, MasterJH5574, YuchenJin, ZihengJiang`**

**Roadmap Description:** This roadmap is intended to track the efforts of the Relay Next (Relax) project, which is an exploratory evolution of Relay IR to expand expressibility, performance, and portability for modern machine learning workloads. **This roadmap will specifically focus on RFCs, Tracking Issues and PRs relevant to the integration of Relax into TVM.** For more details on Relax, please see the discuss forum post [here](https://discuss.tvm.apache.org/t/relax-co-designing-high-level-abstraction-towards-tvm-unity/12496/4), the TVM Community Meeting from 04/20/2022 [here](https://www.youtube.com/watch?v=2aYWGOYmDFY), the TVMCon 2021 talk [here](https://www.youtube.com/watch?v=xVbkjJDMexo), and the project repository [here](https://github.com/tlc-pack/relax).

# Scope

**How are the tasks tracked in this roadmap grouped together? How can we think about this grouping distinct from those made in other roadmaps?**

Tasks in this roadmap are intended to track the Relax project. The roadmap items within the Relax project may fall into one of several categories:

- **Technical RFCs:** The Relax community has had extensive design discussions and built consensus on a set of core design principles for the Relax project. As the project evolves, more of these design discussions will be surfaced to the TVM community, and relevant design discussions should be posted as **Technical RFCs** to TVM's [discuss forum](https://discuss.tvm.apache.org) and [RFC repo](https://github.com/apache/tvm-rfcs), then subsequently listed on the Relax roadmap to provide visibility to the broader TVM community so that they can provide their inputs.
- **Tracking Issues:** Relax aims to evolve Relay IR to enable core capabilities such as cross-IR interactions in TVM, first-class dynamic shape support and optimization, and customizable compilation pipelines. As these efforts reach maturity, the Relax community intends to contribute these capabilities back to TVM. As such, after the relevant **Technical RFCs** are accepted by the TVM community, the Relax roadmap maintainers will create **Tracking Issues** relevant to each feature being integrated into TVM mainline.
- **Outcome-Based:** Relax development follows a ‘vertical’ approach, in which certain outcomes (e.g. end-to-end performance parity with Relay on a static ResNet50 model) are used as initial proofpoints for the TVM community to build confidence in upstreaming and integrating Relax to TVM mainline.

**Is the proposed roadmap intended to represent a perpetually ongoing set of efforts, or is there an end goal which will close/finalize the roadmap?**

Relax is currently an exploratory project in the early phases of development. This proposed roadmap is intended to track the Relax project as it pertains to TVM. As the project matures, the Relax community will continue to consult and build consensus with the broader TVM community on the future integration of Relax into TVM. After the Relax project is fully integrated into TVM, the TVM community may then decide to finalize the Relax roadmap and track Relax efforts in the TVM Graph Computations and High-Level Optimizations roadmap.

**Does the proposed roadmap have any scope overlaps with any existing roadmaps? If so, please list them.**

The proposed roadmap is most closely related to the TVM Graph Computations and High-Level Optimizations roadmap, which tracks ongoing work related to Relay in TVM mainline. However, the scope remains separate since Relax will initially have many more tasks related to upstreaming and integration back into TVM mainline.

# Themes

**List 4-6 proposed “themes” of the roadmap, intended to convey the purpose of the roadmap and the types of tasks that should be added.**

- **For each theme, include a set of definitions specific to the proposed roadmap.**
- **For each theme, include a set of success criteria specific to the proposed roadmap.**

## Expressibility

### Definitions

Expressibility in Relax is the ability for **users of Relax,** whether directly writing Relax IR or indirectly compiling to Relax, to express their desired workloads using Relax. Relax prioritizes expressibility at multiple levels of granularity:

- **Type expressibility:** defined as the expressibility of any types, including standard tensor types with data types such as int8, fp16, fp32, etc., but also non-standard types such as general object type that corresponds to tvm runtime objects such as array and string.
- **Shape expressibility:** defined as the expressibility of any shape, including static shapes, dynamic input size (with symbolic shapes wherever known), and dynamic ranks.
- **Operator expressibility:** defined as the expressibility of ML operators in Relax.
- **End-To-End expressibility:** defined as the expressibility of ML workloads in Relax.

### Success Criteria

Each type of expressibility is measured for success in a different way:

- **Type expressibility:** should be at parity compared with Relay IR.
- **Shape expressibility:** dynamic shapes should be easy to represent and should be inferred by Relax IR.
- **Operator expressibility**: should be comparable to best-in-class ML frameworks (e.g. PyTorch), with support for custom operator expressibility via TVM Script.
- **End-to-End expressibility:** should be at parity compared with Relay IR.

## Extensibility

### Definitions

Extensibility in Relax is the ability for **developers of Relax** to add new functionalities to Relax. Extensibility has a couple of key forms:

- **Core Infrastructure extensibility:** defined as the ability to extend and evolve Relax’s core infrastructure over time. Some examples of core infrastructure extensibility including symbolic shape support and ease of adding new operators through TE and TIR.
- **Model extensibility:** Having the abiility to cover common models of interest, initially, we will reuse the Relay’s infrastructure when necessary through a Relay ➡️ Relax converter. As a next step, we will also explore directly ingesting emerging dynamic models.
- **Functional extensibility**: defined as the ability to extend Relax’s functionality by utilizing Relax’s existing core infrastructure. Some examples of functional extensibility include adding custom operators and adding optimization passes.

### Success Criteria

- At minimum, the Relax core infrastructure should be as extensible as Relay. This is one of the core design principles of Relax.
- Functional extensibility should be straightforward and well-documented to empower TVM developers to add new functionality to Relax.

## Interoperability

### Definitions

Relax, as a graph-level IR within TVM, must interact with other IRs within TVM as well as tools and libraries outside of TVM. This is known as interoperability.

TVM Unity emphasizes interoperability in its design principle of cross-abstraction communication. Therefore, Relax is held to an even higher standard of interoperability than Relay.

### Success Criteria

- Relax should be interoperable with the rest of TVM’s intermediate representations, including loop-level IR and runtime FFI.
    - Relax should also be interoperable with features within each of these IRs, such as automatic tuning within TIR, and framework importers from Relay.
- Relax IR should have interoperabilities through BYOC as the current BYOC flow.

## Performance

### Definitions

Performance can be defined in a couple of different ways:

- **Compilation performance:** defined as the effectiveness of Relax’s optimization passes in transforming and optimizing Relax code.
- **Runtime performance:** defined as the runtime latency for various Relax workloads, whether they are subgraphs or end-to-end models.

### Success Criteria

- Relax IR should provide runtime performance parity with Relay on static workloads.
- Relax IR should provide state-of-the-art performance on dynamic shape workloads.
- Relax’s should preserve compilation performance in its cross-IR interactions and lowering process; meaning that tuning times and cross-IR representations should not be negatively affected by Relax.

## Portability

### Definitions

Portability is defined as the ability of the intermediate representation to compile towards different hardware targets with minimal user intervention.

For example, portability implies that a user shouldn’t have to write different IR code to compile to CPUs vs. GPUs. 

### Success Criteria

- Relax IR should be hardware agnostic, which is a key prerequisite to being a portable IR.
- Relax should provide sufficient pathways to portable code generation.
    - Relax should support lowering to TensorIR.
    - Relax IR should have the ability to support heterogeneous compute capabilities.
    - Relax should support BYOC *(also mentioned in the theme of Interoperability)*.
