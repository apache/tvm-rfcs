- Feature Name: [Process RFC] Empowering New Scoped Module to the Project
- Start Date: 2022-10-19
- RFC PR: [apache/tvm-rfcs#95](https://github.com/apache/tvm-rfcs/pull/95)

# Background

Machine Learning Compilation (MLC) is an emerging field in fast development. With the tremendous help from the whole community, it’s exciting to see that TVM delivers significant needs from and to developers, and thus has become widely popular in both academia and industry.

As a rapidly growing field, inevitable needs keep emerging daily as new workloads and demands come in. For example, demand has been evolving from static shape compilation to dynamic shape compilation, from scalar code to tensor cores. As an early player in the field, we led in some of the most important areas, thanks to our close collaboration and agile iteration for innovations.

Success comes from listening to the community's demands. As one of the first-movers in this field, who wants to build the project toward future success, it is important for us to keep listening and always have the following two goals in mind.

- G0: Maintain stable solutions for existing use-cases
- G1: Always be open-minded to new demands, land technical commitment timely, continue to reinvent ourselves, and welcome new members to the community.

G0 is important in the sense that we would like to continue making sure we do not create disruptions in existing code. In the meantime, enabling G1 in a timely manner helps us to stay up in the competition and keep pushing state of the art.

Definition: We categorize a new module as S0-module if it satisfies the following criteria:

- Clearly isolated in its own namespace.
- Clearly needed by some users in the community.
- No disruptive change to the rest of the codebase
- Can be easily deprecated by removing the related namespaces
- Can be turned off through a feature toggle to contain the overall dependency from the rest of the modules.

Common practices: in most projects is to introduce improvements in different phases.

- S0: as being defined in this proposal
- S1: Evolving the overall solutions to make use of the new component.
- S2: Deprecation of some existing solutions or evolving the solutions.

Notably, not all changes have to be scoped as S0-level changes. There are many features that involve S1 level changes which can also be evaluated as part of the RFC process. But nevertheless, having a clear phased development helps us to bring advances to both goals.

Keeping both goals in mind, it is important to enable a mechanism for the community to welcome new scoped modules to the project. Enabling new modules is one way to quickly enable G1 while keeping the existing G0 part stable. This is a common practice established in Apache and non-apache projects. For example, Apache Spark initially started with an optional module GraphX for the graph process, and then came follow-up improvements along the line of SparkGraph. MLIR enables different improvements as dialects, such as TOSA, Torch-MLIR. PyTorch enables new graph exporting mechanisms named TorchFX while also maintaining TorchScript for other existing use cases.

In those past practices, the new components are introduced as optional modules with minimum changes to existing ones. Notably, there can be perceived overlap with some of the existing components, e.g. Torch-MLIR contains similar features around computational graphs as TOSA, but also brings orthogonal improvements to the overall system. As a related example, TorchFX certainly has overlapping features with TorchScript, but also brings in new capabilities along. While not all of them are ASF projects, they are successful practices that enable some of the open source projects to thrive in a similar field that we are in.

As in practices in other machine learning projects, there can be some levels of duplications or missing features compared with existing components (TorchFX TorchScript, TOSA and other MLIR graph IRs). Following the same practice in those related projects, as a team player in the community, one major principle in Apache is to empower communities by empowering optional components if they do not affect existing workflow. Empowering S0 through scoped modules brings a win-win situation for the community: it also brings in new aspirational members who are willing to collaborate and deliver the best for the community. This way, we keep ourselves up-to-date and grow stronger. On the other hand, failure to do so could result in community members getting discouraged and we lose valuable contributions and opportunities for us to grow in this rapidly growing area.

The type of modules can include, but are not limited to:

- IR dialects such as MLIR’s TOSA (while there are other graph IRs). TorchFX(while there is already TorchScript).
- Vertical flows that leverage some of the dialects.
- Backends/frontends.
- Other types of modules introduced in a self-contained namespace.

S0 changes would be contained in its namespace, with possible integrations also built inside its namespace. There can be follow-up steps (S1), such as making a dialect broadly accessible in existing compilation flows. Importantly, further S1/S2 level changes would require different RFC and longer deliberation for G0. The discussions on S1 also would serve as a way to allow the community to have a floor to talk about where broader areas of the project are going through a longer deliberation to maintain G0. Clearly identifying and empowering the S0 stage helps us to enable improvements quickly while bringing energies to the community, empowering a broader set of users, while not disrupting existing use cases.

# Proposal: Empowering S0 Modules

In this process RFC, We’d like to propose a process to encourage S0 modules and set expectations about what we anticipate in such inclusion.

Note that this RFC focuses on the S0 stage. We propose the following guidelines to expedite to process while ensuring, quality and community support:

- More than three PMC members endorse the S0-level proposal to ensure that there are enough eyes and shepherding in the module. The decision to establish a S0-level module needs to get majority support from PMC.
- The code changes of S0-level modules follow the normal code review process as in all other modules in the codebase.*
- A clear set of community members are committed to maintaining the proposed modules with technical support quantitatively, more than three endorsing committers who can serve as the initial owner.
- No implication that everybody has to immediately work on or switch to the new S0 module.
- We expect discussions of the relation of the proposed module with existing ones and reuse when possible, but we do not enforce hard no-overlap rules at S0 stage, as most OSS projects do not require modules to have zero perceived overlap.
- Relations to existing modules and interaction are being clearly discussed, but no hard requirements on zero duplications as per practices in other projects
- Clean isolation of changes from existing modules, when the change touches existing modules, they should be discussed separately.
- In discussions of S0-level RFC, maintain a clear separation from S1, and S2 level decisions in later stages so we can encourage S0 changes early while enabling informed decisions at S1, and S2 levels in continued discussions as the modules continue to evolve in the ecosystem.
- There should be discussions about how the proposal fits into the project to bring clarity. We also acknowledge that not all S1, S2 level decisions can be made at the beginning. Additionally, an S0-module should show a clear positive fit to some(but not all) aspects of the project and clear cohesion to some of the existing modules. As the development evolves, more discussions will happen in future RFCs with additional evidence that help us to make informed decisions.

After the RFC discussion period. One of the PMC members would serve as a champion, provide a clear technical summary of the state, pros and cons during discussions for the S0-level proposal and suggest a path forward. The champion will also continue to drive the overall process of code upstreaming and follow-up discussions.

Transitions of S0-level module. After an S0-level module is established, it can undergo the following possible transitions:

- S0 -> deprecation: When a S0 module no longer has an active set of maintainers, the module will be deprecated. The removal of the module is easy as they are contained in the respective folders, with no modules that come and depend on them.
- S0 -> S1: When developers propose to incorporate a S0 module broadly into existing flows/components. Each such composition would require its own RFC and following the existing RFC process.

# Questions for Discussion

1. Would love to see other suggestions on encouraging new contributions.
