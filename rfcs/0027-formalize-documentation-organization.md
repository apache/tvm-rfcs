- Feature Name: `Formalize TVM Documentation Organization`
- Start Date: 2021-09-01
- RFC PR: [apache/tvm-rfcs#0027](https://github.com/apache/tvm-rfcs/pull/0027)
- GitHub Issue: [apache/tvm#8987](https://github.com/apache/tvm/issues/8987)

# Summary
[summary]: #summary

This RFC proposes a refactoring of TVM documentation. The goal of this refactor
is to create a document architecture that classifies four major documentation
types:

* Tutorials,
* How-tos,
* Deep Dives,
* and Reference

then organizes the documents based on those types. The desired result is to
make it easier for the entire TVM community to find documentation that meet
their needs, whether they are new users or experienced users. Another goal is
to make it easier for the developer community to contribute to the TVM
documentation. While most communities have a distinct divide between the user and the developer, TVM's community has a significant overlap due to TVM's use as an optimizing compiler.

# Motivation
[motivation]: #motivation

TVM has seen an explosion of growth since it was released as an open source
project, and formally graduated into an official Apache Software Foundation
project. The vision of the Apache TVM Project is to host a "diverse community
of experts and practitioners in machine learning, compilers, and systems
architecture to build an accessible, extensible, and automated open-source
framework that optimizes current and emerging machine learning models for any
hardware platform."

The TVM community has done an excellent job in producing a wide range of
documents to describe how to successfully install, use, and develop for TVM.
The documentation project grew with the community to address the immediate
needs of the developer. However, one consistent piece of feedback is
that the documentation is difficult to navigate, with beginner material mixed
in with advanced material. As a result, it can be difficult for new users to
find the exact information they need, and can work against the vision of the
project.

This RFC aims to refactor the organization of the TVM docs, loosely following
the [formal documentation style described by
Divio](https://documentation.divio.com). This system has been chosen because it
is a:

> "simple, comprehensive and nearly universally-applicable scheme. It is proven
> in practice across a wide variety of fields and applications."

This RFC is primarily concerned with the organization of the documents, and not
the content. As such, the implementation of this RFC would move documents, and
only create new documents as top-level placeholders, indexes, and documentation
about the system itself.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

## The Four Document Types

### Introductory Tutorials

These are step by step guides to introduce new users to a project. An
introductory tutorial is designed to get a user engaged with the software
without necessarily explaining why the software works the way it does. Those
explanations can be saved for other document types. An introductory tutorial
focuses on a successful first experience. These are the most important docs to
turning newcomers into new users and developers. A fully end-to-end tutorial&mdash;
from installing TVM and supporting ML software, to creating and training a
model, to compiling to different architectures&mdash;will give a new user the
opportunity to use TVM in the most efficient way possible. A tutorial teaches a
beginner something they need to know. This is in contrast with a how-to, which
is meant to be an answer to a question that a user with some experience would
ask.

Tutorials need to be repeatable and reliable, because the lack of success means
a user will look for other solutions.

### How-to Guides

These are step by step guides on how to solve particular problems. The user can
ask meaningful questions, and the documents provide answers. An examples of
this type of document might be, “how do I compile an optimized model for ARM
architecture?” or “how do I compile and optimize a TensorFlow model?” These
documents should be open enough that a user could see how to apply it to a new
use case. Practical usability is more important than completeness. The title
should tell the user what problem the how-to is solving.

How are tutorials different from how-tos? A tutorial is oriented towards the
new developer, and focuses on successfully introducing them to the software and
community. A how-to, in contrast, focuses on accomplishing a specific task within
the context of basic understanding. A tutorial helps to on-board and assumes
no prior knowledge. A how-to assumes minimum knowledge, and is meant to guide
someone to accomplish a specific task.

### Reference

Reference documentation describes how the software is configured and operated.
APIs, key functions, commands, and interfaces are all candidates for reference
documentation. These are the technical manuals that let users build their own
interfaces and programs. They are information oriented, focused on lists and
descriptions. You can assume that the audience has a grasp on how the software
works and is looking for specific answers to specific questions. Ideally, the
reference documentation should have the same structure as the code base and
be generated automatically as much as possible.

### Explanations (Deep Dive)

Explanations are background material on a topic. These documents help to
illuminate and understand the application environment. Why are things the way
they are? What were the design decisions, what alternatives were considered,
what are the RFCs describing the existing system? This includes academic papers
and links to publications relevant to the software. Within these documents you
can explore contradictory and conflicting position, and help the reader make
sense of how and why the software was built the way it is. It’s not the place
for how-to’s and descriptions on how to accomplish tasks. They instead focus
on higher level concepts that help with the understanding of the project.
Generally these are written by the architects and developers of the project,
but can useful to help both users and developers to have a deeper understanding
of why the software works the way it does, and how to contribute to it in ways
that are consistent with the underlying design principles.

## Special considerations for TVM

The TVM community has some special considerations that require deviation from
the simple docs style outlined by Divio. The first consideration is that there
is frequently overlap between the user and developer communities. Many projects
document the developer and user experience with separate systems, but it is
appropriate to consider both in this system, with differentiations where
appropriate. As a result the tutorials and how-tos will be divided between
"User Guides" that focus on the user experience, and "Developer Guides" that
focus on the developer experience.

The next consideration is that there are special topics within the TVM
community that benefit from additional attention. These topics include, but are
not limited to, microTVM and VTA. Special "Topic Guides" can be created to
index existing material, and provide context on how to navigate that material
most effectively.

To facilitate newcomers, a special "Getting Started" section with installation
instructions, a overview of why to use TVM, and other first-experience documents
will be produced. 

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

## Document Organization

### Top Level Organization

* Getting Started
* User Guide
* Topic Guide
* Developer Guide
* Architecture Guide
* Reference
* Index

### Organization with Major Sections

* Getting Started
  * About TVM
  * Installing TVM
  * Contributor Guide
* User Guide
  * Tutorial
  * How To
* Topic Guide
  * MicroTVM Guide (index to existing docs)
  * VTA (index to existing docs)
* Developer Guide
  * Contributor Tutorial (new, to be written)
  * How To
* Architecture Guide
  * Architecture Overview (new, diagram/map, to be written)
  * ...
* Reference
  * Language Reference
  * API Reference
* Index

### Organization with Detailed Description

* Getting Started
  * About TVM
  * Installing TVM
  * Contributor Guide
    * Community Guideline
    * Performing Code Reviews
    * Committer Guide
    * Writing Document and Tutorials
    * Code Guide and Tips
    * Error Handling Guide
    * Submitting a Pull Request
    * Git Usage Tips
    * Apache TVM Release Process
* User Guide
  * Tutorial
    * Introduction
    * An Overview of TVM and Model Optimization
    * Installing TVM
    * Compiling and Optimizing a Model with TVMC
    * Compiling and Optimizing a Model with the Python Interface (AutoTVM)
    * Working with Operators Using Tensor Expression
    * Optimizing Operators with Schedule Templates and AutoTVM
    * Optimizing Operators with Auto-scheduling
    * Cross Compilation and RPC
    * Introduction to TOPI
    * Quick Start Tutorial for Compiling Deep Learning Models
  * How To
    * Install TVM
    * Install from Source
    * Docker Images
    * Compile Deep Learning Models
    * Deploy Deep Learning Models
    * Work With Relay
    * Work with Tensor Expression and Schedules
    * Optimize Tensor Operators
    * Auto-Tune with Templates and AutoTVM
    * Use AutoScheduler for Template-Free Auto Scheduling
    * Work With microTVM
* Topic Guide
  * MicroTVM Guide (index to existing docs)
    * -> Work With microTVM
    * ->  microTVM Architecture
  * VTA (index to existing docs)
* Developer Guide
  * Contributor Tutorial
    * ...
  * How To
    * Write an operator
    * Write a backend
    * ...
* Architecture Guide
  * Architecture Overview
  * Research Papers
  * Front-end
  * Relay: Graph-level design: IR, pass, lowering
  * TensorIR: Operator-level design: IR, schedule, pass, lowering
  * TOPI: Pre-defined operators operator coverage
  * AutoScheduler / AutoTVM: Performance tuning design
  * Runtime & microTVM design
  * Customization with vendor libraries BYOC workflow
  * RPC system
  * Target system
* API Reference (reference)
  * Language Reference
  * API Reference
    * Generated C++ Docs…
    * Generated Python Docs…
* Index

## Document Code Organization

This refactor will require a shift of how the documents are organized. In
general, Tutorials and How-Tos are written as Sphinx Gallery documents,
allowing for the generation of text, python source, and Jupyter Notebooks. This
allows the user to consume these working code samples in a number of ways, but
comes at the cost of fixed format that can be confusing to navigate. To help
mitigate this, the tutorials and how-tos will be broken up into a more fine
grained directory structure. For example:

    tvm/
      gallery/
        dev_how_tos/
          compile_models/
          ...
        how_tos/
        tutorial/

Rather than render the gallery in one pass as a nested structure (resulting in
a single page with multiple sections), instead each directory will be rendered
independently. This will aid in navigation through the galleries, and also give
more fine-grained grouping of similar topics. The naming of the directory
reflects the organization of Sphinx documentation folder, for example:

    tvm/
      docs/
        deep_dive/
        how_tos/
          index.rst
          **compile_models/**
          ...
        reference/
        **tutorial/**
        dev_deep_dive/
        dev_how_tos/
        dev_reference/
        dev_tutorial/

Depending on the type of documentation, some of the directories may be
generated. For example, the tutorial and compile_models directories are
auto-generated by Sphinx Gallery. To add a new Sphinx Gallery requires the
following steps:

1. Create a gallery subdirectory with the how-to or tutorial documents
2. Create entries in the docs conf.py example_dirs and gallery_dirs variables
   to reference the source and target directories.
3. Update the appropriate index pages in the docs subdirectories to add the new
   directories to the Sphinx table of contents.

# Drawbacks
[drawbacks]: #drawbacks

One consistent drawback of this approach is how major sub-projects are handled.
For example, microTVM may require a specific set of tutorials and how-tos, but
these can become mixed in with other TVM specific documents. This will be
mitigated through two means:

* Subdirectories within the How-Tos can target specific topics.
* Landing pages can be created for specific topics that collect links to all of
  the pages related to that topic.

Another drawback is that this format may require a user to dig deeper on the
first run experience, requiring them to dig into a tutorial or how-to to
install the software. This can be mitigated by refactoring the landing page to
include a “Quick Start” guide for installing the TVM software.

Throughout the open source ecosystem, there is often a distinction between
documentation for users and documentation for developers. The TVM community is
unique in that frequently users will need to extend TVM to accomplish some
goal, for example adding a new backend for code generation. This issue is
addressed by dividing the user and developer topics, but keeping them within
the same documentation system.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

This style of documentation has been formalized by developed by Divio 3 and
deployed throughout the open source communities. Although it can be difficult
to characterize documents within the system (“Should this be a developer or
user doc?” “Is this a tutorial or a how-to?”), working within the constraints
of a formalized system brings many benefits:

* Preventing documentation sprawl: Rather than create new top-level
  headings to capture new ideas, new ideas are logically documented at
  different levels of detail within the for existing types.
* Creating a consistent user experience: Users know exactly where to look
  depending on their needs. New users will find a path to success through
  tutorials, while existing users who need to solve common tasks can look to
  the how-tos for guidance.
* Encouraging new documentation: Developers have a framework for what docs
  should look like, and where they should go.
* Reusing current content: A proof-of-concept implementation of
  this method consisted largely of moving new documents.
* Creating a framework to improve existing content: Many how-tos duplicate
  steps repeatedly. This will allow us to identify the duplications and
  refactor the documents into more targeted forms.

In researching documentation systems, there aren’t many formalized systems that
have been published.

# Prior art
[prior-art]: #prior-art

## Projects That Follow This Style

### Kubernetes

Kubernetes roughly follows this style, augmented with a landing page and a getting started page.

* Home
* Getting Started
* Concepts
* Tasks
* Tutorials
* Reference
* Contribute

### Numpy

Numpy also follows a similar style, with a very flat organization and additional documents of interest to users.

* What is NumPy?
* Installation
* NumPy quickstart
* NumPy: the absolute basics for beginners
* NumPy fundamentals
* Miscellaneous
* NumPy for MATLAB users
* Building from source
* Using NumPy C-API
* NumPy Tutorials
* NumPy How Tos
* For downstream package authors
* F2PY Users Guide and Reference Manual
* Glossary
* Under-the-hood Documentation for developers
* NumPy’s Documentation
* Reporting bugs
* Release Notes
* Documentation conventions
* NumPy license

## Projects in the ML Community

### PyTorch

PyTorch has a much more fragmented style, with Getting Started, Tutorials, and
Docs (reference docs) spread across a variety of locations and using a variety
of styles. The leads to a much more fragmented user experience. However, it has
also been cited as a positive learning experience, and the tag search feature
is powerful for the volume of documentation. Developing a similar site would
likely be resource intensive.

### TensorFlow

TensorFlow follows a style that’s closer to working from beginner to advanced.
One stand out feature is a graphical representation of the ecosystem, with links
to docs that fall into a particular categorization. When building out the
developer documents, it may be worthwhile to consider a similar structure.

## Projects in ASF

Hadoop and Spark follow a very loose and informal documentation structure.

## Sphinx Documentation Style

It’s instructive to look at the documentation style of a project for producing
documentation. Sphinx follows a structure that is similar to the Divio style,
but focuses more on guiding the user from getting started through advanced
topics, similar to the TensorFlow style.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

* This documentation system only loosely addresses how sub-projects should be handled.
* It does not consider specific future documents, or a plan for refactoring
  duplicated content in existing documents.
* It does not address some style issues, like how to ensure every document in a
  Sphinx Gallery has an appropriate image associated with it.
* It does not address how to incorporate the new RFC process with the
  documentation process.
* It does not address how to handle testing of documents and impact on CI.
* It does not address Incorporating accepted or completed RFCs into the
  documentation structure.
* It does not address the role of documentation in the CI/CD pipeline.
* The style and format of inline reference documentation is out of scope of this
  proposal. For example,
  [how to document passes in Relay](https://github.com/apache/tvm/pull/8893).


# Future possibilities
[future-possibilities]: #future-possibilities

Future work should include graphical navigation of the project, similar to the
TensorFlow ecosystem map, and possibly based on the TVM architecture diagram
described in the [pre-RFC discuss
post](https://discuss.tvm.apache.org/t/updated-docs-pre-rfc/10833)

# Reference

Please refer to the [TVM Discuss
Forum](https://discuss.tvm.apache.org/t/updated-docs-pre-rfc/10833) for
additional discussion on this RFC.
