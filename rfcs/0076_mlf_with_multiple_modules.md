- Feature Name: mlf_with_multiple_module_support
- Start Date: 2022-05-31
- RFC PR: [apache/tvm-rfcs#0075](https://github.com/apache/tvm-rfcs/pull/0075)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)

# Summary
[summary]: #summary

This RFC describes a new version of model library format where we can support adding multiple Relay module builds into a single MLF file.

# Motivation
[motivation]: #motivation

MLF artifact is an important piece of building a microTVM project. In the first version of the MLF artifact, 
MLF only supports a single "Relay module"/"PackedFunc" build. However, there are cases where adding
multiple Relay modules is required. For example, corstone300 tests using multiple Relay modules was introduced by
contributors from ARM. The goal of this RFC is to create a new version of MLF which has standard support for these cases. 

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Users will call `export_model_library_format` function with multiple Relay builds(output from `tvm.relay.build`) or 
a single Relay build module. We change the `export_model_library_format` API to support multiple Relay Build, however 
it is backward compatible and, we handle required changes internally.

Since we are including multiple modules in a single MLF file, we need to make some changes. Let's assume we
want to export an MLF file for with two Relay build modules called `mod1` and `mod2`.

1. We change format of `metadata.json` file. We will add `modules` as a top-level key to JSON file which represents
   multiple Relay build module. `modules` is a dictionary and each Relay build module is differentiated by 
   its module name. Each module key(`mod1`, `mod2`, ...) will have the same properties that was defined for a single
   Module except `version` which will stay as a top-level key. For example:
    ```javascript
    {
      'modules': {
        'mod1': {
          "export_datetime": "",
          "external_dependencies": [],
          "memory": {
            "functions": {
              "main": [],
              "func0": [],
              ...
            }
          },
          "model_name": "mod1",
          "style": "",
          "target": [],
        },
        'mod2': {
          "export_datetime": "",
          "external_dependencies": [],
          "memory": {
            "functions": {
              "main": [],
              "func0": [],
              ...
            }
          },
          "model_name": "mod2",
          "style": "",
          "target": [],
        },
        ...
      },
      'version': XXX,
    }
    ```
  2. Each module also has a Relay text file which has `relay.txt` name. We propose to structure Relay text file
     with different name for each module and use `.relay` extension to differentiate its format and keep all in the same 
     directory(E.g. {`mod1.relay`, `mod2.relay`, ...}). Similarly, for `graph.json` file we propose to restructure the name of the file
     to {`mod1.graph`, `mod2.graph`, ...}. This approach is similar to the existing approach for parameters where we 
     use {`mod1.params`, `mod2.params`, ...}.
  3. Finally, we keep higher level information which are not specific to a module at higher level in the metadata file. 
     Currently, we only have `version` which shows the MLF version, but this could grow in the future.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

Building on the same example in previous section, here we explain what are the API calls and expected output after this change.

First, we build two modules:
```python
mod1 = ...
mod2 = ...

executor = Executor("graph")
runtime = Runtime("crt")
target = tvm.target.target.micro("host")

with tvm.transform.PassContext(opt_level=3):
  factory1 = tvm.relay.build(mod1, target, runtime=runtime, executor=executor, mod_name="mod1")
  factory2 = tvm.relay.build(mod2, target, runtime=runtime, executor=executor, mod_name="mod2")
```

Then, we pass both results of `tvm.relay.build` to `export_model_library_format` function and path to generated model
library format:
```python
micro.export_model_library_format([factory1, factory2], mlf_tar_path)
```

Now, if we extract MLF file here is the file structure:
```bash
# codegen source files
codegen/
  host/
    src/
      mod1_lib0.c
      mod1_lib1.c
      mod1_lib2.c
      mod2_lib0.c
      mod2_lib1.c
      mod2_lib2.c

# graph 
executor-config/
  graph/
    mod1.graph
    mod2.graph

# metadata file
metadata.json

# parameters
parameters/
  mod1.params
  mod2.params

# relay text output
src/
  mod1.relay
  mod2.relay
```

# Drawbacks
[drawbacks]: #drawbacks

The drawback here is that we are changing generated metadata and file structure in the MLF file which means external 
tools which are dependent to this need to update their tool. Hopefully this RFC makes it clear on what steps they need
to take. Also, since we are updating the version field in metadata, external dependencies will be notified of this change.

# **Rationale and alternatives**
An alternative way to implement this feature is to break down each Relay build module to a subdirectory and keep the
previous format inside each sub Relay build module. Using the example of `mod1` and `mod2`, in this approach we have
an MLF file format with structure below if we extract:
```bash
# mod1
mod1/
  codegen/
    host/
      src/
      mod_lib0.c
      mod_lib1.c
      mod_lib2.c
  executor-config/
    graph/
      mod.graph
  parameters/
    mod.params
  src/
    mod.relay

#mod2
mod2/
  codegen/
    host/
      src/
      mod_lib0.c
      mod_lib1.c
      mod_lib2.c
  executor-config/
    graph/
      mod.graph
  parameters/
    mod.params
  src/
    mod.relay
```

One of the benefits of this approach is that it creates a more readable file structure which is modularized for
each Relay build module. However, the downside is that this approach will result in more modifications in project
generation. For instance, since we have multiple C source file directories and also more header file directories
we need to consider all of those in project generation. 

# Prior art
[prior-art]: #prior-art

Prior art is the [RFC](https://discuss.tvm.apache.org/t/rfc-tvm-model-library-format/9121) for the first version of the MLF design.

