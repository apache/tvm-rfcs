- Feature Name: extend_metadata_in_projectoption
- Start Date: 2021-09-09
- RFC PR: [apache/tvm-rfcs#0020](https://github.com/apache/tvm-rfcs/pull/0000)
- GitHub Issue: [apache/tvm#0020](https://github.com/apache/tvm/issues/0000)

# Summary
[summary]: #summary

This RFC proposes to extend the current metadata associated with project options
provided by the Project API to allow a better integration with command line
interface tools, like TVMC.

# Motivation
[motivation]: #motivation

Currently metadata associated with project options provided by the Project API
is insufficent to allow building easily and automatically command line parsers
used by CLI tool like TVMC.

The metadata available for the project options, stored in instances of the
ProjectOption class are limited as they do not:

1. Provide a list of the API methods which support the options;
2. Allow determination if the options are required or optional;
3. Provide a default value if one is used by the Project API server.

As a consequence it complicates the integration with command line interfaces
that need to create command line arguments based on the project options
available for a platform.

This RFC proposes to extend the existing metadata with four new members in
`ProjectOption` (`required`, `optional`, `type`, and `default`) to address
issues **1.**, **2.**, and **3.** and ease the integration of Project API with
CLI tools.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Below it is explained in detail the need and properties of the four new members
(`required`, `optional`, `type`, and `default`) proposed to be added to the
`ProjectOption` class to extend the project option metadata returned by Project
API `server_info_query` method. `required`, `optional`, and `type` are proposed
as **required fields**, whilst `default` is proposed as an **optional field**.

Modals like "must", "may" and similar ones are interpreted in this RFC
accordingly to the semantics defined by the IETF RFC-2119, 1997.

## On "required" and "optional" metadata

Currently even though all options available for a given project can be
discovered via the Project API `server_info_query` interface there is no way to
know which options belong (or apply) to which API method (like the
`generate_project`, `build`, `flash`, and `open_transport` methods).

This is fine when the user knows beforehand which method accepts a set of
options, so it's possible to manually select which options will be passed to a
given API method, like when using the API in a Python script.

However that's a problem when the API user (e.g. TVMC) needs to automatically
determine the options available for the API methods, like when automatically
building a command line parser with subcommand domains that closely mapped to
the API methods (e.g. subcommands to create, build, and flash a project).

Moreover, currently it's impossible to determine which option is required and
which one is optional, so it would be at least necessary for the API user to
build a static ad hoc table with all options available on a given project
stating which option is required and which one is optional for the project. This
is impractical to maintain and would result in the API user having to update the
the static table every time an option is added, removed, or modified in the
Project API server.

Hence to ease the automatic detection of the options available on each Project
API method the following two new metadata are proposed: `required` and
`optional`.

Both will contain a list of method names for which the option is available,
either as a required option (if in `required` list), or as an optional option
(if in `optional` list). At least one API method must be listed in `required` or
in the `optional` list. A method name must be listed only in the `required` or
in the `optional` list, i.e. an option can't be required and optional at the
same time for given API method. An option can be required for a method and
optional for another method.

The elements in the lists `required` and `optional` must be in the set of method
names implemented by the ProjectAPIClient class and that have the parameter
`options` defined. These methods are dispatched to the server, which implements
the server counterparts to properly handle the client dispatches and
ultimately defines the options available for each API method. The current method
names that satisfy these criteria are `generate_project`, `build`, `flash`, and
`open_transport`.

`required` metadatum or `optional` metadatum (or both) must be provided for
every option.

## On "type" metadatum

The option type can sometimes be determined implicitly by what is returned
in metadatum `choices`, but this not ideal. For example, for option `verbose` it
would be possible to infer it is a boolean option and therefore it can be
converted to a command line flag if metadatum `choices` is a couple of True and
False. Nonetheless that would lead to cumbersome logic at API user side (e.g.
TVMC) to infer the option type, like iterating over the tuple elements to search
for True or False. This can be solved directly if the option type is returned
explicitly with the option.

Thus adding a `type` metadatum allows a much simpler way for the API users to
determine the type of an option when that is necessary for various reasons, like
when building a command line parser based on the available project options.

The types must be only non-complex JSON-serializable primitive types, passed as
strings. Hence the following types are proposed for the `type` metadatum:
`"bool"`, `"str"`, `"int"`, and `"float"`.

`type` metadatum must be provided for every option.

## On "default" metadatum

Sometimes Project API uses a default value if an option is not specified, but
currently there is no way to determine it by using the option metadata.

However it's important for CLI tools to inform users what's the default value
for a given option, if applicable, so the user can decide if it's necessary to
provide a different value.

The default values for the options on a project could be defined at the user
side (e.g. TVMC) but that's not ideal.

Hence having an additional field `default` in the metadata that the API can use
to inform the user if the option has any default value is quite useful. It also
avoids one to keep that information at the client / user side.

`default` may be provided for an option, if applicable.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

All that's necessary to complish the proposal is:

1. Extend `ProjectOption` class by adding the new fields discussed above;
2. Adjust the existing platform options to comply with the required fields.

An example follows considering the Zephyr platform:

```
diff --git a/python/tvm/micro/project_api/server.py b/python/tvm/micro/project_api/server.py
index 07d328597..323bf418a 100644
--- a/python/tvm/micro/project_api/server.py
+++ b/python/tvm/micro/project_api/server.py
@@ -50,7 +50,8 @@ class ProjectOption(_ProjectOption):
     def __new__(cls, name, **kw):
         """Override __new__ to force all options except name to be specified as kwargs."""
         assert "name" not in kw
-        assert "type" in kw, "'type' parameter must be specified"
+        assert "required" in kw or "optional" in kw, "'required' or 'optional' must be specified"
+        assert "type" in kw, "'type' field must be specified"

         kw["name"] = name
         for param in ["choices", "default", "required", "optional"]:



diff --git a/apps/microtvm/zephyr/template_project/microtvm_api_server.py b/apps/microtvm/zephyr/template_project/microtvm_api_server.py
index 4e62739d5..8d0b1722c 100644
--- a/apps/microtvm/zephyr/template_project/microtvm_api_server.py
+++ b/apps/microtvm/zephyr/template_project/microtvm_api_server.py
@@ -216,40 +216,58 @@ if IS_TEMPLATE:
 PROJECT_OPTIONS = [
     server.ProjectOption(
         "extra_files_tar",
-        help="If given, during generate_project, uncompress the tarball at this path into the project dir.",
+        optional=["generate_project"],
         type="str",
+        help="If given, during generate_project, uncompress the tarball at this path into the project dir.",
     ),
     server.ProjectOption(
         "gdbserver_port", help=("If given, port number to use when running the local gdbserver."),
+        optional=["open_transport"],
         type="int",
     ),
     server.ProjectOption(
         "nrfjprog_snr",
-        help=("When used with nRF targets, serial # of the attached board to use, from nrfjprog."),
+        optional=["open_transport"],
         type="int",
+        help=("When used with nRF targets, serial # of the attached board to use, from nrfjprog."),
     ),
     server.ProjectOption(
         "openocd_serial",
-        help=("When used with OpenOCD targets, serial # of the attached board to use."),
+        optional=["open_transport"],
         type="int",
+        help=("When used with OpenOCD targets, serial # of the attached board to use."),
     ),
     server.ProjectOption(
         "project_type",
-        help="Type of project to generate.",
         choices=tuple(PROJECT_TYPES),
+        required=["generate_project"],
         type="str",
+        help="Type of project to generate.",
     ),
     server.ProjectOption("verbose", help="Run build with verbose output.", type="bool"),
     server.ProjectOption(
         "west_cmd",
+        optional=["generate_project"],
+        default="python3 -m west",
+        type="str",
         help=(
             "Path to the west tool. If given, supersedes both the zephyr_base "
             "option and ZEPHYR_BASE environment variable."
         ),
+    ),
+    server.ProjectOption(
+        "zephyr_base",
+        optional=["build", "open_transport"],
+        default="ZEPHYR_BASE",
+        type="str",
+        help="Path to the zephyr base directory.",
+    ),
+    server.ProjectOption(
+        "zephyr_board",
+        required=["generate_project", "build", "flash", "open_transport"],
+        help="Name of the Zephyr board to build for.",
         type="str",
     ),
-    server.ProjectOption("zephyr_base", help="Path to the zephyr base directory.", type="str"),
-    server.ProjectOption("zephyr_board", help="Name of the Zephyr board to build for.", type="str"),
 ]
```

It's important to note that every project option must be at least associated to
one API method (at least one method is listed in the option's `'required'` or
`'optional'` list).

It's possible to enforce that an option will never be passed to an API method
call if it's not a valid option for that method (i.e. the method is listed
neither in the option's `required` list nor in its `optional` list).

That ideally must be enforced at the server side (`server.py`) of the Project
API. The enforcement can consist in having the server removing an invalid option
for a method before the method is called effectively on consulting ProjectOption
for the option being passed and not finding the method in either the
`'required'` or `'optional`' list.

# Drawbacks
[drawbacks]: #drawbacks

The impact on existing code is low and the current project options will only
need to be adjusted to define the new mandatory fields, i.e. `required`,
`optional`, and `type`, all are straightforward to be provided for the existing
options on the current supported platforms.

Adding the four new members to `ProjectOption` will increase the size of
`ProjectOption` class and consequently the amount of data returned by
`server_info_query`, however since Project API client and server run on the same
host that is negligible.

# Rationale and alternatives

An alternative would be to implement the data proposed here as metadata at the
user side instead, however it would complicate the use of Project API by CLI
like TVMC, since ah hoc tables would need to be created for each platform and
for each project option available on the platform, allow one to map options data
for their required or optional methods, default value, etc. Hence that
alternative is impractical or hard to maintain at best for syncing up with
Project API updates (updating the tables to match Project API) would be
necessary every time a new option is added to a project.
