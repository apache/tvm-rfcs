- Feature Name: Commit Message Guideline
- Start Date: 2022-08-12
- RFC PR: [apache/tvm-rfcs#0088](https://github.com/apache/tvm-rfcs/pull/88)

# Summary
[summary]: #summary

This RFC proposes adding a Commmit Message Guideline to TVM documentation to
help guide contributors on how to write good commit messages when submitting
code / PRs (Pull Requests) to Apache TVM.

# Motivation
[motivation]: #motivation

Currently TVM commit logs are less than ideal because many commit messages lack
valuable information and don't follow any format standard.

Valuable information is usually left behind in Github PR conversations or
discussion threads in the Discuss forum, making it hard to retrieve them when
inspecting the commit messages -- using `git log`, for instance.

Because commit messages are an indirect but important aspect of code quality,
and also important for code maintenance, it is essential for a long term open
source project to ensure that they meet high standards.

The importance of commit messages conveying enough context and information about
the code being changed will grow as the project grows and bad (poorly written)
commit messages can affect negatively the code quality of future changes that
would otherwise benefit from past good commit messages if they existed.

Beyond code itself, poorly written commit messages can also affect the community
in other ways. For example, by not providing to new contributors a consistent
and complete history or context for the code changes, it can work as a barrier
for new contributions because much more time will be necessary trying to
understand what motivated a past critical but unclear change.

Hence this Commit Message Guideline can help contributors to write good commit
messages and so improve the current situation regarding the TVM commit logs.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Commit Message Guideline

Apache TVM uses the Github (GH) platform for patch submission and code review
via Pull Requests (PRs). The final commit (title and body) that is merged into
the Apache TVM main tree is composed of the PR's title and body and must be kept
updated and reflecting the new changes in the code as per the reviews and
discussions.

Although these guidelines apply essentially to the PRs’ title and body messages,
because GH auto-generates the PR’s title and body from the commits on a given
branch, it’s recommended to follow these guidelines right from the beginning,
when preparing commits in general to be submitted to the Apache TVM project.
This will ease the creation of a new PR, avoiding rework, and also will help the
review.

The rules below will help to achieve uniformity that has several benefits, both
for review and for the code base maintenance as a whole, helping you to write
commit messages with a good quality suitable for the Apache TVM project,
allowing fast log searches, bisecting, and so on.

_PR/commit title_:

* Guarantee a title exists (enforced);
* Don’t use Github usernames in the title, like @username (enforced);
* A tag must be present as a hint about what component(s) of the code
  the PRs / commits “touch” (enforced). For example [BugFix], [CI], [microTVM],
  and [TVMC]. Tags go between square brackets and appear first in the title. If
  more than one tag exist, multiple brackets should be used, like [BugFix][CI].
  The case recommended for tags, in geral, is the upper camel case. For example,
  prefer the forms [Fix], [BugFix], and [Docker] instead of [fix], [bug_fix],
  and [docker]. Acronyms should be kept as such so, for example, use [CI] and
  [TVMC] instead of [ci] and [tvmc]. Tags help reviewers to identify the PRs
  they can/want to review and also help the release folks when generating the
  release notes;
* Use an imperative mood. Avoid titles like “Added operator X” and “Updated
  image Y in the CI”, instead use the forms “Add feature X” and “Update image Y
  in the CI” instead;
* Observe proper use of caps at the beginning (uppercase for the first letter)
  and for acronyms, like, for instance, TVM, FVP, OpenCL. Hence instead of
  “fix tvm use of opencl library”, write it as “Fix TVM use of OpenCL library”;
* Do not put a period at the end of the title.

_PR/commit body_:

* Guarantee a body exists (enforced);
* Don’t use Github usernames in body text, like @username (enforced);
* Avoid “bullet” commit message bodies: “bullet” commit message bodies are not
  bad per se, but “bullet” commit messages without any description or
  explanation is likely as bad as commits without any description, rationale,
  or explanation in the body.

For minor deviations from these guidelines, the community will normally favor
reminding the contributor of this policy over reverting or blocking a commmit /
PR.

Commits and PRs without a title and/or a body are not considered minor
deviations from these guidelines and hence must be avoided.

Most importantly, the contents of the commit message, especially the body,
should be written to convey the intention of the change, so it should avoid
being vague. For example, commits with a title like “Fix”, “Cleanup”, and
“Fix flaky test” and without any body text should be avoided. Also, for the
review, it will leave the reviewer wondering about what exactly was fixed or
changed and why the change is necessary, slowing the review.

Below is an example that can be used as a model:

> [microTVM] Zephyr: Remove zephyr_board option from build, flash, and open_transport methods
>
> Currently it’s necessary to pass the board type via ‘zephyr_board’ option to
> the Project API build, flash, and open_transport methods.
>
> However, since the board type is already configured when the project is
> created (i.e. when the generate_project method is called), it’s possible to
> avoid this redundancy by obtaining the board type from the project
> configuration files.
>
> This commit adds code to obtain the board type from the project CMake files,
> removing this option from build, flash, and open_transport methods, so it’s
> only necessary to specify the ‘zephyr_board’ option when calling
> generate_project.
>
> This commit also moves the ‘verbose’ and ‘west_cmd’ options from ‘build’
> method to ‘generate_project’, reducing further the number of required options
> when building a project, since the ‘build’ method is usually called more often
> than the ‘generate_project’.

After a new PR is created and the review starts it’s common that reviewers will
request changes. Usually the author will address the reviewers’ comments and
push additional commits on top of the initial ones. For these additional commits
there is no recommendation regarding the commit messages. However if the
additional commits render the PR title and/or body outdated then it's the
author's responsibility to keep the PR title and body in sync with new changes
in the code and updated the PR title and body accordingly (remember that the PR
title and body will be used to compose the final commit message that will land
in the main tree).

Committers will seek to fix any issues with the commit message prior to
committing but they retain the right to inform the author of the rules and
encourage them to follow them in future. Also, they retain the right to ask to
the author to update the PR title and/or body when they are not correctly
updated or fixed.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

TVM Community must reach a certain concensus about the rules in this guideline,
hence this RFC will be voted.

Once it's voted and approved the Commit Message Guideline text will be added to
`./docs/contribute/pull_request.rst` doc, under section 'Submit a Pull Request',
below subsection 'Guidelines', as a subsection named “Commit Message Guideline”.
The text in the second-last item in subsection 'Guidelines' that mentions PR
tags will also be extended (a hyperlink will be added) to refer to this
guideline, since it also contains guidelines about use of tags.

New contributors can consult the Commit Message Guidelilne before submitting
PRs. Also, committers and reviewers can use this guideline when reviewing PRs in
case some clarification or help is necessary about how an author or contributor
should write or improve the PR's title and body.

# Drawbacks
[drawbacks]: #drawbacks

None.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

# Prior art
[prior-art]: #prior-art

This guideline is similar to other ones already being used succesfully in other
open source projects, like gcc, Zephyr, LLVM, and Linux.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

# Future possibilities
[future-possibilities]: #future-possibilities

A linter can also be used with tvm-bot to enforce some rules or aspects of these
guidelines, when pertinent. If that is implemented the rules enforced by the bot
would be those marked with "enforce" in the guideline.
