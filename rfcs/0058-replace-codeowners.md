- Feature Name: Replace codeowners
- Start Date: 2022-02-18
- RFC PR: [apache/tvm-rfcs#58](https://github.com/apache/tvm-rfcs/pull/58)

# **Summary**

Move `.github/CODEOWNERS` to `.github/CODEOWNERSHIP` to avoid triggering GitHub’s [automatic review requests](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners#about-code-owners) and add GitHub Actions automation to a) populate review requests/cc's based on pull request (PR) thread traffic and b) ping languishing PRs.

# **Motivation**

Languishing PRs are a relatively common occurrence in TVM today. In order to maintain a vibrant open source community, we should work to reduce or eliminate these occurrences. PRs languish for a variety of reasons, but a common problem new contributors have is finding a reviewer for their PR.

In [June 2021](https://github.com/apache/tvm/pull/8500), TVM attempted to solve this problem by introducing `CODEOWNERS`. The goal was to make it easy for contributors to find a reviewer for their PR. This approach was problematic because the layout of TVM code meant that a file-based approach to sharding code ownership was incompatible with most PRs. The average PR spans so many `CODEOWNERS` directories that the average PR triggered code-owner requests to half of the TVM committers, diluting the responsibility of each reviewer and in many cases generating spam for reviewers who won’t end up reviewing the PR at all.

It would be nice to rely on automation by tuning the `CODEOWNERS` file. However, the organization of TVM is such that the scope of a directory includes many different efforts. For instance, a change in `src/relay/backend` may affect the core compiler but may also affect automation and runtime. Tuning `CODEOWNERS` could well amount to adding file-level ownership, and maintaining that is intractable.

We also attempted to tune `CODEOWNERS` by switching to round-robin review style in [tvm#9057](https://github.com/apache/tvm/issues/9057), but this approach ran into problems, summarized by @areusch in [this comment](https://github.com/apache/tvm/issues/9057#issuecomment-931579113).

## Guide-level explanation

Many reviewers find it difficult to sort through review traffic and determine which PRs they are on the hook for. GitHub provides solutions for this in the form of Review Requests and Assignee fields—reviewers can list the PRs which mention them there. Since `CODEOWNERS` worked by auto-populates the Review Requests field, removing `CODEOWNERS` from the repo in turn allows us to reuse these fields to better track who is on the hook for reviewing a PR. We should take this opportunity in spam reduction to attempt to make GitHub PR traffic more relevant to the community as a whole. A great way to do this is to develop a better system for populating Review Requests and Assignee fields to leverage the GitHub PR review system.

However, absent `CODEOWNERS`, these fields need to be manually filled. One could imagine that TVM Committers and Triagers could triage new PRs and populate those fields. However, there are some limitations on this system imposed by GitHub and the Apache Software Foundation which make this difficult. Specifically, anyone mentioned in a PR must either be a TVM committer or actively participating (e.g. by replying) in the PR in order to be placed in the Review Requests field. This means that a committer must continuously monitor a PR thread to keep those fields as accurate as possible.

Since manual processes often lead to inconsistencies, and the conditions above are somewhat adversarial, some automation is desirable here to attempt to standardize on one system for tagging PRs with assigned reviewers. To address that need, this RFC proposes two additional changes:

1. Automatically assigning reviewers based on cc tags in PR messages
2. Periodic automated ping messages for participants in PRs to prevent PRs from languishing
3. Automatically ping people based on opt-in subscriptions for labels. Use `CONTRIBUTORS.md` to add reviewers to PRs based on people's stated areas of expertise / interest. It will be up to everyone to manage their topic subscriptions (by editing an issue or submitting a PR to change `CONTRIBUTORS.md`).

Committers are responsible for monitoring and triaging new PRs and issues to the relevant parties, and this RFC doesn’t change that. It assists by reducing notification spam so that each notification a committer gets is now something that needs to be addressed.

# **Reference-level explanation**

Removing `CODEOWNERS` solves the spam issue by stopping the deluge of notifications to committers, but introduces a new issue in that PR authors still need to be able to assign reviewers. The combination of [tvm#9934](https://github.com/apache/tvm/pull/9934) and [tvm#9973](https://github.com/apache/tvm/pull/9973) are meant to address this. Since many TVM contributors don’t have permissions to add reviewers themselves, anyone who is a committer that is addressed in a PR body with `cc @username` will be added as a reviewer. [tvm#10322](https://github.com/apache/tvm/pull/10322) introduces a mechanism for anyone to opt-in to certain topics, such as `microTVM` and will be cc'ed by a bot on any PR or issue that has the `microTVM` label or the text `[microTVM]` in the title.

PRs should not stay open forever and should get reviewed in a timely manner. The second PR linked above addresses this by periodically (currently set to wait 7 days) pinging PRs that have not had recent activity.

These two tools should make it so the TVM community is still able to maintain a good velocity on PRs while avoiding spamming committers with notifications.

# **Drawbacks**

It may make it more difficult for some PRs to get reviews. Instead of everyone being tagged, no one is tagged. We need to rely on active committers and triagers to triage new PRs without review requests to the relevant people.

# **Rationale and alternatives**

- Narrow `CODEOWNERS` to people that will commit to reviewing every request they receive. This is likely untenable due to the volume and cross cutting nature of many changes (i.e. a small change to one file as part of a larger PR will trigger reviewers for that file, even if they can’t review the entire PR).
- Drastically lower the requirements to become a committer. This would remove the need for some of the automation above as we could rely on GitHub reviews instead of bespoke tools but we would still need to get rid of `CODEOWNERS` to avoid spam. Additionally, the set of reviewers will become broader, improving PR response latency but increasing the need for coordination amongst reviewers.
- Use [GitHub teams](https://docs.github.com/en/organizations/organizing-members-into-teams/about-teams) to assign reviews. This is difficult since the teams have to be created in the Apache organization which is hard for us to [manage](https://issues.apache.org/jira/browse/INFRA-22864). Despite sharing responsibility, this still leads to lots of notifications for participants.

# **Future possibilities**

- There could be a rotation of triagers for new PRs and issues. When responsibility is shared, it is easy for someone to say they thought another committer would do the triaging and PRs/issues end up unaddressed. There could be a specific triager assigned each week to monitor PRs and issues. PyTorch has a [similar process](http://blog.ezyang.com/2021/01/pytorch-open-source-process/).
