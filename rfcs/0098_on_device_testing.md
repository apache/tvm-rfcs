- Feature Name: On-Device Testing in TVM CI
- Start Date: 2023-01-24
- RFC PR: [apache/tvm-rfcs#0098](https://github.com/apache/tvm-rfcs/pull/0098)
- GitHub Issue: [apache/tvm#0000](https://github.com/apache/tvm/issues/0000)
- Authors: [Mehrdad Hessar](https://github.com/mehrdadh), [David Riazati](https://github.com/driazati) 
# Summary
[summary]: #summary

This RFC describes the approach and challenges to add non-merge-blocking hardware testing in TVM CI.

# Motivation
[motivation]: #motivation

Testing is a major part of any open source project to show its stability to the users and companies who are adopting the project. More than 700 contributors are involved with TVM who works at various companies with different needs/interests in TVM. This means the demand for thorough testing is increasing every day. At the time of writing, TVM tests generally run on the hardware targets when that hardware is available in the cloud (for example, x86 CPU, i386, GPU and AArch64). In addition, TVM has supports hardware targets that are not available in the cloud, such as embedded devices supported by microTVM or the Hexagon DSP. The TVM CI cannot currently test code on those hardware as part of its CI, leaving a gap in testing.

It is possible for TVM to include on-device tests for these non-cloud devices in its CI. However, because they are not widely available to use in cloud services, blocking PR merges over failures in those tests could impose an undue burden on contributors who don’t have access to that hardware. In that hypothetical world, all contributors would, at some point, need to find a way to debug those tests on such non-cloud hardware, even if they didn’t have access to it.

This does not mean TVM community cannot still run tests on these hardware, either as part of CI in a non-merge-blocking way or against `main` at an e.g. nightly or post-merge cadence. This RFC aims to find a way for TVM community members with access to those special hardware to be able to expand coverage of TVM CI in an advisory capacity by adding instances of their hardware to TVM CI.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

We explain this section by following the simplest case of a hardware-in-loop CI testing which is nightly regression testing. Anyone in the community can run nightly regression tests and provide the results to the community. We expect hardware vendors to be one of the parties primarily interested in having nightly testing on their hardware targets that are supported in TVM, but this document refers to anyone running a test as Device Test Maintainers.

## Test Procedure
There is a minimal set of requirements that TVM community expects Device Test Maintainers to follow. To add nightly tests to TVM, Device Test Maintainers should implement automation that performs the following steps:
1. **Lookup nightly SHA for testing.** To ensure that results from disparate nightly test suites can be compares, an automated nightly process chooses a TVM sha1 which everyone should use. The bot will merge the new daily commits on main branch to TVM `nightly` branch ([PR13564](https://github.com/apache/tvm/pull/13564) implemented this) at 9:00PM PST. Device Test Maintainers should use the sha1 from this `nightly` branch for testing so we have consistent results across multiple CIs.

2. **Testing.** At a minimum, Device Test Maintainers should re-run any simulated integration tests ordinarily ran in TVM’s CI on real hardware targets. In addition, they are welcome to bring more tests with more input samples or tuning with more trials to show better accuracy and performance benchmark. For nightly, running the test could be trigger based on timer and implemented however the HW vendor desires. This way Device Test Maintainers have flexibility on the implementation and are not required to make a connection to TVM Jenkins node.

3. **Test results.** We expect Device Test Maintainers to publicly report functional test results for any on-device tests which also run on simulators in the TVM CI. To facilitate this, TVM will provide reporting infrastructure (i.e. a test dashboard) to present those results in public domain. Our proposal is that Device Test Maintainers upload the tests results in the form of pytest artifacts to an S3 bucket which is provided by TVM community. Device Test Maintainers are also welcome to show the results in the form of a website, but the tests artifacts should be uploaded to the S3 bucket so they can be retrieved in future.
    - Other alternative is to use a Github repository to host the test results. Github repo is not the ideal solution for saving and downloading files and it could be slow for hosting large number of files for a website.

## What is Tested?
Nightly tests could vary based on the target. Some hardware targets have minimal testing in TVM which runs on simulator. For these hardware, Device Test Maintainers should at least run the same tests on physical hardware to validate the simulator tests. In addition, the HW vendor could add any other unit test or full model end-to-end testing which is in the interest of the maintainer or TVM community. In addition they can run existing tests in TVM with modification. For instance, in tuning tests we only run limited trials in the TVM CI, or for accuracy check we only check for limited number of samples. However, nightly regression could run for larger samples or trials to show better accuracy/performance results.

# Test Tiers
So far we explained a minimal setup to bring a on-device testing CI to TVM on a nightly basis. However, in principle one could enable more frequent testing. TVM defines these tiers:
1. **Tier 1: Run CI for all PRs.** This tier is equivalent to testing support for existing hardware targets that exist in cloud. This case requires large resources to avoid increasing the CI time. TVM community expects close CI infrastructure monitoring if they a Device Test Maintainer registers at this tier. If failures are observed in a CI at this tier which are due to to failures in the CI infrastructure, TVM community expect it to be resolved in one day time frame. If this requirement is not fulfilled the mentioned CI would be degraded to lower tiers.

2. **Tier 2: Run CI for PRs with specific tags.** In this case, TVM CI can parse the PR titles (i.e. `[microtvm] add schedules for baz operator`) and based on that decide to run this CI. In addition, if the contributors/committers think running this PR in certain CI is important they can use defined tags with TVM-BOT to trigger the hardware CI (i.e. `@tvm-bot run-odt microtvm`).

3. **Tier 3: Run CI nightly.** This scenario is the bare minimum case which was explained earlier. Nightly testing is not only useful to catch errors introduced by PRs in 24 hours. It also could be useful for longer regression tests which has been discussed for TVM in TVM community meetings.

## Test Hooks for Tier 1 and Tier 2
The three steps that was explained above are mostly in common between different tiers. However, there are some adjustments that are required for Tier 1 and 2 explained below.

Triggering the test pipeline for tier 1 and 2 is different than nightly tests. Nightly test could happen independently from the TVM Jenkins pipeline, but in tier 1 and 2 the structure changes. Given that PRs change frequently, it is more efficient for TVM to trigger a Device Test Maintainer’s CI. In this case, Device Test Maintainers can receive notifications [via GitHub Webhook](https://docs.github.com/en/developers/webhooks-and-events/webhooks/webhook-events-and-payloads) with an API token to send (1) an event stating that a job has started and should be marked `pending` on a commit and (2) an event to say that a job has completed and should be marked `success` / `failed` along with a results artifact. 

Using this mechanism, Device Test Maintainers can present the On-Device Test status on GitHub per executed PR and also present the result after the CI finishes. TVM will provide templates of the result artifact so Device Test Maintainers can adopt it. In the beginning, the the result artifact would be a plain text log file and overtime we will upgrade it to use pytest artifacts.

# FAQ
In this section we answer some of the questions that are important to answer.

1. Do Tier 1 and 2 On-Device CIs delay PR merging?
    
    Tier 1 and Tier 2 On-Device CIs are merely advisory and they are not considered to block merging a PR. Therefore, the timing on a CI is flexible.
    
    However, we recommend the Device Test Maintainer to provide enough resources based on the Tier that the CI is running. This would help contributors/committers who are actively working on certain not-cloud targets and try to keep the hardware CI error free.

2. How will the results of Tier 1 and 2 On-Device CIs be shown?
    - If an On-Device CI is triggered as a result of a PR, e.g. `cortexm-zephyr-hardware-test`, it will show up as an entry of the CI tests. When it is not triggered, it wouldn’t show up at all.
    - The results will be pending/success/failure which is the same as the rest of the CI steps.
3. In the case of failures, how will contributors know if they can ignore it?
    It is tough to answer this question since it is very subjective. Generally, On-Device CIs exist to provide additional information for the contributors. Here are some suggestions, using `cortex-m` as an example here:
    
    - A contributor who is submitting a PR directly related to `cortex-m`*. The community expectation is that the contributor does not ignore failures on cortex-M hardware CI. Community also expects the device test maintainer to provide help/guidance to the contributor to fix the issue. However, this is totally based on the good faith of the contributor and cannot be forced by the committers.
    - A contributor who is submitting a PR not related to `cortex-m`*. In case of failure the first question is “Why didn’t our unit tests capture this issue?” The community expects the contributor for file an issue and move on. Follow up fixes to capture this issue in unit tests and fix on the hardware could happen in follow up PRs.

4. What happens if there are not enough resources to run a Tier 1 or 2 On-Device CI, and a PR is in pending situation?
    - First, this shows that the resource allocation is not well considered considering the tier that device test maintainer registered their CI. If this happens often, TVM contributors can report this to the Device Test Maintainer and it is expected that device test maintainer would fix it.
    - Contributors/Committers can ignore this CI, if it happens often, and merge without the green check mark from the hardware CI.

5. How should a TVM community member propose a new On-Device CI?
  
    Prospective Device Test Maintainers should propose an RFC explaining the motivation, tests that are considered and TVM integration with their CI (i.e. tier) to show tests results.   
    
6. Are there any starting points or templates to help Device Test Maintainers implement Tier 1 or 2 On-Device CIs?
    
    [Recent changes](https://github.com/apache/tvm/pull/13300) in TVM CI have divided TVM previous single large CI into multiple smaller Jenkins files which are more readable and easy to manage. A device test maintainer could reuse those Jenkins files with their own Jenkins instance to add an On-Device CI for specific hardware. We expect TVM to keep this CI reusability. 
    
    In addition to Jenkins file any device test maintainer requires an infrastructure to manage their device fleet on each server. Since each device test maintainer is specialized in their domain we expect the infrastructure to be internally managed. From the TVM community perspective, there are few expectations from the device test maintainer:
    
    - Provide N number workers to Jenkins head to have enough support to run PRs in parallel. N is defined based on the tier that the device test maintainer is registering their CI and it is subjective to the target and CI traffic, CI time, etc.
    - For each worker provide M number of devices. M is defined based on number of tests that are running on device and it should provide enough parallelism to avoid this CI to be the bottle neck.
    - Device fleet management should be reliable. The community does not expect a hardware CI to have overly frequent failures, otherwise TVM community can decide on disabling it until it is further reliable.

7. What if TVM tests in a Device Test Maintainer's CI fails unacceptably more often than the CI? How do we handle that?
    
    In general, the Device Test Maintainer is responsible for the CI they added and TVM community expects them to provide points of contact for the CI. Considering that, Maintainers can find the root cause of the issue. 
    
    If they determine the root cause to be a change introduced in a PR, Maintainers can file issues or request the PR owner to do further investigation. Since the assumption is that the community does not have access to these special devices, it is the responsibility of the CI Maintainer to provide guidance and help to fix the issue.
    
    If the root cause is in the On-Device CI infrastructure, the Maintainer should try to resolve the problem, working with the community as needed. If Tier 1 or 2 CIs begin to fail noticeably more often, TVM CI Monitoring Rotation could eventually disable the test to avoid contributors presuming that On-Device tests always fail. TVM CI Maintainers are expected to coordinate with Device Test Maintainers before doing this.

8. Is CI Monitoring Rotation responsible for monitoring for failures?

    The CI monitoring is focused on branch `main` which is post merge testing. In the case of tier 3, a nightly run essentially mean running the CI once or more in 24 hours. In this case we might see failures in the On-Device CI. If a Device Test Maintainer chooses this tier, they should contribute in CI monitoring by having designated people watching their CI and addressing issues.