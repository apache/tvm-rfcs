- Feature Name: Release Schedule
- Start Date: 2022-04-21
- RFC PR: [apache/tvm-rfcs#67](https://github.com/apache/tvm-rfcs/pull/67)

# Summary
This RFC proposes that TVM move to a quarterly release schedule. Releases would happen every 3 months or so on a schedule set well in advance, independent of individual feature development in TVM.

# Motivation
Releases are essential to the usage of TVM, especially now that are beginning to work on publishing binary packages for TVM under PyPi. Making TVM releases frequent forces the release process to become well documented and simple rather than bespoke and only achievable by a small group. Users benefit from releases by seeing that the the project is still under active development and providing an easy way to get new features. As of this RFC it has been five months since the last release. It could easily confuse new users when they expect some TVM feature that was only developed recently but is not present in the latest official release.


# Guide-level explanation
TVM has [release process documentation](https://tvm.apache.org/docs/contribute/release_process.html). This RFC proposes that the release candidate vote thread be abolished in favor of a mechanical schedule where releases happen roughly every three months. A release branch will be cut, evaluated for a period of two weeks, then a release published. Publishing a release entails:

* Gathering and organizing release notes since the last release
* Posting a source code release on GitHub
* Uploading the source code to the Apache SVN repository
* Uploading binary packages to tlcpack
* Uploading binary packages to PyPi

# Reference-level explanation

Prior to a release a new, lightweight vote would instead be used to nominate a release manager, a committer who will be responsible for guiding along the release process. Releases will roughly match the quarterly [calendar dates](https://en.wikipedia.org/wiki/Calendar_year#Quarters) shifted two weeks earlier (Q1: mid March, Q2: mid June, Q3: mid September, Q4: mid December). The release manager will have final say on all dates and consideration should be given for their personal schedule. The timeline will be as follows:
* Three weeks prior to the release
    * The release manager will cut a release branch and create a new tag
    * The release manager will open a GitHub issue announcing the release branch cut and target date for the release and state that any further inclusions in the release must be manually cherry-picked
    * The release manager will create a PR targeted to merge into the release branch with the necessary changes to make a release (i.e. changing version numbers)
    * Contributors that wish changes to be cherry-picked should comment on the announcement issue with the relevant PRs and commits and their reasoning. The release manager has final say on which changes should be included but should aim to be inclusive at this stage
    * The release manager begins gathering release notes

* One week prior to release
    * Cherry picks become limited to critical changes only
    * The release manager begins building binaries and testing them against TVM's test suite

* Day of the release
    * The release manager publishes the relevant binaries
    * The release manager closes the release issue
    * The release manager makes a GitHub release and updates an in-repo file `RELEASE.md` on both `main` and the release branch with the release notes

Much of this can be automated via GitHub Actions on the apache/tvm repo. Eventually (though maybe not for the upcoming release) these will handle all the building, testing, and publishing of releases so the job of the release manager will become simpler over time.

# Prior Work

* This [proposal](https://discuss.tvm.apache.org/t/pre-rfc-switch-to-time-based-releases/4245) suggests a similar thing for TVM and links to a great [write-up](https://cwiki.apache.org/confluence/display/KAFKA/Time+Based+Release+Plan) from the Kafka project describing their plan.

# Drawbacks

* This requires a high level of commitment from a single individual (the release manager). If someone is not able to fill or execute this role effectively the release will be stalled.

# Rationale and alternatives

* The main alternative is waiting until specific features have been developed in order to make a release. This has caveats as TVM has many sub-projects within it, so it's not clear which is significant enough to warrant a release.

