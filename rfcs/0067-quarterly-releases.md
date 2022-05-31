- Feature Name: Release Schedule
- Start Date: 2022-04-21
- RFC PR: [apache/tvm-rfcs#67](https://github.com/apache/tvm-rfcs/pull/67)

# Summary
This RFC proposes that TVM move to a quarterly release schedule. Releases would happen every 3 months or so on a schedule set well in advance, independent of individual feature development in TVM.

# Motivation
Releases are essential to the usage of TVM, especially now that are beginning to work on publishing binary packages for TVM under PyPi. Making TVM releases frequent forces the release process to become well documented and simple rather than bespoke and only achievable by a small group. Users benefit from releases by seeing that the the project is still under active development and providing an easy way to get new features. As of this RFC it has been five months since the last release. It could easily confuse new users when they expect some TVM feature that was only developed recently but is not present in the latest official release.


# Guide-level explanation
TVM has [release process documentation](https://tvm.apache.org/docs/contribute/release_process.html). This RFC proposes that a release candidate vote thread sent on a schedule roughly every three months. A release branch will be cut, evaluated for a period of two weeks, evaluated by the PMC, then a release published. Publishing a release entails:

* Gathering and organizing release notes since the last release
* Posting a source code release on GitHub
* Uploading the source code to the Apache SVN repository
* Uploading binary packages to tlcpack
* Uploading binary packages to PyPi
* Getting an approval vote from the PMC

# Reference-level explanation

Prior to a release a new, lightweight vote would instead be used to nominate a release manager, a committer who will be responsible for guiding along the release process. Releases will be published roughly every three months, with one in January, April, July and October. The release manager will have final say on all dates and consideration should be given for their personal schedule. The timeline will be as follows:

* Three weeks prior to the release
    * The release manager will cut a release branch and create a new tag
    * The release manager will audit the licenses of all project dependencies to ensure they are compatible with Apache
    * The release manager will open a GitHub issue announcing the release branch cut and target date for the release and state that any further inclusions in the release must be manually cherry-picked
    * The release manager will create a PR targeted to merge into the release branch with the necessary changes to make a release (i.e. changing version numbers)
    * Contributors that wish changes to be cherry-picked should comment on the announcement issue with the relevant PRs and commits and their reasoning. The release manager has final say on which changes should be included but should aim to be inclusive at this stage
    * The release manager begins gathering release notes

* Two weeks prior to release
    * Cherry picks become limited to critical changes only
    * The release manager begins building binaries and testing them against TVM's test suite and running on relevant hardware
    * The release manager sends the release notes and plan to a vote thread for approval (feedback thread will be open for one week)
    * The release manager performs another license audit of all project dependencies to ensure they are compatible with Apache

* One week prior to release
    * The release manager sends the release and binaries to a vote thread for approval (thread open for one week). If rejected the building and test process must be started anew and repeated until the PMC approves.

* Day of the release
    * The release manager publishes the relevant binaries, including the source `.tar` to Apache
    * The release manager closes the release issue
    * The release manager makes a GitHub release via a tag and updates an in-repo file `RELEASE.md` on both `main` and the release branch with the release notes

Much of this can be automated via GitHub Actions on the apache/tvm repo. Eventually (though maybe not for the upcoming release) these will handle all the building, testing, and publishing of releases so the job of the release manager will become simpler over time.

Developers are heavily encouraged to use feature flags and Python API conventions (i.e. prefixing unstable or private APIs with an underscore) to enable in-progress features. It is important both for users to be able to access new or experimental behavior but also that they can get a stable interface, so these kinds of changes should be hidden behind opt-in flags.

Even with this RFC the ultimate decision to releases rests with the community. If no one wants to do a release, then it will be skipped or delayed until the community agrees and a release manager can be selected.

When publishing a new release, the release manager should also check the previous release version's branch to see if any commits have been added since the last release and publish a new minor version if so.

## Versioning
The release manager will use the release notes and discussions with developers to determine the next version number. Releases will continue the current versioning scheme of `major.minor.patch`, with a typical release involving a bump of the minor release version. Patch versions will be used for follow up releases onto a quarterly release, but not for the next quarter's release. The release manager will be responsible for maintaining the release until the next release, which mainly entails putting up a patch release if necessary. The release manager may also delegate this responsibility to another party if both agree.

Releases will be supported and get patch updates so long as that minor version is the most recent. For example, version 0.N.0 will get patches until version 0.(N + 1).0 is released. Critical patch fixes may also be backported to older releases at the community and release manager's descretion. The release manager is responsible for patch releases for bug fixes as well, though they may delegate to another if they so choose. When asking for support on forums, one of the first steps for users should be to reproduce their bug on most recently nightly build of TVM.

# Prior Work

* This [proposal](https://discuss.tvm.apache.org/t/pre-rfc-switch-to-time-based-releases/4245) suggests a similar thing for TVM and links to a great [write-up](https://cwiki.apache.org/confluence/display/KAFKA/Time+Based+Release+Plan) from the Kafka project describing their plan.

# Drawbacks

* This requires a high level of commitment from a single individual (the release manager). If someone is not able to fill or execute this role effectively the release will be stalled.

# Rationale and alternatives

* The main alternative is waiting until specific features have been developed in order to make a release. This has caveats as TVM has many sub-projects within it, so it's not clear which is significant enough to warrant a release.

