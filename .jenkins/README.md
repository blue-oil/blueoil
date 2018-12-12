## Jenkins Server
Here is jenkins server for CI : https://jenkins.leapmind:8443

## Auto Test with PR
This test will be run automatically by PR.

The triggers of running test are below.
* Create PR
* Push commit to branch related PR
* Comment `run test` in PR

### Test scripts
This test executes `./.jenkins/lmnet.sh`

### Test setting
The detail of setting : https://jenkins.leapmind:8443/job/lmnet/

## Auto Document Builder for master branch
There is a job of document builder on jenkins.

This job is triggered by pushing to master branch (poling each 2 minutes).

Current master branch is protected, so this job will be kicked only by merging PR.

This job update documents and restart document server, so we can always access latest document in https://lmnet.docs.leapmind:8000/

### Job scripts
This job executes `./.jenkins/lmnet_doc_builder.sh`

### Job setting
The detail of setting : https://jenkins.leapmind:8443/job/lmnet_doc_builder/
