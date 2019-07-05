# CI Servers for Blueoil
We have several CI Servers
* With GPU/OpenDatasets : https://jenkins.blueoil.org
* With FPGA device : https://jenkins.leapmind.local:8080 (only for LeapMind local access for now)
* Buildkite Pipelines : https://buildkite.com/blueoil

## Automated Test with PR
The tests run automatically by making PR.

The triggers to run tests are below.
* Create PR
* Push commit to branch related PR
* Comment in PR
    * Run all tests
        * Comment `run test`
    * Run the specified test individually (Jenkins)
        * blueoil test : `run blueoil test`
        * lmnet test : `run lmnet test`
        * dlk test : `run dlk test`
    * Run the specified test individually (Buildkite)
        * blueoil/lmnet test: `run gpu test`
        * dlk test : `run fpga test`
* Add label in PR (Only available for Buildkite)
    * Run all tests
        * Add `CI: force-run`
    * Run the specified test individually
        * blueoil/lmnet test: Add `CI: force-run-gpu`
        * dlk test : Add `CI: force-run-fpga`


## Test Jobs
All `make` tasks should be made in repository root dir.

### [blueoil] test blueoil entire workflow with GPU
#### Jenkins
* Cofiguration URL : https://jenkins.blueoil.org/job/blueoil_main/configure
* Script run by Jenkins : `make test`
#### Buildkite
See: [.buildkite/blueoil-pipeline.yml](../.buildkite/blueoil-pipeline.yml)

### [lmnet] test lmnet training part with GPU
#### Jenkins
* Cofiguration URL : https://jenkins.blueoil.org/job/blueoil_lmnet/configure
* Script run by Jenkins : `make test-lmnet`
#### Buildkite
See: [.buildkite/lmnet-pipeline.yml](../.buildkite/lmnet-pipeline.yml)

### [dlk] test compiling and inference part on FPGA device
#### Jenkins
* Cofiguration URL : http://jenkins.leapmind.local:8080/job/blueoil_dlk_test/configure
* Now this test is available in LeapMind internal only because of the device limitation
* Script run by Jenkins : `make test-dlk`
#### Buildkite
See: [.buildkite/dlk-pipeline.yml](../.buildkite/dlk-pipeline.yml)
