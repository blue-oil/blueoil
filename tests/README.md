# CI Servers for Blueoil
We have several CI Servers
* With GPU/OpenDatasets : https://jenkins.blueoil.org
* With FPGA device : https://jenkins.leapmind.local:8080 (only for LeapMind local access for now)

## Automated Test with PR
The tests run automatically by making PR.

The triggers to run tests are below.
* Create PR
* Push commit to branch related PR
* Comment in PR
    * Run all tests
        * Comment `run test`
    * Run the specified test individually
        * blueoil test : `run blueoil test`
        * lmnet test : `run lmnet test`
        * dlk test : `run dlk test`


## Test Jobs
All `make` tasks should be made in repository root dir.

### [blueoil] test blueoil entire workflow with GPU
* Cofiguration URL : https://jenkins.blueoil.org/job/blueoil_main/configure
* Script run by Jenkins : `make test`

### [lmnet] test lmnet training part with GPU
* Cofiguration URL : https://jenkins.blueoil.org/job/blueoil_lmnet/configure
* Script run by Jenkins : `make test-lmnet`

### [dlk] test compiling and inference part on FPGA device
* Cofiguration URL : http://jenkins.leapmind.local:8080/job/blueoil_dlk_test/configure
* Now this test is available in LeapMind internal only because of the device limitation
* Script run by Jenkins : `make test-dlk`
