IMAGE_NAME:=blueoil_$$(id -un)
INFERENCE_IMAGE_NAME:=blueoil_inference_$$(id -un)
BUILD_VERSION:=$(shell git describe --tags --always --dirty --match="v*" 2> /dev/null || cat $(CURDIR/.version 2> /dev/null || echo v0))
DOCKER_OPT:=--rm -it -u $$(id -u):$$(id -g) -e CUDA_VISIBLE_DEVICES=-1
CWD:=$$(pwd)

default: build

.PHONY: deps
deps:
	# Update dependencies
	git submodule update --init --recursive

.PHONY: install
install: deps
	pip3 install -U pip
	pip3 install -e .[cpu]
	pip3 install pycocotools==2.0.0

.PHONY: install-dev
install-dev: deps install
	pip3 install -e .[test,docs]

.PHONY: install-gpu
install-gpu: install
	pip3 install -e .[gpu]
	pip3 install -e .[dist]

.PHONY: build
build: deps
	# Build docker image
	docker build -t $(IMAGE_NAME):$(BUILD_VERSION) -f docker/Dockerfile .

.PHONY: build-inference
build-inference: deps
	# Build docker image
	docker build -t $(INFERENCE_IMAGE_NAME):$(BUILD_VERSION) -f docker/Dockerfile-inference .

.PHONY: setup-test
setup-test:
	mkdir -p $(CWD)/tmp

.PHONY: test
test: build
	docker run ${DOCKER_OPT} $(IMAGE_NAME):$(BUILD_VERSION) pytest -n auto tests/e2e/

.PHONY: test-classification
test-classification: build setup-test
	# Run Blueoil test of classification
	docker run ${DOCKER_OPT} -v $(CWD)/tmp:/home/blueoil/tmp $(IMAGE_NAME):$(BUILD_VERSION) pytest -n auto tests/e2e/test_classification.py

.PHONY: test-object-detection
test-object-detection: build setup-test
	# Run Blueoil test of object-detection
	docker run ${DOCKER_OPT} -v $(CWD)/tmp:/home/blueoil/tmp $(IMAGE_NAME):$(BUILD_VERSION) pytest -n auto tests/e2e/test_object_detection.py

.PHONY: test-semantic-segmentation
test-semantic-segmentation: build setup-test
	# Run Blueoil test of semantic-segmentation
	docker run ${DOCKER_OPT} -v $(CWD)/tmp:/home/blueoil/tmp $(IMAGE_NAME):$(BUILD_VERSION) pytest -n auto tests/e2e/test_semantic_segmentation.py

.PHONY: test-keypoint-detection
test-keypoint-detection: build setup-test
	# Run Blueoil test of keypoint-detection
	docker run ${DOCKER_OPT} -v $(CWD)/tmp:/home/blueoil/tmp $(IMAGE_NAME):$(BUILD_VERSION) pytest -n auto tests/e2e/test_keypoint_detection.py

.PHONY: test-lint
test-lint: install-dev
	# Check blueoil pep8
	# FIXME: blueoil/templates have a lot of errors with flake8
	flake8 ./blueoil ./tests ./output_template/python --exclude=templates

.PHONY: test-mypy
test-mypy: install-dev
	# Check blueoil mypy
	mypy blueoil

.PHONY: test-unit-main
test-unit-main: build
	# Run lmnet test with Python3.6
	docker run ${DOCKER_OPT} $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "cd tests; pytest --cov=../blueoil --cov-config=unit/.coveragerc -n auto unit/"

.PHONY: test-unit-inference
test-unit-inference: build-inference
	# Run inference test with Python3.6
	docker run ${DOCKER_OPT} $(INFERENCE_IMAGE_NAME):$(BUILD_VERSION) python3 -m unittest discover tests/output_template/

.PHONY: test-dlk
test-dlk: test-dlk-main test-dlk-x86_64 test-dlk-x86_64_avx test-dlk-arm test-dlk-arm_fpga test-dlk-aarch64 test-dlk-aarch64_fpga

.PHONY: test-dlk-main
test-dlk-main: build
	# Run dlk test
	docker run ${DOCKER_OPT} $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "pytest --cov=blueoil/converter tests/converter --ignore=tests/converter/test_code_generation.py"

.PHONY: test-dlk-x86_64
test-dlk-x86_64: build
	# Run dlk test of code_generation for x86_64
	docker run ${DOCKER_OPT} $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "pytest -n auto tests/converter/test_code_generation.py::TestCodeGenerationX8664"

.PHONY: test-dlk-x86_64_avx
test-dlk-x86_64_avx: build
	# Run dlk test of code_generation for x86_64_avx
	docker run ${DOCKER_OPT} $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "pytest -n auto tests/converter/test_code_generation.py::TestCodeGenerationX8664Avx"

.PHONY: test-dlk-arm
test-dlk-arm: build
	# Run dlk test of code_generation for arm
	docker run --rm -it -v $(HOME)/.ssh:/tmp/.ssh -e FPGA_HOST --net=host $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "cp -R /tmp/.ssh /root/.ssh && apt-get update && apt-get install -y iputils-ping && pytest tests/converter/test_code_generation.py::TestCodeGenerationArm"

.PHONY: test-dlk-arm_fpga
test-dlk-arm_fpga: build
	# Run dlk test of code_generation for arm_fpga
	docker run --rm -it -v $(HOME)/.ssh:/tmp/.ssh -e FPGA_HOST --net=host $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "cp -R /tmp/.ssh /root/.ssh && apt-get update && apt-get install -y iputils-ping && pytest tests/converter/test_code_generation.py::TestCodeGenerationArmFpga"

.PHONY: test-dlk-aarch64
test-dlk-aarch64: build
	# Run dlk test of code_generation for aarch64
	mkdir -p $(CWD)/output
	docker run ${DOCKER_OPT} -v $(CWD)/output:/home/blueoil/output $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "pytest -n auto tests/converter/test_code_generation.py::TestCodeGenerationAarch64"

.PHONY: test-dlk-aarch64_fpga
test-dlk-aarch64_fpga: build
	# Run dlk test of code_generation for aarch64_fpga
	docker run ${DOCKER_OPT} $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "pytest -n auto tests/converter/test_code_generation.py::TestCodeGenerationAarch64Fpga"

.PHONY: clean
clean:
	# Clean created files
	docker rmi  $(IMAGE_NAME):$(BUILD_VERSION)
	rm -rf tmp/*
