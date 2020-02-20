IMAGE_NAME:=blueoil_$$(id -un)
BUILD_VERSION:=$(shell git describe --tags --always --dirty --match="v*" 2> /dev/null || cat $(CURDIR/.version 2> /dev/null || echo v0))
DOCKER_OPT:=--rm -it -u $$(id -u):$$(id -g) -e CUDA_VISIBLE_DEVICES=-1
CWD:=$$(pwd)

default: build

.PHONY: deps
deps:
	# Update dependencies
	git submodule update --init --recursive

.PHONY: build
build: deps
	# Build docker image
	docker build -t $(IMAGE_NAME):$(BUILD_VERSION) --build-arg python_version="3.6.3" -f docker/Dockerfile .

.PHONY: test
test: build
	docker run ${DOCKER_OPT} $(IMAGE_NAME):$(BUILD_VERSION) pytest -n auto tests/e2e/

.PHONY: test-classification
test-classification: build
	# Run Blueoil test of classification
	docker run ${DOCKER_OPT} $(IMAGE_NAME):$(BUILD_VERSION) pytest -n auto tests/e2e/test_classification.py

.PHONY: test-object-detection
test-object-detection: build
	# Run Blueoil test of object-detection
	docker run ${DOCKER_OPT} $(IMAGE_NAME):$(BUILD_VERSION) pytest -n auto tests/e2e/test_object_detection.py

.PHONY: test-semantic-segmentation
test-semantic-segmentation: build
	# Run Blueoil test of semantic-segmentation
	docker run ${DOCKER_OPT} $(IMAGE_NAME):$(BUILD_VERSION) pytest -n auto tests/e2e/test_semantic_segmentation.py

.PHONY: test-keypoint-detection
test-keypoint-detection: build
	# Run Blueoil test of keypoint-detection
	docker run ${DOCKER_OPT} $(IMAGE_NAME):$(BUILD_VERSION) pytest -n auto tests/e2e/test_keypoint_detection.py

.PHONY: test-lmnet
test-lmnet: test-blueoil-pep8 test-unit-main

.PHONY: test-blueoil-pep8
test-blueoil-pep8: build
	# Check blueoil pep8
	# FIXME: blueoil/templates and blueoil/converter have a lot of errors with flake8
	docker run ${DOCKER_OPT} $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "flake8 ./blueoil --exclude=templates,converter"

.PHONY: test-unit-main
test-unit-main: build
	# Run lmnet test with Python3.6
	docker run ${DOCKER_OPT} $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "cd tests; pytest -n auto unit/"

.PHONY: test-dlk
test-dlk: test-dlk-main test-dlk-x86_64 test-dlk-arm test-dlk-arm_fpga test-dlk-aarch64

.PHONY: test-dlk-main
test-dlk-main: build
	# Run dlk test
	docker run ${DOCKER_OPT} $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "pytest tests/converter --ignore=tests/converter/test_code_generation.py"

.PHONY: test-dlk-x86_64
test-dlk-x86_64: build
	# Run dlk test of code_generation for x86_64
	docker run ${DOCKER_OPT} $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "pytest -n auto tests/converter/test_code_generation.py::TestCodeGenerationX8664"

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

.PHONY: rootfs-docker
rootfs-docker:
	docker build -t $(IMAGE_NAME)_os -f docker/Dockerfile_make_os . #--no-cache=true

.PHONY: rootfs-armhf
rootfs-armhf: rootfs-docker
	docker run -v $(CWD)/make_os/build:/build -it $(IMAGE_NAME)_os /build/make_rootfs.sh armhf

.PHONY: rootfs-arm64
rootfs-arm64: rootfs-docker
	docker run -v $(CWD)/make_os/build:/build -it $(IMAGE_NAME)_os /build/make_rootfs.sh arm64

.PHONY: clean
clean:
	# Clean created files
	docker rmi  $(IMAGE_NAME):$(BUILD_VERSION)
	rm -rf tmp/*
