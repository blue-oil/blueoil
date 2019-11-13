IMAGE_NAME:=blueoil_$$(id -un)
BUILD_VERSION:=$(shell git describe --tags --always --dirty --match="v*" 2> /dev/null || cat $(CURDIR/.version 2> /dev/null || echo v0))
DOCKER_OPT:=--runtime=nvidia
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
test: build test-classification test-object-detection test-semantic-segmentation

.PHONY: test-classification
test-classification: build
	# Run Blueoil test of classification
	docker run $(DOCKER_OPT) -e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) --rm $(IMAGE_NAME):$(BUILD_VERSION) pytest tests/e2e/test_classification.py

.PHONY: test-object-detection
test-object-detection: build
	# Run Blueoil test of object-detection
	docker run $(DOCKER_OPT) -e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) --rm $(IMAGE_NAME):$(BUILD_VERSION) pytest tests/e2e/test_object_detection.py

.PHONY: test-semantic-segmentation
test-semantic-segmentation: build
	# Run Blueoil test of semantic-segmentation
	docker run $(DOCKER_OPT) -e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) --rm $(IMAGE_NAME):$(BUILD_VERSION) pytest tests/e2e/test_semantic_segmentation.py

.PHONY: test-lmnet
test-lmnet: test-lmnet-pep8 test-lmnet-main

.PHONY: test-lmnet-pep8
test-lmnet-pep8: build
	# Check lmnet pep8
	docker run --rm $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "cd lmnet; tox -e flake8"

.PHONY: test-lmnet-main
test-lmnet-main: build
	# Run lmnet test with Python3.6
	docker run $(DOCKER_OPT) -e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) --rm $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "cd lmnet; tox -e py36-pytest"

.PHONY: test-lmnet-check-dataset-storage
test-lmnet-check-dataset-storage: build
	# Check datasets storage with Python3.6
	docker run $(DOCKER_OPT) -v /storage/dataset:/storage/dataset -e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) -e DATA_DIR=/storage/dataset --rm $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "cd lmnet; tox -e py36-check_dataset_storage"

.PHONY: test-dlk
test-dlk: test-dlk-pep8 test-dlk-main

.PHONY: test-dlk-pep8
test-dlk-pep8: build
	# Check dlk PEP8
	docker run --rm -t $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "cd dlk && pycodestyle --ignore=W --max-line-length=120 --exclude='*static/pb*','*docs/*','*.eggs*','*tests/*','backends/*' ."

.PHONY: test-dlk-main
test-dlk-main: build
	# Run dlk test
	docker run --rm -t -v $(HOME)/.ssh:/tmp/.ssh -v $(CWD)/output:/home/blueoil/dlk/output -e FPGA_HOST --net=host $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "cp -R /tmp/.ssh /root/.ssh && apt-get update && apt-get install -y iputils-ping && cd dlk && python setup.py test"

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
