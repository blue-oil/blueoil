IMAGE_NAME:=blueoil_$$(id -un)
BUILD_VERSION:=$(shell git describe --tags --always --dirty --match="v*" 2> /dev/null || cat $(CURDIR/.version 2> /dev/null || echo v0))
DOCKER_OPT:=--runtime=nvidia

default: build

.PHONY: deps
deps:
	@echo Update dependencies
	git submodule update --init --recursive

.PHONY: build
build: deps
	@echo Build docker image with ubuntu18.04
	docker build -t $(IMAGE_NAME):$(BUILD_VERSION) .

.PHONY: test
test: build test-classification test-object-detection test-semantic-segmentation
	@echo Run all tests

.PHONY: test-classification
test-classification: build
	@echo Run Blueoil test of classification
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) bash ./blueoil_test.sh  --task classification

.PHONY: test-object-detection
test-object-detection: build
	@echo Run Blueoil test of object-detection
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) bash ./blueoil_test.sh  --task object_detection

.PHONY: test-semantic-segmentation
test-semantic-segmentation: build
	@echo Run Blueoil test of semantic-segmentation
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) bash ./blueoil_test.sh  --task semantic_segmentation --additional_test

.PHONY: test-lmnet
test-lmnet: test-lmnet-pep8 test-lmnet-main test-lmnet-check-dataset-storage

.PHONY: test-lmnet-pep8
test-lmnet-pep8: build
	@echo Check lmnet pep8
	docker run --rm $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "cd lmnet; tox -e flake8"

.PHONY: test-lmnet-main
test-lmnet-main: build
	@echo Run lmnet test with Python3.6
	docker run $(DOCKER_OPT) -e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) --rm $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "cd lmnet; tox -e py36-pytest"

.PHONY: test-lmnet-check-dataset-storage
test-lmnet-check-dataset-storage: build
	@echo Check datasets storage with Python3.6 (only available on Jenkins)
	docker run $(DOCKER_OPT) -v /storage/dataset:/storage/dataset -e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) -e DATA_DIR=/storage/dataset --rm $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "cd lmnet; tox -e py36-check_dataset_storage"

.PHONY: test-dlk
test-dlk: test-dlk-pep8 test-dlk-main

.PHONY: test-dlk-pep8
test-dlk-pep8: build
	@echo Check dlk PEP8
	docker run --rm -t $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "cd dlk && pycodestyle --ignore=W --max-line-length=120 --exclude='*static/pb*','*docs/*','*.eggs*','*tvm/*','*tests/*','backends/*' ."

.PHONY: test-dlk-main
test-dlk-main: build
	@echo Run dlk test
	docker run --rm -t -v $(HOME)/.ssh:/tmp/.ssh -e FPGA_HOST --net=host $(IMAGE_NAME):$(BUILD_VERSION) /bin/bash -c "cp -R /tmp/.ssh /root/.ssh && apt-get update && apt-get install -y iputils-ping openssh-client && cd dlk && python setup.py test"

.PHONY: clean
clean:
	@echo Clean created files
	docker rmi  $(IMAGE_NAME):$(BUILD_VERSION)
	rm -rf tmp/*
