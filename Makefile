IMAGE_NAME=ner_experiments
IMAGE_TAG=dev
HAS_IMAGE=$(shell sh -c "docker image inspect ${IMAGE_NAME}:${IMAGE_TAG} >/dev/null 2>&1 && echo yes || echo no")

.PHONY:
experiments:
ifeq ($(HAS_IMAGE),no)
	docker build -f docker/train.Dockerfile -t ${IMAGE_NAME}:${IMAGE_TAG} .
endif
	docker run -it --rm -p 3000:3000 -p 6006:6006 ${IMAGE_NAME}:${IMAGE_TAG} bash

.PHONY: clean
clean:
	docker rmi ner_experiments:dev --force
