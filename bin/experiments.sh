#!/bin/bash
IMAGE_NAME=ner_experiments
IMAGE_TAG=dev

if [[ "$(docker images ${IMAGE_NAME}:${IMAGE_TAG} 2> /dev/null)" == "" ]]; then
    echo You already have ${IMAGE_NAME}:${IMAGE_TAG}. So we will run it.
    docker run -it --rm -p 3000:3000 -p 6006:6006 ${IMAGE_NAME}:${IMAGE_TAG} bash
else
    echo You do not have ${IMAGE_NAME}:${IMAGE_TAG}. So we will create one.
    docker build -f docker/train.Dockerfile -t ${IMAGE_NAME}:${IMAGE_TAG} .
    docker run -it --rm -p 3000:3000 -p 6006:6006 ${IMAGE_NAME}:${IMAGE_TAG} bash
fi
