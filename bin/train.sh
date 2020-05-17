#!/bin/bash
source venv/bin/activate
# Start model training
CONFIGS_DIR = $1
INCLUDE_PACKAGES = $2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
allennlp train $CONFIGS_DIR/bilstm_config.jsonnet \
    # Construct model name from Configs Directory
    --serialization-dir bilstm_model \
    --include-package $INCLUDE_PACKAGES

allennlp train $CONFIGS_DIR/bilstm_config.jsonnet \
    # Construct model name from Configs Directory
    --serialization-dir bilstm_model \
    --include-package $INCLUDE_PACKAGES

allennlp train $CONFIGS_DIR/tener_config.jsonnet \
    # Construct model name from Configs Directory
    --serialization-dir tener_model \
    --include-package $INCLUDE_PACKAGES
