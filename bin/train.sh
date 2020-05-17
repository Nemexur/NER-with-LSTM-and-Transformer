#!/bin/bash
source venv/bin/activate
# Start model training
MODELS_DIR=$1
CONFIGS_DIR=$2
INCLUDE_PACKAGES=$3
rm -rf bilstm_model
allennlp train $CONFIGS_DIR/bilstm_config.jsonnet -s $MODELS_DIR/bilstm_model --include-package $INCLUDE_PACKAGES

rm -rf bilstm_attn_model
allennlp train $CONFIGS_DIR/bilstm_config.jsonnet -s $MODELS_DIR/bilstm_attn_model --include-package $INCLUDE_PACKAGES

rm -rf tener_model
allennlp train $CONFIGS_DIR/tener_config.jsonnet -s $MODELS_DIR/tener_model --include-package $INCLUDE_PACKAGES
