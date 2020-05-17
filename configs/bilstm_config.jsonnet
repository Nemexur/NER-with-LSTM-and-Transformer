local NUM_THREADS = 2;
local CHARS_NUM_FILTERS = 32;
local PRETRAINED_EMBEDDING_DIM = 100;
local EMBEDDING_DIM = 100;
local IS_CONLL_DATA = std.extVar("IS_CONLL_DATA");
local USE_PRETRAINED_EMBEDDINGS = std.extVar("USE_PRETRAINED_EMBEDDINGS");

// Classes used in training
//// Readers
local READER = {
  "type": "conll2003_ner",
  "lazy": true,
  "tag_label": "ner",
  "token_indexers": {
    "tokens": {
      "type": "single_id",
      "lowercase_tokens": true
    },
    "token_characters": {
      "type": "characters",
      "min_padding_length": 3
    }
  }
};

//// Iterators
local BUCKET_ITERATOR = {
  "type": "bucket",
  "batch_size": 128,
  "biggest_batch_first": true,
  "max_instances_in_memory": 128 * 8,
  "sorting_keys": [["tokens", "num_tokens"]]
};

//// Embedders
local PRETRAINED_TOKEN_EMBEDDER = {
  "type": "embedding",
  "embedding_dim": PRETRAINED_EMBEDDING_DIM,
  "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.100d.txt.gz",
  "trainable": true
};

local TOKEN_EMBEDDER = {
  "type": "embedding",
  "embedding_dim": EMBEDDING_DIM,
  "trainable": true
};

local CHAR_EMBEDDER = {
  "type": "cnn",
  "embedding_dim": 16,
  "num_filters": CHARS_NUM_FILTERS,
  "ngram_filter_sizes": [2, 3],
  "conv_layer_activation": "relu"
};

//// Encoders
local LSTM_ENCODER = {
  "type": "lstm",
  "input_size": if USE_PRETRAINED_EMBEDDINGS == "1" then PRETRAINED_EMBEDDING_DIM + CHARS_NUM_FILTERS * 2 else EMBEDDING_DIM + CHARS_NUM_FILTERS * 2,
  "hidden_size": 256,
  "num_layers": 2,
  "dropout": 0.5,
  "bidirectional": true
};

{
  "dataset_reader": READER,
  "train_data_path": std.extVar("NER_TRAIN_DATA"),
  "validation_data_path": std.extVar("NER_TEST_DATA"),
  "model": {
    "type": "crf_tagger",
    "dropout": 0.5,
    "label_encoding": if IS_CONLL_DATA == "1" then "IOB1" else "BIO",
    "constrain_crf_decoding": true,
    "calculate_span_f1": true,
    "verbose_metrics": true,
    "include_start_end_transitions": false,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": if USE_PRETRAINED_EMBEDDINGS == "1" then PRETRAINED_TOKEN_EMBEDDER else TOKEN_EMBEDDER,
        "token_characters": {
          "type": "character_encoding",
          "embedding": {
            "embedding_dim": 16
          },
          "encoder": CHAR_EMBEDDER
        }
      }
    },
    "encoder": LSTM_ENCODER
  },
  "iterator": BUCKET_ITERATOR,
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "learning_rate_scheduler": {
      "type": "exponential_from_epoch",
      "gamma": 0.6,
      "from_epoch": 2
    },
    "shuffle": true,
    "validation_metric": "+f1-measure-overall",
    "num_serialized_models_to_keep": 2,
    "num_epochs": 10,
    "grad_norm": 5.0,
    "patience": 2,
    "should_log_learning_rate": true,
    "should_log_parameter_statistics": true,
    "cuda_device": 2
  }
}
