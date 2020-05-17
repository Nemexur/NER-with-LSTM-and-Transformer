###################################################################################
# =================================================================================
# Custom DatasetReader for Conll2003
# =================================================================================
###################################################################################


from typing import (
    Dict, List, Iterable, Sequence
)
import itertools
import logging
from overrides import overrides
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.dataset_readers import (
    Conll2003DatasetReader, DatasetReader
)
from allennlp.data.fields import (
    TextField, SequenceLabelField, Field, MetadataField
)

logger = logging.getLogger(__name__)


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        if first_token == "-DOCSTART-":
            return True
        else:
            return False


@DatasetReader.register("conll2003_ner")
class Conll2003NERDatasetReader(Conll2003DatasetReader):
    """
    Reads instances from a pretokenized file where each line is in the following format:
    WORD POS-TAG CHUNK-TAG NER-TAG
    with a blank line indicating the end of each sentence
    and '-DOCSTART- -X- -X- O' indicating the end of each article,
    and converts it into a ``Dataset`` suitable for sequence tagging.
    Each ``Instance`` contains the words in the ``"tokens"`` ``TextField``.
    The values corresponding to the ``tag_label``
    values will get loaded into the ``"tags"`` ``SequenceLabelField``.
    And if you specify any ``feature_labels`` (you probably shouldn't),
    the corresponding values will get loaded into their own ``SequenceLabelField`` s.
    This dataset reader ignores the "article" divisions and simply treats
    each sentence as an independent ``Instance``. (Technically the reader splits sentences
    on any combination of blank lines and "DOCSTART" tags; in particular, it does the right
    thing on well formed inputs.)

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    tag_label: ``str``, optional (default=``ner``)
        Specify `ner`, `pos`, or `chunk` to have that tag loaded into the instance field `tag`.
    feature_labels: ``Sequence[str]``, optional (default=``()``)
        These labels will be loaded as features into the corresponding instance fields:
        ``pos`` -> ``pos_tags``, ``chunk`` -> ``chunk_tags``, ``ner`` -> ``ner_tags``
        Each will have its own namespace: ``pos_tags``, ``chunk_tags``, ``ner_tags``.
        If you want to use one of the tags as a `feature` in your model, it should be
        specified here.
    coding_scheme: ``str``, optional (default=``IOB1``)
        Specifies the coding scheme for ``ner_labels`` and ``chunk_labels``.
        Valid options are ``IOB1`` and ``BIOUL``.  The ``IOB1`` default maintains
        the original IOB1 scheme in the CoNLL 2003 NER data.
        In the IOB1 scheme, I is a token inside a span, O is a token outside
        a span and B is the beginning of span immediately following another
        span of the same type.
    label_namespace: ``str``, optional (default=``labels``)
        Specifies the namespace for the chosen ``tag_label``.
    """
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        tag_label: str = "ner",
        feature_labels: Sequence[str] = (),
        lazy: bool = False,
        coding_scheme: str = "IOB1",
        label_namespace: str = "labels"
    ) -> None:
        super().__init__(
            token_indexers=token_indexers,
            tag_label=tag_label,
            feature_labels=feature_labels,
            coding_scheme=coding_scheme,
            label_namespace=label_namespace,
            lazy=lazy
        )

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            # Group into alternative divider / sentence chunks.
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    fields = [line.strip().split() for line in lines]
                    # unzipping trick returns tuples, but our Fields need lists
                    fields = [list(field) for field in zip(*fields)]
                    tokens_, ner_tags = fields
                    # TextField requires ``Token`` objects
                    tokens = [Token(token) for token in tokens_]
                    yield self.text_to_instance(tokens, ner_tags)

    def text_to_instance(
        self,
        tokens: List[Token],
        ner_tags: List[str] = None
    ) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # tokens, ner_tags = self._build_sentence_piece_tokens_and_tags(tokens, ner_tags)
        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {'tokens': sequence}
        instance_fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})
        coded_ner = ner_tags
        if 'ner' in self.feature_labels:
            if coded_ner is None:
                raise ConfigurationError(
                    "Dataset reader was specified to use NER tags as "
                    " features. Pass them to text_to_instance."
                )
            instance_fields['ner_tags'] = SequenceLabelField(coded_ner, sequence, "ner_tags")
        if self.tag_label == 'ner' and coded_ner is not None:
            instance_fields['tags'] = SequenceLabelField(coded_ner, sequence, self.label_namespace)
        return Instance(instance_fields)


###################################################################################
# =================================================================================
# Custom Encoder to compose several encoders
# =================================================================================
###################################################################################


from overrides import overrides
import torch
from torch.nn import ModuleList
from typing import List

from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


@Seq2SeqEncoder.register("compose_encoders")
class ComposeEncoder(Seq2SeqEncoder):

    """This class can be used to compose several encoders in sequence.
    Among other things, this can be used to add a "pre-contextualizer" before a Seq2SeqEncoder.
    # Parameters
    encoders : `List[Seq2SeqEncoder]`, required.
        A non-empty list of encoders to compose. The encoders must match in bidirectionality.
    """

    def __init__(self, encoders: List[Seq2SeqEncoder]):
        super().__init__()
        self.encoders = ModuleList([])
        for enc in encoders:
            self.encoders.append(enc)

        # Compute bidirectionality.
        any_bidirectional = any(encoder.is_bidirectional() for encoder in encoders)
        self.bidirectional = any_bidirectional

        if len(self.encoders) < 1:
            raise ValueError("Need at least one encoder.")

        last_enc = None
        for enc in encoders:
            if last_enc is not None and last_enc.get_output_dim() != enc.get_input_dim():
                raise ValueError("Encoder input and output dimensions don't match.")
            last_enc = enc

    @overrides
    def forward(
        self,  # pylint: disable=arguments-differ
        inputs: torch.Tensor,
        mask: torch.LongTensor = None,
    ) -> torch.Tensor:
        """
        # Parameters
        inputs : `torch.Tensor`, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : `torch.LongTensor`, optional (default = None).
            A tensor of shape (batch_size, timesteps).
        # Returns
        A tensor computed by composing the sequence of encoders.
        """
        for encoder in self.encoders:
            inputs = encoder(inputs, mask)
        return inputs

    @overrides
    def get_input_dim(self) -> int:
        return self.encoders[0].get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return self.encoders[-1].get_output_dim()

    @overrides
    def is_bidirectional(self) -> bool:
        return self.bidirectional


###################################################################################
# =================================================================================
# Custom LR Scheduler
# =================================================================================
###################################################################################


from typing import Any, Dict, List

from overrides import overrides
import torch

from allennlp.training.optimizers import Optimizer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler


class _PyTorchLearningRateSchedulerWrapper(LearningRateScheduler):
    def __init__(self, lr_scheduler: torch.optim.lr_scheduler._LRScheduler) -> None:
        self.lr_scheduler = lr_scheduler

    def get_values(self):
        return self.lr_scheduler.get_lr()

    @overrides
    def step(self, metric: float = None, epoch: int = None) -> None:
        self.lr_scheduler.step(epoch)

    @overrides
    def state_dict(self) -> Dict[str, Any]:
        return self.lr_scheduler.state_dict()

    @overrides
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.lr_scheduler.load_state_dict(state_dict)


class _PyTorchLearningRateSchedulerWithMetricsWrapper(_PyTorchLearningRateSchedulerWrapper):
    @overrides
    def step(self, metric: float = None, epoch: int = None) -> None:
        if metric is None:
            raise ConfigurationError(
                "This learning rate scheduler requires "
                "a validation metric to compute the schedule and therefore "
                "must be used with a validation dataset."
            )
        self.lr_scheduler.step(metric, epoch)


@LearningRateScheduler.register("exponential_from_epoch")
class ExponentialLearningRateSchedulerFromEpoch(_PyTorchLearningRateSchedulerWrapper):
    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float = 0.1,
        from_epoch: int = 0,
        last_epoch: int = -1
    ) -> None:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=gamma, last_epoch=last_epoch
        )
        super().__init__(lr_scheduler)
        self._from_epoch = from_epoch

    @overrides
    def step(self, metric: float = None, epoch: int = None) -> None:
        if epoch >= self._from_epoch:
            self.lr_scheduler.step(epoch - self._from_epoch)
        else:
            self.lr_scheduler.step(0)


###################################################################################
# =================================================================================
# Custom Predictor for NER with JustSpaces Tokenizer
# =================================================================================
###################################################################################


from typing import List, Dict

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.predictors.predictor import Predictor


@Predictor.register("sentence-tagger-ner")
class SentenceTaggerPredictorNER(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single set of tags for it.  In particular, it can be used with
    the [`CrfTagger`](../models/crf_tagger.md) model
    and also the [`SimpleTagger`](../models/simple_tagger.md) model.

    ``P.S.``: For words tokenization is uses ``JustSpacesWordSplitter`` from ``word_splitter``
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"sentence": "..."}`.
        Runs the underlying model, and adds the `"words"` to the output.
        """
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance(tokens)


###################################################################################
# =================================================================================
# TENER (Transformer Encoder for Named Entity Recognition)
# =================================================================================
###################################################################################


from common.tener import *  # noqa: F401,F403
