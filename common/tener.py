from typing import Callable
import math

import torch
import torch.nn.functional as F

from allennlp.nn import util
from allennlp.modules.layer_norm import LayerNorm
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


def subsequent_mask(size: int, device: str = 'cpu') -> torch.Tensor:
    """Mask out subsequent positions."""
    mask = torch.tril(torch.ones(size, size, device=device, dtype=torch.int32)).unsqueeze(0)
    return mask


class PositionalEncodingTENER(torch.nn.Module):
    """
    Implement the Positional Encoding function.
    Like in `TENER`: https://arxiv.org/pdf/1911.04474.pdf.

    Parameters
    ----------
    input_size : `int`, required
        Input dimension.
    """
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self._input_size = input_size

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        _, timesteps = x.size()
        # (2 * timesteps) x input_size
        return self._get_pos_embeddings(2 * timesteps)

    def _get_pos_embeddings(self, timesteps: int):
        half_dim = self._input_size // 2
        position = torch.arange(-timesteps // 2, timesteps // 2, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(half_dim, dtype=torch.float) * -(math.log(10000) / (half_dim - 1))
        )
        positional_encoding = torch.cat(
            [torch.sin(position * div_term.unsqueeze(0)),
             torch.cos(position * div_term.unsqueeze(0))],
            dim=1
        ).view(timesteps, -1)
        if self._input_size % 2 == 1:
            # Zero pad as there are some problems
            # when number of timesteps is not divisible by 2
            positional_encoding = torch.cat(
                [positional_encoding, torch.zeros(timesteps, 1)],
                dim=1
            )
        return positional_encoding


class PositionwiseFeedForward(torch.nn.Module):
    """Implements FFN equation."""
    def __init__(self, input_size: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self._w_1 = torch.nn.Linear(input_size, ff_dim)
        self._w_2 = torch.nn.Linear(ff_dim, input_size)
        self._dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return self._w_2(self._dropout(F.relu(self._w_1(x))))


class TransformerEncoder(torch.nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self, layer: torch.nn.Module, num_layers: int) -> None:
        super().__init__()
        self._layers = util.clone(layer, num_layers)
        self._norm = LayerNorm(layer._size)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Pass the input (and mask) through each layer in turn."""
        for layer in self._layers:
            x = layer(x, mask)
        return self._norm(x)


class SublayerConnection(torch.nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size: int, dropout: float) -> None:
        super().__init__()
        self._norm = LayerNorm(size)
        self._dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        sublayer: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """Apply residual connection to any sublayer with the same size."""
        return x + self._dropout(sublayer(self._norm(x)))


class EncoderLayer(torch.nn.Module):
    """
    Encoder is made up of self-attention and feedforward
    both followed by Residual Connection.
    """
    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: torch.nn.Module,
        dropout: float
    ) -> None:
        super().__init__()
        self._self_attn = self_attn
        self._feed_forward = feed_forward
        self._sublayer = util.clone(SublayerConnection(size, dropout), 2)
        self._size = size

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        x = self._sublayer[0](x, lambda x: self._self_attn(x, x, x, mask))
        return self._sublayer[1](x, self._feed_forward)


class MultiHeadSelfAttentionTENER(torch.nn.Module):
    """
    This class implements the key-value scaled dot product attention mechanism
    detailed in the paper `Attention is all you Need` but with some changes
    improvements detailed in `TENER` paper: https://arxiv.org/pdf/1911.04474.pdf.

    Parameters
    ----------
    num_heads : `int`, required.
        The number of attention heads to use.
    input_size : `int`, required.
        The size of the last dimension of the input tensor.
    attention_dim `int`, required.
        The total dimension of the query and key projections which comprise the
        dot product attention function. Must be divisible by `num_heads`.
    values_size : `int`, required.
        The total dimension which the input is projected to for representing the values,
        which are combined using the attention. Must be divisible by `num_heads`.
    dropout : `float`, optional (default = `0.1`).
        The dropout probability applied to the normalised attention
        distributions. If None, dropout probability is not applied.
    """
    def __init__(
        self,
        num_heads: int,
        input_size: int,
        values_size: int = None,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        assert input_size % num_heads == 0, "input_size must be a multiple of num_heads"
        if values_size:
            assert values_size % num_heads == 0, "values_size must be a multiple of num_heads"
        self._input_size = input_size
        self._attention_dim = input_size // num_heads
        self._num_heads = num_heads
        self._values_size = values_size or input_size
        self._query_linear = torch.nn.Linear(input_size, input_size)
        self._value_linear = torch.nn.Linear(input_size, values_size)
        self._output_projection = torch.nn.Linear(values_size, input_size)
        if dropout:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        # Biases
        self._v_bias = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.zeros(self._num_heads, self._attention_dim))
        )
        # Positional Encoding
        self._position = PositionalEncodingTENER(self._attention_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size = query.size(0)
        # 0) Relative positional encoding
        position_emb = self._position(mask).to(query.device)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        key = key.view(batch_size, -1, self._num_heads, self._attention_dim).transpose(1, 2)
        query = self._query_linear(query).view(
            batch_size, -1, self._num_heads, self._attention_dim
        ).transpose(1, 2)
        value = self._value_linear(value).view(
            batch_size, -1, self._num_heads, self._values_size // self._num_heads
        ).transpose(1, 2)
        # 2) Make all elements in equation 18 (except u_bias * relative, it is not needed)
        # query_key ~ batch_size x num_heads x timesteps x timesteps
        query_key = torch.einsum('bnqd,bnkd->bnqk', query, key)
        query_relative = torch.einsum('bnqd,ld->bnql', query, position_emb)
        v_bias_relative = torch.einsum('nd,ld->nl', self._v_bias, position_emb)[None, :, None]
        query_rel_and_v_bias_rel = query_relative + v_bias_relative
        # 3) Shift last dimension
        query_rel_and_v_bias_rel = self._shift(query_rel_and_v_bias_rel)
        attn = query_key + query_rel_and_v_bias_rel
        # 4) Masked softmax for Attention
        attn_softmax = self._masked_softmax(attn, mask)
        # 5) Multiply Softmax(Attention) with value
        x = torch.einsum('bnkq,bnqv->bknv', attn_softmax, value).reshape(batch_size, -1, self._values_size)
        # 6) One more linear projection
        return self._output_projection(x)

    def _masked_softmax(
        self,
        attn: torch.Tensor,
        mask: torch.Tensor = None,
        should_scale: bool = False
    ) -> torch.Tensor:
        """
        Compute Softmax with mask and
        perform additional attention scaling if needed.
        """
        if should_scale:
            attn = attn / math.sqrt(self._attention_dim)
        if mask is not None:
            attn = attn.masked_fill(mask[:, None, None, :].eq(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        # Dropout after softmax
        attn = self._dropout(attn)
        return attn

    def _shift(self, tensor):
        """
        Shift dimensions.
        Input tensor: batch_size x num_heads x timesteps x 2 * timesteps.
        Return tensor: batch_size x num_heads x timesteps x timesteps.
        """
        batch_size, num_heads, timesteps, _ = tensor.size()
        zero_pad = tensor.new_zeros(batch_size, num_heads, timesteps, 1)
        # tensor ~ batch_size x num_heads x (2 * timesteps + 1) x timesteps
        tensor = torch.cat([tensor, zero_pad], dim=-1).view(batch_size, num_heads, -1, timesteps)
        # tensor ~ batch_size x num_heads x timesteps x (2 * timesteps)
        tensor = tensor[:, :, :-1].view(batch_size, num_heads, timesteps, -1)
        tensor = tensor[:, :, :, timesteps:]
        return tensor


def make_model(
    num_layers: int = 6,
    input_size: int = 512,  # Attention size
    hidden_size: int = 2048,  # FF layer size
    heads: int = 8,
    values_size: int = 256,
    dropout: float = 0.1
) -> TransformerEncoder:
    """`Helper`: Construct a model from hyperparameters."""
    attn = MultiHeadSelfAttentionTENER(
        heads, input_size, values_size, dropout
    )
    ff = PositionwiseFeedForward(input_size, hidden_size, dropout)
    model = TransformerEncoder(
        EncoderLayer(input_size, attn, ff, dropout),
        num_layers
    )
    return model


@Seq2SeqEncoder.register('tener')
class TENER(Seq2SeqEncoder):
    """
    Implementation of TENER for Named-Entity Recognition.

    Parameters
    ----------
    input_size : `int`, required
        The input dimension of the encoder.
    hidden_size : `int`, required
        The hidden dimension used for the _input_ to self-attention layers
        and the _output_ from the feedforward layers.
    heads : `int`, required
        Number of Attention Heads.
    num_layers : `int`, required
        Number of self-attention layers.
    output_projection_dim : `int`, optional (default = `None`)
        The dimensionality of the final output projection. If this is not passed
        explicitly, the projection has size `input_size`.
    dropout : `float`, optional (default = `0.1`)
        The dropout probability for the attention distributions
        in each attention layer and feedforward network.
    input_dropout : `float`, optional (default = `None`)
        Whether to use input dropout or not.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        values_size: int,
        heads: int,
        num_layers: int,
        output_projection_dim: int = None,
        dropout: float = 0.1,
        input_dropout: float = None,
    ) -> None:
        super().__init__()
        self._transformer_layers = num_layers
        self._num_layers = num_layers
        self._model = make_model(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            heads=heads,
            values_size=values_size,
            dropout=dropout
        )
        self._input_size = input_size
        self._output_dim = output_projection_dim or input_size
        self._output_projection = torch.nn.Linear(
            self._input_size,
            self._output_dim
        )
        if input_dropout:
            self._dropout = torch.nn.Dropout(input_dropout)
        else:
            self._dropout = lambda x: x
        self._should_log_activations = False

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, timesteps, _ = inputs.size()
        if mask is None:
            mask = inputs.new_ones(batch_size, timesteps)
        inputs = self._dropout(inputs)
        model_output = self._model(inputs, mask)
        return self._output_projection(model_output)

    def get_regularization_penalty(self):
        return 0.

    def get_input_dim(self) -> int:
        return self._input_size

    def get_output_dim(self) -> int:
        return self._output_dim

    def is_bidirectional(self) -> bool:
        return True
