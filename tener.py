from typing import Callable
import math

import torch
import torch.nn.functional as F

from allennlp.modules.layer_norm import LayerNorm
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn import util


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
    input_dim : `int`, required
        Input dimension.
    """
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self._input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size()
        return self._get_pos_embeddings(2 * seq_len)

    def _get_pos_embeddings(self, seq_len: int):
        half_dim = self._input_dim // 2
        positional_encoding = torch.zeros(seq_len, self._input_dim, requires_grad=False)
        position = torch.arange(-seq_len // 2, seq_len // 2, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(half_dim, dtype=torch.float) * -(math.log(10000) / (half_dim - 1))
        )
        positional_encoding = torch.cat(
            [torch.sin(position * div_term.unsqueeze(0)),
             torch.cos(position * div_term.unsqueeze(0))],
            dim=1
        ).view(seq_len, -1)
        return positional_encoding


class PositionwiseFeedForward(torch.nn.Module):
    """Implements FFN equation."""
    def __init__(self, input_dim: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w_1 = torch.nn.Linear(input_dim, ff_dim)
        self.w_2 = torch.nn.Linear(ff_dim, input_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class TransformerEncoder(torch.nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self, layer: torch.nn.Module, num_layers: int) -> None:
        super().__init__()
        self.layers = util.clone(layer, num_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(torch.nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(torch.nn.Module):
    """
    Encoder is made up of self-attention and feed forward
    both followed by Residual Connection.
    """
    def __init__(self,
                 size: int,
                 self_attn: torch.nn.Module,
                 feed_forward: torch.nn.Module,
                 dropout: float) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = util.clone(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(torch.nn.Module):
    """
    This class implements the key-value scaled dot product attention mechanism
    detailed in the paper `Attention is all you Need` but with some changes
    improvements detailed in `TENER` paper: https://arxiv.org/pdf/1911.04474.pdf.

    Parameters
    ----------
    num_heads : `int`, required.
        The number of attention heads to use.
    input_dim : `int`, required.
        The size of the last dimension of the input tensor.
    attention_dim `int`, required.
        The total dimension of the query and key projections which comprise the
        dot product attention function. Must be divisible by `num_heads`.
    values_dim : `int`, required.
        The total dimension which the input is projected to for representing the values,
        which are combined using the attention. Must be divisible by `num_heads`.
    dropout : `float`, optional (default = `0.1`).
        The dropout probability applied to the normalised attention
        distributions.
    """
    def __init__(
        self,
        num_heads: int,
        input_dim: int,
        values_dim: int = None,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be a multiple of num_heads"
        self.input_dim = input_dim
        self.attention_dim = input_dim // num_heads
        self.num_heads = num_heads
        self.values_dim = values_dim or input_dim
        self.query_linear = torch.nn.Linear(input_dim, input_dim)
        self.value_linear = torch.nn.Linear(input_dim, values_dim)
        self.output_projection = torch.nn.Linear(values_dim, input_dim)
        self.dropout = torch.nn.Dropout(p=dropout)
        # Biases
        self.v_bias = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.zeros(self.num_heads, self.attention_dim))
        )
        self.position = PositionalEncodingTENER(self.attention_dim)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = query.size(0)
        # 0) Relative positional encoding
        position_emb = self.position(mask)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        key = key.view(batch_size, -1, self.num_heads, self.attention_dim).transpose(1, 2)
        query = self.query_linear(query).view(
            batch_size, -1, self.num_heads, self.attention_dim
        ).transpose(1, 2)
        value = self.value_linear(value).view(
            batch_size, -1, self.num_heads, self.values_dim // self.num_heads
        ).transpose(1, 2)
        # 2) Make all elements in equation 18 (except u * relative, it is not needed)
        query_key = torch.einsum('bnqd,bnkd->bnqk', query, key)
        v_bias_relative = torch.einsum('nd,ld->nl', self.v_bias, position_emb)[None, :, None]
        query_relative = torch.einsum('bnqd,ld->bnql', query, position_emb)
        query_rel_and_v_bias_rel = query_relative + v_bias_relative
        # 3) Shift last dimension
        query_rel_and_v_bias_rel = self._shift(query_rel_and_v_bias_rel)
        attn = query_key + query_rel_and_v_bias_rel
        # 4) Masked softmax
        x = self.masked_softmax(attn, value, batch_size=batch_size)
        # 5) One more linear projection
        return self.output_projection(x)

    def masked_softmax(
        self,
        attn: torch.Tensor,
        value: torch.Tensor,
        batch_size: int,
        should_scale: bool = False,
        mask: torch.Tensor = None,
        dropout: Callable = None
    ) -> torch.Tensor:
        if should_scale:
            attn = attn / math.sqrt(self.attention_dim)
        if mask is not None:
            attn = attn.masked_fill(mask[:, None, None, :].eq(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        if dropout is not None:
            attn = dropout(attn)
        return torch.matmul(attn, value).transpose(1, 2).reshape(batch_size, -1, self.values_dim)

    def _shift(self, tensor):
        """
        Shift dimensions.
        Input tensor: batch_size x num_heads x seq_len x 2seq_len.
        Return tensor: batch_size x num_heads x seq_len x seq_len.
        """
        batch_size, num_heads, seq_len, _ = tensor.size()
        zero_pad = tensor.new_zeros(batch_size, num_heads, seq_len, 1)
        # tensor ~ batch_size x num_heads x (2seq_len+1) x seq_len
        tensor = torch.cat([tensor, zero_pad], dim=-1).view(batch_size, num_heads, -1, seq_len)
        # tensor ~ batch_size x num_heads x 2seq_len x seq_len
        tensor = tensor[:, :, :-1].view(batch_size, num_heads, seq_len, -1)
        tensor = tensor[:, :, :, seq_len:]
        return tensor


def make_model(
    num_layers: int = 6,
    input_size: int = 512,  # Attention size
    hidden_size: int = 2048,  # FF layer size
    heads: int = 8,
    values_dim: int = 256,
    dropout: float = 0.1
) -> TransformerEncoder:
    """`Helper`: Construct a model from hyperparameters."""
    attn = MultiHeadedAttention(
        heads, input_size, values_dim, dropout
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
    input_dim : `int`, required
        The input dimension of the encoder.
    hidden_dim : `int`, required
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
        input_dim: int,
        hidden_dim: int,
        values_dim: int,
        heads: int,
        num_layers: int,
        output_projection_dim: int = None,
        dropout: float = 0.1,
        input_dropout: float = None,
    ) -> None:
        super().__init__()
        self.transformer_layers = num_layers
        self.num_layers = num_layers
        self.output_projection_dim = output_projection_dim or input_dim
        self.model = make_model(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            values_dim=values_dim,
            dropout=dropout
        )
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.output_projection = torch.nn.Linear(
            self.input_dim,
            self.output_projection_dim
        )
        if input_dropout:
            self.dropout = torch.nn.Dropout(input_dropout)
        else:
            self.dropout = lambda x: x
        self.should_log_activations = False

    def forward(self, token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.dropout(token_embeddings)
        model_output = self.model(token_embeddings, mask)
        return self.output_projection(model_output)

    def get_regularization_penalty(self):
        return 0.

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim

    def is_bidirectional(self) -> bool:
        return True
