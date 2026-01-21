import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    """
    Embeds the input tokens using an embedding layer.
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input embeddings.
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        #Create a vector of shape (seq_len, 1)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        #Create denominator vector of shape (d_model/2, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # Apply sin on even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos on odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe) # Register as a buffer to be saved and loaded with the model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False) # (batch_size, seq_len, d_model); we clip the positional encoding to the sequence length of the input
        return self.dropout(x)

class LayerNormalization(nn.Module):
    """
        Layer Normalization to scale the activations
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)

        return self.alpha * ((x- mean) / (std + self.eps)) + self.bias

class FeedForwardBlock(nn.Module):
    """
        Feed Forward Block to transform the activations
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x is (batch_size, seq_len, d_model)
        x = torch.relu(self.linear_1(x)) # (batch_size, seq_len, d_ff)
        x = self.dropout(x)
        x = self.linear_2(x) # (batch_size, seq_len, d_model)
        return x
    
class MultiHeadAttention(nn.Module):
    """
        Multi-Head Attention mechanism
    """

    def __init__(self, d_model: int, h: int, dropout: float) -> None:

        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)

        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, dropout: nn.Dropout) -> tuple[torch.Tensor, torch.Tensor]:
        
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask==0, -torch.inf)
        attention_scores = attention_scores.softmax(dim = -1) # this takes the softmax along the key dimension

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout: nn.Dropout | None, mask: torch.Tensor | None) -> torch.Tensor:

        query = self.w_q(q) # (batch_size, seq_len, d_model)
        key = self.w_k(k) # (batch_size, seq_len, d_model)
        value = self.q_v(v) # (batch_size, seq_len, d_model)

        # Calculate the batch_size and seq_length

        batch_size, seq_len = q.shape[0], q.shape[1]

        # Transform from: (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)

        query = query.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2) # (batch_size, h, seq_len, d_k)
        key = key.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2) # (batch_size, h, seq_len, d_k)
        value = value.view(batch_size, seq_len, self.h, self.d_k).tranpose(1, 2) # (batch_size, h, seq_len, d_k)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        x = x.tranpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.w_0(x)

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))