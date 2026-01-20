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