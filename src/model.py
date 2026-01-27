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
        self.w_0 = nn.Linear(d_model, d_model)

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
        value = self.w_v(v) # (batch_size, seq_len, d_model)

        # Calculate the batch_size and seq_length

        batch_size, seq_len = q.shape[0], q.shape[1]

        # Transform from: (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)

        query = query.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2) # (batch_size, h, seq_len, d_k)
        key = key.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2) # (batch_size, h, seq_len, d_k)
        value = value.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2) # (batch_size, h, seq_len, d_k)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.w_0(x)

class ResidualConnection(nn.Module):
    """
        Residual Connections
    """

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    """
        Encoder Block
    """

    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        self.dropout = dropout

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor):

        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, self.dropout, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x

class Encoder(nn.Module):
    """
        Encoder
    """
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask) -> torch.Tensor:

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, target_mask) -> torch.Tensor:

        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, self.dropout, target_mask))
        x = self.residual_connections[1](x, lambda x: self.self_attention_block(x, encoder_output, encoder_output, self.dropout, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, target_mask) -> torch.Tensor:
        
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> torch.Tensor:
        
        x = self.proj(x)
        x = torch.log_softmax(x, dim = -1)
        return x

class Transformer(nn.Module):
    """
        The Transformer Architecture
    """
    def __init__(self, 
                encoder: Encoder, 
                decoder: Decoder, 
                src_embedding: InputEmbeddings, 
                target_embedding: InputEmbeddings, 
                src_pos: PositionalEncoding, 
                target_pos: PositionalEncoding,
                projection_layer: ProjectionLayer) -> None:
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.target_embedding = target_embedding
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask) -> torch.Tensor:
        src = self.src_embedding(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, target, target_mask) -> torch.Tensor:
        target = self.target_embedding(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_output, src_mask, target_mask)

    def project(self, x) -> torch.Tensor:
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, 
                    target_vocab_size: int, 
                    src_seq_len: int, 
                    target_seq_len: int, 
                    d_model: int = 512, 
                    N: int = 6,
                    h: int = 8,
                    dropout: float = 0.1,
                    d_ff = 2048
                    ) -> Transformer:

    """
        Transformer Architecture being built
    """

    # Create the embedding layers
    src_embedding = InputEmbeddings(d_model, src_vocab_size)
    target_embedding = InputEmbeddings(d_model, target_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    target_pos = PositionalEncoding(d_model, target_seq_len, dropout)

    # Create Encoder Blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, encoder_feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create Decoder Blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, target_vocab_size)

    transformer = Transformer(encoder, decoder, src_embedding, target_embedding, src_pos, target_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer