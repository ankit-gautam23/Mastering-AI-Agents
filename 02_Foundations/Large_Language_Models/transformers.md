# Transformers

This guide covers the fundamental concepts and implementation of transformer models, which are the backbone of modern Large Language Models.

## Architecture Overview

### Basic Transformer Architecture
```python
import torch
import torch.nn as nn
import math

class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512
    ):
        super().__init__()
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Get sequence length
        seq_length = x.size(1)
        
        # Create position indices
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0)
        
        # Get embeddings
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)
        
        # Combine embeddings
        x = token_embeddings + position_embeddings
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final normalization and output
        x = self.norm(x)
        return self.output(x)
```

## Core Components

### Multi-Head Attention
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size = q.size(0)
        
        # Linear projections and reshape
        q = self.q_proj(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Combine heads
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out_proj(output)
```

### Position-wise Feed Forward
```python
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
```

### Transformer Block
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

## Advanced Features

### Relative Position Encoding
```python
class RelativePositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_relative_position: int = 32):
        super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # Create relative position embeddings
        self.relative_attention_bias = nn.Parameter(
            torch.randn(2 * max_relative_position + 1)
        )
    
    def compute_bias(self, length: int) -> torch.Tensor:
        # Create relative position matrix
        context_position = torch.arange(length, dtype=torch.long)[:, None]
        memory_position = torch.arange(length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        
        # Clip relative positions
        relative_position_bucket = relative_position + self.max_relative_position
        relative_position_bucket = torch.clamp(
            relative_position_bucket,
            0,
            2 * self.max_relative_position
        )
        
        return self.relative_attention_bias[relative_position_bucket]
```

### Rotary Position Embedding
```python
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Generate position indices
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        
        # Compute sin and cos embeddings
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)]
```

## Best Practices

1. **Architecture Design**:
   - Use proper initialization
   - Implement residual connections
   - Apply layer normalization
   - Use dropout for regularization

2. **Attention Mechanisms**:
   - Scale attention scores
   - Use proper masking
   - Implement relative positions
   - Consider memory efficiency

3. **Position Embeddings**:
   - Choose appropriate type
   - Consider sequence length
   - Handle variable lengths
   - Use learned embeddings

4. **Training**:
   - Use learning rate warmup
   - Implement gradient clipping
   - Monitor attention patterns
   - Regularize appropriately

## Common Patterns

1. **Attention Masking**:
```python
def create_attention_mask(seq_length: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
    return mask == 0
```

2. **Position Embedding**:
```python
def create_position_embedding(seq_length: int, d_model: int) -> torch.Tensor:
    position = torch.arange(seq_length).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_length, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
```

3. **Model Initialization**:
```python
def initialize_transformer(model: nn.Module):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
```

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Transformer Architecture](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Relative Positional Encoding](https://arxiv.org/abs/1803.02155) 