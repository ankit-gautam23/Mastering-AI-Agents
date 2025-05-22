# Model Architecture

This guide covers the fundamental architecture of Large Language Models, focusing on transformer architecture, attention mechanisms, model components, and scaling laws.

## Transformer Architecture

### Basic Transformer Block
```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
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

### Multi-Head Attention
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = q.size(0)
        
        # Linear projections
        q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Combine heads
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out_linear(output)
```

## Attention Mechanisms

### Scaled Dot-Product Attention
```python
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attention = torch.softmax(scores, dim=-1)
    return torch.matmul(attention, v)
```

### Relative Position Attention
```python
class RelativePositionAttention(nn.Module):
    def __init__(self, d_model: int, max_relative_position: int):
        super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        self.relative_attention_bias = nn.Parameter(
            torch.randn(2 * max_relative_position + 1)
        )
    
    def compute_bias(self, length: int) -> torch.Tensor:
        context_position = torch.arange(length, dtype=torch.long)[:, None]
        memory_position = torch.arange(length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        
        relative_position_bucket = relative_position + self.max_relative_position
        relative_position_bucket = torch.clamp(relative_position_bucket, 0, 2 * self.max_relative_position)
        
        return self.relative_attention_bias[relative_position_bucket]
```

## Model Components

### Position-wise Feed Forward
```python
class FeedForward(nn.Module):
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

### Layer Normalization
```python
class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

## Scaling Laws

### Model Scaling
```python
class ModelScaling:
    def __init__(self, base_model_size: int, scaling_factor: float):
        self.base_model_size = base_model_size
        self.scaling_factor = scaling_factor
    
    def compute_model_size(self, num_layers: int, d_model: int, n_heads: int) -> int:
        # Compute total parameters
        attention_params = 4 * d_model * d_model  # Q, K, V, O matrices
        ff_params = 2 * d_model * (4 * d_model)  # Two linear layers
        layer_params = attention_params + ff_params
        total_params = layer_params * num_layers
        
        return total_params
    
    def scale_model(self, target_params: int) -> Dict[str, int]:
        # Compute scaling factors
        scale = (target_params / self.base_model_size) ** (1/3)
        
        # Scale dimensions
        d_model = int(self.base_model_size * scale)
        n_heads = max(1, int(d_model / 64))  # Keep head dimension at 64
        num_layers = int(12 * scale)  # Base model has 12 layers
        
        return {
            "d_model": d_model,
            "n_heads": n_heads,
            "num_layers": num_layers
        }
```

### Training Scaling
```python
class TrainingScaling:
    def __init__(self, base_batch_size: int, base_learning_rate: float):
        self.base_batch_size = base_batch_size
        self.base_learning_rate = base_learning_rate
    
    def compute_training_params(self, model_size: int) -> Dict[str, float]:
        # Scale batch size and learning rate
        batch_size = self.base_batch_size * (model_size / self.base_model_size) ** 0.5
        learning_rate = self.base_learning_rate * (model_size / self.base_model_size) ** 0.5
        
        return {
            "batch_size": batch_size,
            "learning_rate": learning_rate
        }
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

3. **Model Scaling**:
   - Follow scaling laws
   - Balance model size
   - Consider hardware constraints
   - Monitor performance

## Common Patterns

1. **Transformer Block Pattern**:
```python
def create_transformer_block(d_model: int, n_heads: int) -> nn.Module:
    return nn.Sequential(
        MultiHeadAttention(d_model, n_heads),
        LayerNorm(d_model),
        FeedForward(d_model, 4 * d_model),
        LayerNorm(d_model)
    )
```

2. **Model Initialization**:
```python
def initialize_model(model: nn.Module):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
```

3. **Attention Masking**:
```python
def create_attention_mask(seq_length: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
    return mask == 0
```

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Transformer Architecture](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [DeepSpeed Documentation](https://www.deepspeed.ai/) 