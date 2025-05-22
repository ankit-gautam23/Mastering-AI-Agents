# Mixture of Experts (MoE)

This guide covers the fundamental concepts and implementations of Mixture of Experts (MoE) architecture in large language models, including routing mechanisms, expert networks, and optimization techniques.

## Basic MoE Architecture

### Simple MoE Implementation
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class MoELayer(nn.Module):
    def __init__(self, input_size, output_size, num_experts, hidden_size):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(input_size, num_experts)
        self.experts = nn.ModuleList([
            Expert(input_size, hidden_size, output_size)
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # Get routing weights
        routing_weights = F.softmax(self.router(x), dim=-1)
        
        # Get expert outputs
        expert_outputs = torch.stack([
            expert(x) for expert in self.experts
        ], dim=1)
        
        # Combine expert outputs
        return torch.sum(
            expert_outputs * routing_weights.unsqueeze(-1),
            dim=1
        )
```

### Advanced MoE Implementation
```python
class AdvancedMoELayer(nn.Module):
    def __init__(self, input_size, output_size, num_experts, hidden_size, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k  # Number of experts to route to
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts)
        )
        
        # Expert networks
        self.experts = nn.ModuleList([
            Expert(input_size, hidden_size, output_size)
            for _ in range(num_experts)
        ])
        
        # Load balancing loss
        self.aux_loss = 0.0
    
    def forward(self, x):
        # Get routing weights
        routing_logits = self.router(x)
        routing_weights = F.softmax(routing_logits, dim=-1)
        
        # Top-k routing
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, self.k, dim=-1
        )
        
        # Normalize top-k weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Get expert outputs
        expert_outputs = torch.stack([
            expert(x) for expert in self.experts
        ], dim=1)
        
        # Combine expert outputs
        final_output = torch.zeros_like(x)
        for i in range(self.k):
            expert_idx = top_k_indices[:, i]
            expert_weight = top_k_weights[:, i].unsqueeze(-1)
            expert_output = expert_outputs[torch.arange(len(x)), expert_idx]
            final_output += expert_weight * expert_output
        
        # Calculate load balancing loss
        self.aux_loss = self._load_balancing_loss(routing_weights)
        
        return final_output
    
    def _load_balancing_loss(self, routing_weights):
        # Calculate mean routing weights per expert
        mean_routing = routing_weights.mean(dim=0)
        
        # Calculate load balancing loss
        return torch.sum(mean_routing * torch.log(mean_routing + 1e-10))
```

## Routing Mechanisms

### Basic Router
```python
class BasicRouter(nn.Module):
    def __init__(self, input_size, num_experts):
        super().__init__()
        self.router = nn.Linear(input_size, num_experts)
    
    def forward(self, x):
        return F.softmax(self.router(x), dim=-1)
```

### Advanced Router
```python
class AdvancedRouter(nn.Module):
    def __init__(self, input_size, num_experts, hidden_size):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts),
            nn.Dropout(0.1)
        )
    
    def forward(self, x, temperature=1.0):
        logits = self.router(x) / temperature
        return F.softmax(logits, dim=-1)
```

## Expert Networks

### Basic Expert
```python
class BasicExpert(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.network(x)
```

### Advanced Expert
```python
class AdvancedExpert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)
```

## Optimization Techniques

### Load Balancing
```python
class LoadBalancer:
    def __init__(self, num_experts):
        self.num_experts = num_experts
        self.expert_usage = torch.zeros(num_experts)
    
    def update(self, routing_weights):
        self.expert_usage += routing_weights.mean(dim=0)
    
    def get_balance_loss(self):
        mean_usage = self.expert_usage.mean()
        return torch.sum(
            (self.expert_usage - mean_usage) ** 2
        ) / self.num_experts
```

### Expert Pruning
```python
class ExpertPruner:
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.expert_importance = {}
    
    def update(self, expert_id, importance):
        self.expert_importance[expert_id] = importance
    
    def get_pruned_experts(self):
        return [
            expert_id for expert_id, importance in self.expert_importance.items()
            if importance < self.threshold
        ]
```

## Best Practices

1. **Architecture Design**:
   - Choose appropriate number of experts
   - Balance expert capacity
   - Implement efficient routing

2. **Training Strategy**:
   - Use load balancing loss
   - Implement expert pruning
   - Monitor expert utilization

3. **Deployment Considerations**:
   - Memory management
   - Inference optimization
   - Expert distribution

4. **Performance Optimization**:
   - Batch processing
   - Expert caching
   - Load balancing

## Common Patterns

1. **MoE Factory**:
```python
class MoEFactory:
    @staticmethod
    def create_moe(moe_type: str, **kwargs) -> nn.Module:
        if moe_type == 'basic':
            return MoELayer(**kwargs)
        elif moe_type == 'advanced':
            return AdvancedMoELayer(**kwargs)
        else:
            raise ValueError(f"Unknown MoE type: {moe_type}")
```

2. **MoE Monitor**:
```python
class MoEMonitor:
    def __init__(self):
        self.metrics = {}
    
    def record_metric(self, name: str, value: float) -> None:
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_statistics(self, name: str) -> Dict[str, float]:
        values = self.metrics.get(name, [])
        if not values:
            return {}
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
```

## Further Reading

- [Mixture of Experts](https://arxiv.org/abs/2004.07213)
- [Sparse MoE](https://arxiv.org/abs/2004.07213)
- [Load Balancing](https://arxiv.org/abs/2004.07213)
- [Expert Networks](https://arxiv.org/abs/2004.07213)
- [MoE Optimization](https://arxiv.org/abs/2004.07213) 