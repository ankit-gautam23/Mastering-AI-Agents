# Training and Fine-tuning

This guide covers the training and fine-tuning of Large Language Models, including pre-training, fine-tuning, parameter-efficient methods, and training infrastructure.

## Pre-training

### Basic Pre-training Loop
```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

class PreTrainer:
    def __init__(self, model_name: str, learning_rate: float = 1e-4):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            labels=batch["labels"].to(self.device)
        )
        
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, train_dataloader: DataLoader, num_epochs: int):
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_dataloader:
                loss = self.train_step(batch)
                total_loss += loss
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
```

### Distributed Training
```python
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

class DistributedTrainer:
    def __init__(self, model_name: str, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
        
        # Initialize distributed environment
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank
        )
        
        # Load model and move to GPU
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device(f"cuda:{rank}")
        self.model.to(self.device)
        
        # Wrap model for distributed training
        self.model = DistributedDataParallel(
            self.model,
            device_ids=[rank],
            output_device=rank
        )
    
    def train(self, train_dataset, batch_size: int, num_epochs: int):
        # Create distributed sampler
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank
        )
        
        # Create dataloader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler
        )
        
        # Training loop
        for epoch in range(num_epochs):
            sampler.set_epoch(epoch)
            for batch in train_dataloader:
                self.train_step(batch)
```

## Fine-tuning

### Full Fine-tuning
```python
class FineTuner:
    def __init__(self, model_name: str, learning_rate: float = 2e-5):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def fine_tune(self, train_dataset, eval_dataset, num_epochs: int):
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=8)
        
        best_eval_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch in train_dataloader:
                loss = self.train_step(batch)
                train_loss += loss
            
            # Evaluation
            eval_loss = self.evaluate(eval_dataloader)
            
            # Save best model
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                self.save_checkpoint(f"best_model_epoch_{epoch}")
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Training Loss: {train_loss/len(train_dataloader):.4f}")
            print(f"Evaluation Loss: {eval_loss:.4f}")
    
    def evaluate(self, eval_dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["labels"].to(self.device)
                )
                total_loss += outputs.loss.item()
        
        return total_loss / len(eval_dataloader)
```

## Parameter-Efficient Methods

### LoRA (Low-Rank Adaptation)
```python
from peft import get_peft_model, LoraConfig, TaskType

class LoRATrainer:
    def __init__(self, model_name: str, learning_rate: float = 1e-4):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        
        # Get PEFT model
        self.model = get_peft_model(self.model, peft_config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def train(self, train_dataset, num_epochs: int):
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_dataloader:
                loss = self.train_step(batch)
                total_loss += loss
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
```

### Prefix Tuning
```python
from peft import PrefixTuningConfig

class PrefixTuner:
    def __init__(self, model_name: str, num_virtual_tokens: int = 20):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Configure prefix tuning
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=num_virtual_tokens,
            prefix_projection=True
        )
        
        # Get PEFT model
        self.model = get_peft_model(self.model, peft_config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
```

## Training Infrastructure

### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.scaler = GradScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            outputs = self.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                labels=batch["labels"].to(self.device)
            )
            loss = outputs.loss
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
```

### Gradient Checkpointing
```python
class CheckpointTrainer:
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.gradient_checkpointing_enable()  # Enable gradient checkpointing
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
```

## Best Practices

1. **Pre-training**:
   - Use large batch sizes
   - Implement gradient accumulation
   - Monitor loss curves
   - Save checkpoints regularly

2. **Fine-tuning**:
   - Start with small learning rate
   - Use validation set
   - Implement early stopping
   - Save best model

3. **Parameter-Efficient Methods**:
   - Choose appropriate method
   - Tune hyperparameters
   - Monitor memory usage
   - Validate performance

4. **Training Infrastructure**:
   - Use mixed precision
   - Enable gradient checkpointing
   - Implement distributed training
   - Monitor GPU usage

## Common Patterns

1. **Learning Rate Scheduling**:
```python
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

2. **Gradient Clipping**:
```python
def clip_gradients(model, max_norm: float = 1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
```

3. **Model Checkpointing**:
```python
def save_checkpoint(model, optimizer, epoch: int, path: str):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
```

## Further Reading

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/training)
- [PEFT Documentation](https://huggingface.co/docs/peft/index)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/notes/amp_examples.html) 