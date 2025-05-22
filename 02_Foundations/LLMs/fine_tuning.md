# Fine-tuning Large Language Models

This guide covers the fundamental concepts and implementations of fine-tuning large language models, including techniques, optimization strategies, and best practices.

## Basic Fine-tuning

### Simple Fine-tuning Implementation
```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings.input_ids)

def fine_tune_model(
    model_name: str,
    train_texts: list,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5
):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare dataset
    dataset = TextDataset(train_texts, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    
    # Train model
    trainer.train()
    
    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
```

## Advanced Fine-tuning Techniques

### LoRA (Low-Rank Adaptation)
```python
from peft import get_peft_model, LoraConfig, TaskType

def apply_lora(
    model,
    r: int = 8,
    lora_alpha: int = 32,
    target_modules: list = ["q_proj", "v_proj"]
):
    # Define LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    return model
```

### QLoRA (Quantized LoRA)
```python
from transformers import BitsAndBytesConfig
import torch

def prepare_qlora_model(
    model_name: str,
    r: int = 8,
    lora_alpha: int = 32
):
    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    # Apply LoRA
    model = apply_lora(model, r=r, lora_alpha=lora_alpha)
    return model
```

## Training Strategies

### Gradient Checkpointing
```python
def enable_gradient_checkpointing(model):
    model.gradient_checkpointing_enable()
    return model

def configure_training_args(
    output_dir: str,
    gradient_checkpointing: bool = True,
    **kwargs
):
    return TrainingArguments(
        output_dir=output_dir,
        gradient_checkpointing=gradient_checkpointing,
        **kwargs
    )
```

### Mixed Precision Training
```python
def configure_mixed_precision(
    fp16: bool = True,
    bf16: bool = False
):
    return {
        "fp16": fp16,
        "bf16": bf16
    }
```

## Evaluation and Monitoring

### Training Metrics
```python
from torchmetrics import Perplexity
import wandb

class TrainingMonitor:
    def __init__(self):
        self.perplexity = Perplexity()
        self.metrics = {}
    
    def update_metrics(self, loss, predictions, labels):
        # Update perplexity
        self.perplexity.update(predictions, labels)
        
        # Log metrics
        self.metrics.update({
            "loss": loss,
            "perplexity": self.perplexity.compute()
        })
        
        # Log to wandb
        wandb.log(self.metrics)
```

### Model Evaluation
```python
def evaluate_model(model, eval_dataset, tokenizer):
    # Initialize metrics
    metrics = {
        "perplexity": Perplexity(),
        "accuracy": Accuracy()
    }
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        for batch in eval_dataset:
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            
            # Update metrics
            for metric in metrics.values():
                metric.update(predictions, batch["labels"])
    
    # Compute final metrics
    return {
        name: metric.compute()
        for name, metric in metrics.items()
    }
```

## Best Practices

1. **Data Preparation**:
   - Clean and preprocess data
   - Balance dataset
   - Use appropriate tokenization

2. **Model Selection**:
   - Choose appropriate base model
   - Consider model size
   - Evaluate resource requirements

3. **Training Configuration**:
   - Set appropriate learning rate
   - Use learning rate scheduling
   - Implement early stopping

4. **Optimization Techniques**:
   - Use gradient accumulation
   - Implement mixed precision
   - Apply gradient checkpointing

## Common Patterns

1. **Fine-tuning Pipeline**:
```python
class FineTuningPipeline:
    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
    
    def prepare_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        return self
    
    def prepare_data(self, train_texts: list):
        self.dataset = TextDataset(train_texts, self.tokenizer)
        return self
    
    def configure_training(self, **kwargs):
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            **kwargs
        )
        return self
    
    def train(self):
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset
        )
        trainer.train()
        return self
    
    def save(self):
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        return self
```

2. **Model Checkpointing**:
```python
class ModelCheckpointer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.best_metric = float('inf')
    
    def save_checkpoint(self, model, metric: float):
        if metric < self.best_metric:
            self.best_metric = metric
            model.save_pretrained(
                f"{self.output_dir}/best_model"
            )
    
    def load_best_model(self, model):
        return model.from_pretrained(
            f"{self.output_dir}/best_model"
        )
```

## Further Reading

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
- [Parameter-Efficient Fine-tuning](https://arxiv.org/abs/2106.09685)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) 