# Generation Models

This guide covers the fundamental concepts and implementations of generation models in RAG systems, including language models, text generation, response formatting, and optimization techniques.

## Language Models

### Basic Language Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict, Any

class LanguageModel:
    def __init__(self, model_name: str = 'gpt2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
    
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Generate text from prompt"""
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Advanced Language Model
```python
class AdvancedLanguageModel:
    def __init__(self, model_name: str = 'gpt2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.generation_config = {
            'max_length': 100,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'num_beams': 5
        }
    
    def generate_with_config(self,
                           prompt: str,
                           config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate text with custom configuration"""
        if config:
            self.generation_config.update(config)
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=self.generation_config['max_length'],
                temperature=self.generation_config['temperature'],
                top_p=self.generation_config['top_p'],
                top_k=self.generation_config['top_k'],
                num_beams=self.generation_config['num_beams'],
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return {
            'text': self.tokenizer.decode(outputs[0], skip_special_tokens=True),
            'config': self.generation_config
        }
```

## Text Generation

### Basic Text Generator
```python
class TextGenerator:
    def __init__(self, model_name: str = 'gpt2'):
        self.model = LanguageModel(model_name)
    
    def generate_text(self,
                     prompt: str,
                     max_length: int = 100) -> str:
        """Generate text from prompt"""
        return self.model.generate(prompt, max_length)
    
    def generate_with_context(self,
                            prompt: str,
                            context: str,
                            max_length: int = 100) -> str:
        """Generate text with context"""
        full_prompt = f"Context: {context}\nPrompt: {prompt}"
        return self.model.generate(full_prompt, max_length)
```

### Advanced Text Generator
```python
class AdvancedTextGenerator:
    def __init__(self, model_name: str = 'gpt2'):
        self.model = AdvancedLanguageModel(model_name)
        self.templates = {}
    
    def add_template(self, name: str, template: str) -> None:
        """Add a text generation template"""
        self.templates[name] = template
    
    def generate_with_template(self,
                             template_name: str,
                             **kwargs) -> Dict[str, Any]:
        """Generate text using a template"""
        if template_name not in self.templates:
            raise ValueError(f"Template {template_name} not found")
        
        template = self.templates[template_name]
        prompt = template.format(**kwargs)
        
        return self.model.generate_with_config(prompt)
    
    def generate_with_retrieval(self,
                              prompt: str,
                              retrieved_docs: List[Dict[str, Any]],
                              max_length: int = 100) -> Dict[str, Any]:
        """Generate text with retrieved documents"""
        context = "\n".join([doc['document'] for doc in retrieved_docs])
        full_prompt = f"Context: {context}\nPrompt: {prompt}"
        
        return self.model.generate_with_config(full_prompt)
```

## Response Formatting

### Basic Response Formatter
```python
class ResponseFormatter:
    def __init__(self):
        self.templates = {
            'default': "{text}",
            'qa': "Question: {question}\nAnswer: {answer}",
            'summary': "Summary: {text}"
        }
    
    def format_response(self,
                       template_name: str,
                       **kwargs) -> str:
        """Format response using a template"""
        if template_name not in self.templates:
            raise ValueError(f"Template {template_name} not found")
        
        template = self.templates[template_name]
        return template.format(**kwargs)
```

### Advanced Response Formatter
```python
class AdvancedResponseFormatter:
    def __init__(self):
        self.templates = {
            'default': "{text}",
            'qa': "Question: {question}\nAnswer: {answer}",
            'summary': "Summary: {text}",
            'structured': "Title: {title}\nContent: {content}\nReferences: {references}"
        }
        self.post_processors = {}
    
    def add_template(self, name: str, template: str) -> None:
        """Add a response template"""
        self.templates[name] = template
    
    def add_post_processor(self, name: str, processor: callable) -> None:
        """Add a post-processing function"""
        self.post_processors[name] = processor
    
    def format_response(self,
                       template_name: str,
                       post_processors: List[str] = None,
                       **kwargs) -> Dict[str, Any]:
        """Format response with post-processing"""
        if template_name not in self.templates:
            raise ValueError(f"Template {template_name} not found")
        
        template = self.templates[template_name]
        formatted_text = template.format(**kwargs)
        
        result = {
            'text': formatted_text,
            'template': template_name
        }
        
        # Apply post-processors
        if post_processors:
            for processor_name in post_processors:
                if processor_name in self.post_processors:
                    result['text'] = self.post_processors[processor_name](result['text'])
                    result['post_processors'] = result.get('post_processors', []) + [processor_name]
        
        return result
```

## Generation Optimization

### Batch Generation
```python
class BatchGenerator:
    def __init__(self, model_name: str = 'gpt2'):
        self.model = LanguageModel(model_name)
        self.batch_size = 32
    
    def generate_batch(self,
                      prompts: List[str],
                      max_length: int = 100) -> List[str]:
        """Generate text for multiple prompts in batch"""
        results = []
        
        # Process in batches
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
            batch_results = [
                self.model.generate(prompt, max_length)
                for prompt in batch_prompts
            ]
            results.extend(batch_results)
        
        return results
```

### Caching
```python
class CachedGenerator:
    def __init__(self, model_name: str = 'gpt2'):
        self.model = LanguageModel(model_name)
        self.cache = {}
        self.max_cache_size = 10000
    
    def generate(self,
                prompt: str,
                max_length: int = 100) -> str:
        """Generate text with caching"""
        cache_key = f"{prompt}_{max_length}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = self.model.generate(prompt, max_length)
        
        # Update cache
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = result
        return result
```

## Best Practices

1. **Language Models**:
   - Model selection
   - Configuration tuning
   - Memory management
   - Error handling

2. **Text Generation**:
   - Prompt engineering
   - Context management
   - Template usage
   - Quality control

3. **Response Formatting**:
   - Template design
   - Post-processing
   - Error handling
   - Output validation

4. **Optimization**:
   - Batch processing
   - Caching
   - Memory management
   - Performance monitoring

## Common Patterns

1. **Generator Factory**:
```python
class GeneratorFactory:
    @staticmethod
    def create_generator(generator_type: str, **kwargs) -> Any:
        if generator_type == 'basic':
            return TextGenerator(**kwargs)
        elif generator_type == 'advanced':
            return AdvancedTextGenerator(**kwargs)
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")
```

2. **Generator Monitor**:
```python
class GeneratorMonitor:
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

3. **Generator Validator**:
```python
class GeneratorValidator:
    def __init__(self, generator: Any):
        self.generator = generator
    
    def validate_prompt(self, prompt: str) -> bool:
        """Validate prompt before generation"""
        if not isinstance(prompt, str):
            return False
        if not prompt.strip():
            return False
        return True
    
    def validate_response(self, response: str) -> bool:
        """Validate generated response"""
        if not isinstance(response, str):
            return False
        if not response.strip():
            return False
        return True
```

## Further Reading

- [Language Models](https://arxiv.org/abs/2004.07213)
- [Text Generation](https://arxiv.org/abs/2004.07213)
- [Response Formatting](https://arxiv.org/abs/2004.07213)
- [Generation Optimization](https://arxiv.org/abs/2004.07213)
- [Natural Language Generation](https://arxiv.org/abs/2004.07213) 