# Deployment and Scaling

This guide covers the deployment and scaling of Large Language Models, including model serving, load balancing, cost optimization, and monitoring.

## Model Serving

### Basic Model Server
```python
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn

class ModelServer:
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    async def generate(self, prompt: str, max_length: int = 100) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# FastAPI application
app = FastAPI()
model_server = ModelServer("gpt2")

@app.post("/generate")
async def generate_text(prompt: str):
    try:
        response = await model_server.generate(prompt)
        return {"generated_text": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Batch Processing
```python
class BatchModelServer:
    def __init__(self, model_name: str, batch_size: int = 8):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.request_queue = []
    
    async def process_batch(self):
        if not self.request_queue:
            return
        
        # Prepare batch
        prompts = [req["prompt"] for req in self.request_queue[:self.batch_size]]
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate responses
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=100,
                num_return_sequences=1
            )
        
        # Process responses
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Update request queue
        for req, resp in zip(self.request_queue[:self.batch_size], responses):
            req["future"].set_result(resp)
        
        self.request_queue = self.request_queue[self.batch_size:]
```

## Load Balancing

### Round Robin Load Balancer
```python
class RoundRobinLoadBalancer:
    def __init__(self, model_servers: List[ModelServer]):
        self.servers = model_servers
        self.current_index = 0
    
    def get_next_server(self) -> ModelServer:
        server = self.servers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.servers)
        return server
```

### Weighted Load Balancer
```python
class WeightedLoadBalancer:
    def __init__(self, model_servers: List[ModelServer], weights: List[float]):
        self.servers = model_servers
        self.weights = weights
        self.total_weight = sum(weights)
    
    def get_next_server(self) -> ModelServer:
        # Select server based on weights
        r = random.random() * self.total_weight
        cumsum = 0
        for server, weight in zip(self.servers, self.weights):
            cumsum += weight
            if r <= cumsum:
                return server
        return self.servers[-1]
```

## Cost Optimization

### Model Quantization
```python
class QuantizedModel:
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Quantize model
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
    
    def save_quantized(self, path: str):
        torch.save(self.quantized_model.state_dict(), path)
    
    def load_quantized(self, path: str):
        self.quantized_model.load_state_dict(torch.load(path))
```

### Model Pruning
```python
class PrunedModel:
    def __init__(self, model_name: str, sparsity: float = 0.5):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.sparsity = sparsity
    
    def prune_model(self):
        # Prune model weights
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(
                    module,
                    name='weight',
                    amount=self.sparsity
                )
    
    def save_pruned(self, path: str):
        torch.save(self.model.state_dict(), path)
```

## Monitoring

### Performance Metrics
```python
class ModelMonitor:
    def __init__(self):
        self.metrics = {
            "latency": [],
            "throughput": [],
            "memory_usage": [],
            "error_rate": []
        }
    
    def record_metrics(self, latency: float, throughput: float, memory: float, errors: int):
        self.metrics["latency"].append(latency)
        self.metrics["throughput"].append(throughput)
        self.metrics["memory_usage"].append(memory)
        self.metrics["error_rate"].append(errors)
    
    def get_average_metrics(self) -> Dict[str, float]:
        return {
            metric: sum(values) / len(values)
            for metric, values in self.metrics.items()
        }
```

### Health Checks
```python
class HealthChecker:
    def __init__(self, model_server: ModelServer):
        self.model_server = model_server
        self.last_check = time.time()
        self.health_status = True
    
    async def check_health(self) -> bool:
        try:
            # Test generation
            response = await self.model_server.generate("test")
            
            # Check memory usage
            memory_usage = torch.cuda.memory_allocated() / 1024**2
            
            # Update status
            self.health_status = (
                response is not None and
                memory_usage < 10000  # 10GB limit
            )
            
            self.last_check = time.time()
            return self.health_status
            
        except Exception as e:
            self.health_status = False
            return False
```

## Best Practices

1. **Model Serving**:
   - Use async processing
   - Implement batching
   - Handle errors gracefully
   - Monitor resource usage

2. **Load Balancing**:
   - Consider server capacity
   - Monitor server health
   - Implement failover
   - Use appropriate strategy

3. **Cost Optimization**:
   - Use model quantization
   - Implement pruning
   - Cache responses
   - Optimize batch size

4. **Monitoring**:
   - Track key metrics
   - Set up alerts
   - Monitor resources
   - Log errors

## Common Patterns

1. **Response Caching**:
```python
class ResponseCache:
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[str]:
        return self.cache.get(key)
    
    def set(self, key: str, value: str):
        if len(self.cache) >= self.max_size:
            # Remove oldest item
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
```

2. **Error Handling**:
```python
class ErrorHandler:
    def __init__(self):
        self.error_counts = {}
        self.max_retries = 3
    
    def handle_error(self, error: Exception, operation: str) -> bool:
        if operation not in self.error_counts:
            self.error_counts[operation] = 0
        
        self.error_counts[operation] += 1
        return self.error_counts[operation] < self.max_retries
```

3. **Resource Management**:
```python
class ResourceManager:
    def __init__(self, max_memory_gb: float = 10):
        self.max_memory = max_memory_gb * 1024**2  # Convert to bytes
    
    def check_resources(self) -> bool:
        memory_usage = torch.cuda.memory_allocated()
        return memory_usage < self.max_memory
```

## Further Reading

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [Model Pruning](https://pytorch.org/docs/stable/notes/pruning.html)
- [Load Balancing Strategies](https://www.nginx.com/resources/glossary/load-balancing/)
- [Monitoring Best Practices](https://prometheus.io/docs/practices/naming/) 