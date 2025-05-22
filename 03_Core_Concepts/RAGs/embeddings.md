# Embeddings

This guide covers the fundamental concepts and implementations of embeddings in RAG systems, including text embeddings, image embeddings, multimodal embeddings, and optimization techniques.

## Text Embeddings

### Basic Text Embedding
```python
from sentence_transformers import SentenceTransformer
import numpy as np

class TextEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        return self.model.encode(texts, batch_size=batch_size)
    
    def encode_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.model.encode([text])[0]
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        emb1 = self.encode_single(text1)
        emb2 = self.encode_single(text2)
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
```

### Advanced Text Embedding
```python
class AdvancedTextEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.cache = {}
    
    def preprocess(self, text: str) -> str:
        """Preprocess text before embedding"""
        # Add preprocessing steps
        return text.lower().strip()
    
    def encode_with_metadata(self, text: str) -> Dict[str, Any]:
        """Generate embedding with metadata"""
        processed_text = self.preprocess(text)
        if processed_text in self.cache:
            return self.cache[processed_text]
        
        embedding = self.model.encode([processed_text])[0]
        result = {
            'embedding': embedding,
            'text': text,
            'processed_text': processed_text,
            'timestamp': time.time()
        }
        self.cache[processed_text] = result
        return result
```

## Image Embeddings

### Basic Image Embedding
```python
from torchvision import models, transforms
import torch

class ImageEmbedder:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def encode(self, image: Image.Image) -> np.ndarray:
        """Generate embedding for an image"""
        with torch.no_grad():
            img_tensor = self.transform(image).unsqueeze(0)
            embedding = self.model(img_tensor)
            return embedding.squeeze().numpy()
```

### Advanced Image Embedding
```python
class AdvancedImageEmbedder:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.cache = {}
    
    def preprocess(self, image: Image.Image) -> Image.Image:
        """Preprocess image before embedding"""
        # Add preprocessing steps
        return image.convert('RGB')
    
    def encode_with_metadata(self, image: Image.Image) -> Dict[str, Any]:
        """Generate embedding with metadata"""
        processed_image = self.preprocess(image)
        image_hash = hash(processed_image.tobytes())
        
        if image_hash in self.cache:
            return self.cache[image_hash]
        
        with torch.no_grad():
            img_tensor = self.transform(processed_image).unsqueeze(0)
            embedding = self.model(img_tensor)
            
            result = {
                'embedding': embedding.squeeze().numpy(),
                'image_hash': image_hash,
                'timestamp': time.time()
            }
            self.cache[image_hash] = result
            return result
```

## Multimodal Embeddings

### Basic Multimodal Embedding
```python
class MultimodalEmbedder:
    def __init__(self):
        self.text_embedder = TextEmbedder()
        self.image_embedder = ImageEmbedder()
    
    def encode(self, text: str = None, image: Image.Image = None) -> Dict[str, np.ndarray]:
        """Generate embeddings for text and/or image"""
        result = {}
        
        if text:
            result['text_embedding'] = self.text_embedder.encode_single(text)
        
        if image:
            result['image_embedding'] = self.image_embedder.encode(image)
        
        return result
```

### Advanced Multimodal Embedding
```python
class AdvancedMultimodalEmbedder:
    def __init__(self):
        self.text_embedder = AdvancedTextEmbedder()
        self.image_embedder = AdvancedImageEmbedder()
        self.fusion_model = None  # Add fusion model
    
    def encode_with_metadata(self, text: str = None, 
                           image: Image.Image = None) -> Dict[str, Any]:
        """Generate embeddings with metadata"""
        result = {
            'timestamp': time.time()
        }
        
        if text:
            text_result = self.text_embedder.encode_with_metadata(text)
            result.update({
                'text_embedding': text_result['embedding'],
                'text_metadata': {
                    'processed_text': text_result['processed_text'],
                    'text_timestamp': text_result['timestamp']
                }
            })
        
        if image:
            image_result = self.image_embedder.encode_with_metadata(image)
            result.update({
                'image_embedding': image_result['embedding'],
                'image_metadata': {
                    'image_hash': image_result['image_hash'],
                    'image_timestamp': image_result['timestamp']
                }
            })
        
        if text and image:
            result['multimodal_embedding'] = self.fuse_embeddings(
                text_result['embedding'],
                image_result['embedding']
            )
        
        return result
```

## Embedding Optimization

### Batch Processing
```python
class BatchEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.batch_size = 32
        self.queue = []
    
    def add_to_queue(self, text: str) -> None:
        """Add text to processing queue"""
        self.queue.append(text)
    
    def process_queue(self) -> List[np.ndarray]:
        """Process queued texts in batches"""
        embeddings = []
        for i in range(0, len(self.queue), self.batch_size):
            batch = self.queue[i:i + self.batch_size]
            batch_embeddings = self.model.encode(batch)
            embeddings.extend(batch_embeddings)
        self.queue = []
        return embeddings
```

### Caching
```python
class CachedEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.cache = {}
        self.max_cache_size = 10000
    
    def encode(self, text: str) -> np.ndarray:
        """Generate embedding with caching"""
        if text in self.cache:
            return self.cache[text]
        
        embedding = self.model.encode([text])[0]
        
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[text] = embedding
        return embedding
```

## Best Practices

1. **Text Embeddings**:
   - Model selection
   - Text preprocessing
   - Batch processing
   - Caching strategy

2. **Image Embeddings**:
   - Model selection
   - Image preprocessing
   - Batch processing
   - Memory management

3. **Multimodal Embeddings**:
   - Model alignment
   - Feature fusion
   - Cross-modal learning
   - Performance optimization

4. **Embedding Optimization**:
   - Batch processing
   - Caching
   - Memory management
   - Performance monitoring

## Common Patterns

1. **Embedding Pipeline**:
```python
class EmbeddingPipeline:
    def __init__(self):
        self.steps = []
    
    def add_step(self, step: callable) -> None:
        self.steps.append(step)
    
    def process(self, data: Any) -> Any:
        result = data
        for step in self.steps:
            result = step(result)
        return result
```

2. **Embedding Registry**:
```python
class EmbeddingRegistry:
    def __init__(self):
        self.embedders = {}
    
    def register_embedder(self, name: str, embedder: Any) -> None:
        self.embedders[name] = embedder
    
    def get_embedder(self, name: str) -> Any:
        return self.embedders.get(name)
```

3. **Embedding Monitor**:
```python
class EmbeddingMonitor:
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

- [Text Embeddings](https://arxiv.org/abs/2004.07213)
- [Image Embeddings](https://arxiv.org/abs/2004.07213)
- [Multimodal Learning](https://arxiv.org/abs/2004.07213)
- [Embedding Optimization](https://arxiv.org/abs/2004.07213)
- [Vector Representations](https://arxiv.org/abs/2004.07213) 