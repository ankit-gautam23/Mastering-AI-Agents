# Vector Store

This guide covers the fundamental concepts and implementations of vector stores in RAG systems, including in-memory stores, persistent stores, distributed stores, and optimization techniques.

## In-Memory Vector Store

### Basic In-Memory Store
```python
import numpy as np
from typing import List, Dict, Any, Optional

class InMemoryVectorStore:
    def __init__(self):
        self.vectors = []
        self.metadata = []
    
    def add(self, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Add a vector and its metadata to the store"""
        self.vectors.append(vector)
        self.metadata.append(metadata)
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for k nearest neighbors"""
        if not self.vectors:
            return []
        
        # Calculate cosine similarity
        similarities = np.dot(self.vectors, query_vector) / (
            np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vector)
        )
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return results with metadata
        return [
            {
                'vector': self.vectors[i],
                'metadata': self.metadata[i],
                'similarity': similarities[i]
            }
            for i in top_k_indices
        ]
```

### Advanced In-Memory Store
```python
class AdvancedInMemoryVectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = np.zeros((0, dimension))
        self.metadata = []
        self.index = None
    
    def add_batch(self, vectors: np.ndarray, metadata_list: List[Dict[str, Any]]) -> None:
        """Add multiple vectors and their metadata"""
        if len(vectors) != len(metadata_list):
            raise ValueError("Number of vectors and metadata must match")
        
        self.vectors = np.vstack([self.vectors, vectors])
        self.metadata.extend(metadata_list)
        self._rebuild_index()
    
    def _rebuild_index(self) -> None:
        """Rebuild the search index"""
        if len(self.vectors) > 0:
            self.index = np.argsort(np.linalg.norm(self.vectors, axis=1))
    
    def search_with_threshold(self, query_vector: np.ndarray, 
                            k: int = 5, 
                            threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Search with similarity threshold"""
        if not self.vectors.size:
            return []
        
        similarities = np.dot(self.vectors, query_vector) / (
            np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vector)
        )
        
        # Filter by threshold
        mask = similarities >= threshold
        filtered_indices = np.where(mask)[0]
        filtered_similarities = similarities[mask]
        
        # Get top k from filtered results
        top_k_indices = filtered_indices[np.argsort(filtered_similarities)[-k:][::-1]]
        
        return [
            {
                'vector': self.vectors[i],
                'metadata': self.metadata[i],
                'similarity': similarities[i]
            }
            for i in top_k_indices
        ]
```

## Persistent Vector Store

### Basic Persistent Store
```python
import faiss
import numpy as np
import json
import os

class PersistentVectorStore:
    def __init__(self, dimension: int, index_path: str, metadata_path: str):
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        
        # Load existing index and metadata if available
        self._load()
    
    def _load(self) -> None:
        """Load index and metadata from disk"""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
    
    def _save(self) -> None:
        """Save index and metadata to disk"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
    
    def add(self, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Add a vector and its metadata"""
        self.index.add(vector.reshape(1, -1))
        self.metadata.append(metadata)
        self._save()
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for k nearest neighbors"""
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        
        return [
            {
                'vector': self.index.reconstruct(int(idx)),
                'metadata': self.metadata[int(idx)],
                'distance': float(dist)
            }
            for dist, idx in zip(distances[0], indices[0])
            if idx != -1  # FAISS returns -1 for empty slots
        ]
```

### Advanced Persistent Store
```python
class AdvancedPersistentVectorStore:
    def __init__(self, dimension: int, index_path: str, metadata_path: str):
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = faiss.IndexIVFFlat(
            faiss.IndexFlatL2(dimension),
            dimension,
            min(100, max(1, dimension // 10))  # Number of clusters
        )
        self.metadata = []
        self._load()
    
    def train(self, vectors: np.ndarray) -> None:
        """Train the index with sample vectors"""
        if not self.index.is_trained:
            self.index.train(vectors)
    
    def add_batch(self, vectors: np.ndarray, metadata_list: List[Dict[str, Any]]) -> None:
        """Add multiple vectors and their metadata"""
        if len(vectors) != len(metadata_list):
            raise ValueError("Number of vectors and metadata must match")
        
        self.index.add(vectors)
        self.metadata.extend(metadata_list)
        self._save()
    
    def search_with_filter(self, query_vector: np.ndarray, 
                          k: int = 5,
                          filter_func: Optional[callable] = None) -> List[Dict[str, Any]]:
        """Search with custom filter function"""
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            
            metadata = self.metadata[int(idx)]
            if filter_func is None or filter_func(metadata):
                results.append({
                    'vector': self.index.reconstruct(int(idx)),
                    'metadata': metadata,
                    'distance': float(dist)
                })
        
        return results
```

## Distributed Vector Store

### Basic Distributed Store
```python
from typing import List, Dict, Any, Optional
import numpy as np
import redis
import json

class DistributedVectorStore:
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port)
        self.vector_key = 'vectors'
        self.metadata_key = 'metadata'
    
    def add(self, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Add a vector and its metadata to Redis"""
        # Store vector
        vector_bytes = vector.tobytes()
        self.redis.rpush(self.vector_key, vector_bytes)
        
        # Store metadata
        metadata_json = json.dumps(metadata)
        self.redis.rpush(self.metadata_key, metadata_json)
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for k nearest neighbors"""
        # Get all vectors and metadata
        vector_bytes_list = self.redis.lrange(self.vector_key, 0, -1)
        metadata_json_list = self.redis.lrange(self.metadata_key, 0, -1)
        
        if not vector_bytes_list:
            return []
        
        # Convert bytes to numpy arrays
        vectors = np.array([
            np.frombuffer(vec_bytes, dtype=np.float32)
            for vec_bytes in vector_bytes_list
        ])
        
        # Calculate similarities
        similarities = np.dot(vectors, query_vector) / (
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector)
        )
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return results
        return [
            {
                'vector': vectors[i],
                'metadata': json.loads(metadata_json_list[i]),
                'similarity': similarities[i]
            }
            for i in top_k_indices
        ]
```

### Advanced Distributed Store
```python
class AdvancedDistributedVectorStore:
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port)
        self.vector_key = 'vectors'
        self.metadata_key = 'metadata'
        self.index_key = 'index'
    
    def add_batch(self, vectors: np.ndarray, metadata_list: List[Dict[str, Any]]) -> None:
        """Add multiple vectors and their metadata"""
        if len(vectors) != len(metadata_list):
            raise ValueError("Number of vectors and metadata must match")
        
        # Use pipeline for atomic operations
        pipe = self.redis.pipeline()
        
        for vector, metadata in zip(vectors, metadata_list):
            # Store vector
            vector_bytes = vector.tobytes()
            pipe.rpush(self.vector_key, vector_bytes)
            
            # Store metadata
            metadata_json = json.dumps(metadata)
            pipe.rpush(self.metadata_key, metadata_json)
        
        pipe.execute()
    
    def search_with_filter(self, query_vector: np.ndarray,
                          k: int = 5,
                          filter_func: Optional[callable] = None) -> List[Dict[str, Any]]:
        """Search with custom filter function"""
        # Get all vectors and metadata
        vector_bytes_list = self.redis.lrange(self.vector_key, 0, -1)
        metadata_json_list = self.redis.lrange(self.metadata_key, 0, -1)
        
        if not vector_bytes_list:
            return []
        
        # Convert bytes to numpy arrays
        vectors = np.array([
            np.frombuffer(vec_bytes, dtype=np.float32)
            for vec_bytes in vector_bytes_list
        ])
        
        # Calculate similarities
        similarities = np.dot(vectors, query_vector) / (
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector)
        )
        
        # Apply filter if provided
        if filter_func:
            filtered_indices = [
                i for i, metadata in enumerate(metadata_json_list)
                if filter_func(json.loads(metadata))
            ]
            filtered_similarities = similarities[filtered_indices]
            top_k_indices = filtered_indices[np.argsort(filtered_similarities)[-k:][::-1]]
        else:
            top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return results
        return [
            {
                'vector': vectors[i],
                'metadata': json.loads(metadata_json_list[i]),
                'similarity': similarities[i]
            }
            for i in top_k_indices
        ]
```

## Vector Store Optimization

### Index Optimization
```python
class OptimizedVectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexIVFFlat(
            faiss.IndexFlatL2(dimension),
            dimension,
            min(100, max(1, dimension // 10))
        )
        self.metadata = []
    
    def optimize_index(self, vectors: np.ndarray) -> None:
        """Optimize the index for better search performance"""
        if not self.index.is_trained:
            self.index.train(vectors)
        
        # Set search parameters
        self.index.nprobe = min(20, self.index.nlist)  # Number of clusters to visit
    
    def add_with_optimization(self, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Add vector with index optimization"""
        self.index.add(vector.reshape(1, -1))
        self.metadata.append(metadata)
        
        # Periodically optimize index
        if len(self.metadata) % 1000 == 0:
            self.optimize_index(np.array([m['vector'] for m in self.metadata]))
```

### Memory Optimization
```python
class MemoryOptimizedVectorStore:
    def __init__(self, dimension: int, max_vectors: int = 1000000):
        self.dimension = dimension
        self.max_vectors = max_vectors
        self.index = faiss.IndexIVFFlat(
            faiss.IndexFlatL2(dimension),
            dimension,
            min(100, max(1, dimension // 10))
        )
        self.metadata = []
    
    def add_with_memory_management(self, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Add vector with memory management"""
        if len(self.metadata) >= self.max_vectors:
            # Remove oldest entries
            self.metadata = self.metadata[-self.max_vectors:]
            # Rebuild index
            vectors = np.array([m['vector'] for m in self.metadata])
            self.index = faiss.IndexIVFFlat(
                faiss.IndexFlatL2(self.dimension),
                self.dimension,
                min(100, max(1, self.dimension // 10))
            )
            self.index.train(vectors)
            self.index.add(vectors)
        
        self.index.add(vector.reshape(1, -1))
        self.metadata.append(metadata)
```

## Best Practices

1. **In-Memory Stores**:
   - Memory management
   - Batch operations
   - Index optimization
   - Caching strategy

2. **Persistent Stores**:
   - Data persistence
   - Index management
   - Backup strategy
   - Recovery procedures

3. **Distributed Stores**:
   - Load balancing
   - Data replication
   - Consistency management
   - Fault tolerance

4. **Optimization**:
   - Index optimization
   - Memory management
   - Search optimization
   - Performance monitoring

## Common Patterns

1. **Vector Store Factory**:
```python
class VectorStoreFactory:
    @staticmethod
    def create_store(store_type: str, **kwargs) -> Any:
        if store_type == 'memory':
            return InMemoryVectorStore(**kwargs)
        elif store_type == 'persistent':
            return PersistentVectorStore(**kwargs)
        elif store_type == 'distributed':
            return DistributedVectorStore(**kwargs)
        else:
            raise ValueError(f"Unknown store type: {store_type}")
```

2. **Vector Store Monitor**:
```python
class VectorStoreMonitor:
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

3. **Vector Store Validator**:
```python
class VectorStoreValidator:
    def __init__(self, store: Any):
        self.store = store
    
    def validate_add(self, vector: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Validate vector and metadata before adding"""
        if not isinstance(vector, np.ndarray):
            return False
        if not isinstance(metadata, dict):
            return False
        return True
    
    def validate_search(self, query_vector: np.ndarray, k: int) -> bool:
        """Validate search parameters"""
        if not isinstance(query_vector, np.ndarray):
            return False
        if not isinstance(k, int) or k <= 0:
            return False
        return True
```

## Further Reading

- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Redis Documentation](https://redis.io/documentation)
- [Vector Search](https://arxiv.org/abs/2004.07213)
- [Distributed Systems](https://arxiv.org/abs/2004.07213)
- [Performance Optimization](https://arxiv.org/abs/2004.07213) 