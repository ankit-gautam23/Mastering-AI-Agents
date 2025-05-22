# Retrieval Models

This guide covers the fundamental concepts and implementations of retrieval models in RAG systems, including dense retrievers, sparse retrievers, hybrid retrievers, and optimization techniques.

## Dense Retrievers

### Basic Dense Retriever
```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any

class DenseRetriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, documents: List[str]) -> None:
        """Add documents to the retriever"""
        self.documents.extend(documents)
        new_embeddings = self.model.encode(documents)
        self.embeddings.extend(new_embeddings)
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve k most relevant documents"""
        query_embedding = self.model.encode([query])[0]
        
        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return [
            {
                'document': self.documents[i],
                'score': float(similarities[i])
            }
            for i in top_k_indices
        ]
```

### Advanced Dense Retriever
```python
class AdvancedDenseRetriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = []
        self.metadata = []
    
    def add_documents_with_metadata(self, 
                                  documents: List[str],
                                  metadata_list: List[Dict[str, Any]]) -> None:
        """Add documents with metadata"""
        if len(documents) != len(metadata_list):
            raise ValueError("Number of documents and metadata must match")
        
        self.documents.extend(documents)
        self.metadata.extend(metadata_list)
        
        # Batch encode documents
        new_embeddings = self.model.encode(documents, batch_size=32)
        self.embeddings.extend(new_embeddings)
    
    def retrieve_with_filter(self, 
                           query: str,
                           k: int = 5,
                           filter_func: callable = None) -> List[Dict[str, Any]]:
        """Retrieve with custom filter function"""
        query_embedding = self.model.encode([query])[0]
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Apply filter if provided
        if filter_func:
            filtered_indices = [
                i for i, metadata in enumerate(self.metadata)
                if filter_func(metadata)
            ]
            filtered_similarities = similarities[filtered_indices]
            top_k_indices = filtered_indices[np.argsort(filtered_similarities)[-k:][::-1]]
        else:
            top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return [
            {
                'document': self.documents[i],
                'metadata': self.metadata[i],
                'score': float(similarities[i])
            }
            for i in top_k_indices
        ]
```

## Sparse Retrievers

### Basic Sparse Retriever
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any

class SparseRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.documents = []
        self.tfidf_matrix = None
    
    def add_documents(self, documents: List[str]) -> None:
        """Add documents to the retriever"""
        self.documents.extend(documents)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve k most relevant documents"""
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = (query_vector * self.tfidf_matrix.T).toarray()[0]
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return [
            {
                'document': self.documents[i],
                'score': float(similarities[i])
            }
            for i in top_k_indices
        ]
```

### Advanced Sparse Retriever
```python
class AdvancedSparseRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.documents = []
        self.tfidf_matrix = None
        self.metadata = []
    
    def add_documents_with_metadata(self,
                                  documents: List[str],
                                  metadata_list: List[Dict[str, Any]]) -> None:
        """Add documents with metadata"""
        if len(documents) != len(metadata_list):
            raise ValueError("Number of documents and metadata must match")
        
        self.documents.extend(documents)
        self.metadata.extend(metadata_list)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
    
    def retrieve_with_filter(self,
                           query: str,
                           k: int = 5,
                           filter_func: callable = None) -> List[Dict[str, Any]]:
        """Retrieve with custom filter function"""
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = (query_vector * self.tfidf_matrix.T).toarray()[0]
        
        # Apply filter if provided
        if filter_func:
            filtered_indices = [
                i for i, metadata in enumerate(self.metadata)
                if filter_func(metadata)
            ]
            filtered_similarities = similarities[filtered_indices]
            top_k_indices = filtered_indices[np.argsort(filtered_similarities)[-k:][::-1]]
        else:
            top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return [
            {
                'document': self.documents[i],
                'metadata': self.metadata[i],
                'score': float(similarities[i])
            }
            for i in top_k_indices
        ]
```

## Hybrid Retrievers

### Basic Hybrid Retriever
```python
class HybridRetriever:
    def __init__(self, dense_model_name: str = 'all-MiniLM-L6-v2'):
        self.dense_retriever = DenseRetriever(dense_model_name)
        self.sparse_retriever = SparseRetriever()
    
    def add_documents(self, documents: List[str]) -> None:
        """Add documents to both retrievers"""
        self.dense_retriever.add_documents(documents)
        self.sparse_retriever.add_documents(documents)
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve using both models and combine results"""
        dense_results = self.dense_retriever.retrieve(query, k)
        sparse_results = self.sparse_retriever.retrieve(query, k)
        
        # Combine results using reciprocal rank fusion
        combined_results = self._combine_results(dense_results, sparse_results)
        return combined_results[:k]
    
    def _combine_results(self,
                        dense_results: List[Dict[str, Any]],
                        sparse_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine results using reciprocal rank fusion"""
        scores = {}
        
        for i, result in enumerate(dense_results):
            doc = result['document']
            if doc not in scores:
                scores[doc] = 0
            scores[doc] += 1 / (i + 1)
        
        for i, result in enumerate(sparse_results):
            doc = result['document']
            if doc not in scores:
                scores[doc] = 0
            scores[doc] += 1 / (i + 1)
        
        # Sort by combined score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                'document': doc,
                'score': score
            }
            for doc, score in sorted_docs
        ]
```

### Advanced Hybrid Retriever
```python
class AdvancedHybridRetriever:
    def __init__(self, dense_model_name: str = 'all-MiniLM-L6-v2'):
        self.dense_retriever = AdvancedDenseRetriever(dense_model_name)
        self.sparse_retriever = AdvancedSparseRetriever()
        self.weights = {'dense': 0.7, 'sparse': 0.3}
    
    def add_documents_with_metadata(self,
                                  documents: List[str],
                                  metadata_list: List[Dict[str, Any]]) -> None:
        """Add documents with metadata to both retrievers"""
        self.dense_retriever.add_documents_with_metadata(documents, metadata_list)
        self.sparse_retriever.add_documents_with_metadata(documents, metadata_list)
    
    def retrieve_with_filter(self,
                           query: str,
                           k: int = 5,
                           filter_func: callable = None) -> List[Dict[str, Any]]:
        """Retrieve using both models with filter"""
        dense_results = self.dense_retriever.retrieve_with_filter(
            query, k, filter_func
        )
        sparse_results = self.sparse_retriever.retrieve_with_filter(
            query, k, filter_func
        )
        
        # Combine results using weighted reciprocal rank fusion
        combined_results = self._combine_results_weighted(
            dense_results, sparse_results
        )
        return combined_results[:k]
    
    def _combine_results_weighted(self,
                                dense_results: List[Dict[str, Any]],
                                sparse_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine results using weighted reciprocal rank fusion"""
        scores = {}
        
        for i, result in enumerate(dense_results):
            doc = result['document']
            if doc not in scores:
                scores[doc] = 0
            scores[doc] += self.weights['dense'] / (i + 1)
        
        for i, result in enumerate(sparse_results):
            doc = result['document']
            if doc not in scores:
                scores[doc] = 0
            scores[doc] += self.weights['sparse'] / (i + 1)
        
        # Sort by combined score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                'document': doc,
                'score': score
            }
            for doc, score in sorted_docs
        ]
```

## Retrieval Optimization

### Batch Processing
```python
class BatchRetriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = []
        self.batch_size = 32
    
    def add_documents_batch(self, documents: List[str]) -> None:
        """Add documents in batches"""
        self.documents.extend(documents)
        
        # Process in batches
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            batch_embeddings = self.model.encode(batch)
            self.embeddings.extend(batch_embeddings)
    
    def retrieve_batch(self, queries: List[str], k: int = 5) -> List[List[Dict[str, Any]]]:
        """Retrieve for multiple queries in batches"""
        all_results = []
        
        # Process queries in batches
        for i in range(0, len(queries), self.batch_size):
            batch_queries = queries[i:i + self.batch_size]
            batch_embeddings = self.model.encode(batch_queries)
            
            # Calculate similarities for batch
            similarities = np.dot(self.embeddings, batch_embeddings.T)
            similarities = similarities / (
                np.linalg.norm(self.embeddings, axis=1)[:, np.newaxis] *
                np.linalg.norm(batch_embeddings, axis=1)
            )
            
            # Get top k for each query
            for j in range(len(batch_queries)):
                top_k_indices = np.argsort(similarities[:, j])[-k:][::-1]
                results = [
                    {
                        'document': self.documents[idx],
                        'score': float(similarities[idx, j])
                    }
                    for idx in top_k_indices
                ]
                all_results.append(results)
        
        return all_results
```

### Caching
```python
class CachedRetriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = []
        self.cache = {}
        self.max_cache_size = 10000
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve with caching"""
        cache_key = f"{query}_{k}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        query_embedding = self.model.encode([query])[0]
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = [
            {
                'document': self.documents[i],
                'score': float(similarities[i])
            }
            for i in top_k_indices
        ]
        
        # Update cache
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = results
        return results
```

## Best Practices

1. **Dense Retrievers**:
   - Model selection
   - Batch processing
   - Memory management
   - Caching strategy

2. **Sparse Retrievers**:
   - Feature selection
   - Text preprocessing
   - Index optimization
   - Memory management

3. **Hybrid Retrievers**:
   - Model combination
   - Weight tuning
   - Result fusion
   - Performance optimization

4. **Optimization**:
   - Batch processing
   - Caching
   - Memory management
   - Performance monitoring

## Common Patterns

1. **Retriever Factory**:
```python
class RetrieverFactory:
    @staticmethod
    def create_retriever(retriever_type: str, **kwargs) -> Any:
        if retriever_type == 'dense':
            return DenseRetriever(**kwargs)
        elif retriever_type == 'sparse':
            return SparseRetriever(**kwargs)
        elif retriever_type == 'hybrid':
            return HybridRetriever(**kwargs)
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
```

2. **Retriever Monitor**:
```python
class RetrieverMonitor:
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

3. **Retriever Validator**:
```python
class RetrieverValidator:
    def __init__(self, retriever: Any):
        self.retriever = retriever
    
    def validate_query(self, query: str) -> bool:
        """Validate query before retrieval"""
        if not isinstance(query, str):
            return False
        if not query.strip():
            return False
        return True
    
    def validate_results(self, results: List[Dict[str, Any]]) -> bool:
        """Validate retrieval results"""
        if not isinstance(results, list):
            return False
        for result in results:
            if not isinstance(result, dict):
                return False
            if 'document' not in result or 'score' not in result:
                return False
        return True
```

## Further Reading

- [Dense Retrieval](https://arxiv.org/abs/2004.07213)
- [Sparse Retrieval](https://arxiv.org/abs/2004.07213)
- [Hybrid Retrieval](https://arxiv.org/abs/2004.07213)
- [Retrieval Optimization](https://arxiv.org/abs/2004.07213)
- [Information Retrieval](https://arxiv.org/abs/2004.07213) 