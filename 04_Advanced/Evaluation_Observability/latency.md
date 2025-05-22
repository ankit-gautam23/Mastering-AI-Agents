# Latency Monitoring and Optimization

This guide covers the fundamental concepts and implementations of latency monitoring and optimization for AI agents, including measurement techniques, analysis, and best practices.

## Basic Latency Measurement

### Simple Latency Monitor
```python
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

@dataclass
class LatencyMetrics:
    start_time: float
    end_time: float
    duration: float
    operation: str
    metadata: Dict[str, Any]

class LatencyMonitor:
    def __init__(self):
        self.metrics_history: List[LatencyMetrics] = []
    
    def measure_latency(self, operation: str, metadata: Optional[Dict] = None) -> float:
        start_time = time.time()
        return start_time
    
    def record_latency(self, start_time: float, operation: str, metadata: Optional[Dict] = None) -> None:
        end_time = time.time()
        duration = end_time - start_time
        
        metrics = LatencyMetrics(
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            operation=operation,
            metadata=metadata or {}
        )
        
        self.metrics_history.append(metrics)
    
    def get_average_latency(self, operation: Optional[str] = None) -> float:
        if not self.metrics_history:
            return 0.0
        
        filtered_metrics = (
            self.metrics_history
            if operation is None
            else [m for m in self.metrics_history if m.operation == operation]
        )
        
        return sum(m.duration for m in filtered_metrics) / len(filtered_metrics)
```

### Context Manager Implementation
```python
class LatencyContext:
    def __init__(self, monitor: LatencyMonitor, operation: str, metadata: Optional[Dict] = None):
        self.monitor = monitor
        self.operation = operation
        self.metadata = metadata
        self.start_time = None
    
    def __enter__(self):
        self.start_time = self.monitor.measure_latency(self.operation, self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.record_latency(self.start_time, self.operation, self.metadata)
```

## Advanced Latency Analysis

### Percentile Analysis
```python
class LatencyAnalyzer:
    def __init__(self):
        self.monitor = LatencyMonitor()
    
    def analyze_latency(self, operation: Optional[str] = None) -> Dict[str, float]:
        if not self.monitor.metrics_history:
            return {}
        
        filtered_metrics = (
            self.monitor.metrics_history
            if operation is None
            else [m for m in self.monitor.metrics_history if m.operation == operation]
        )
        
        durations = [m.duration for m in filtered_metrics]
        
        return {
            "p50": np.percentile(durations, 50),
            "p90": np.percentile(durations, 90),
            "p95": np.percentile(durations, 95),
            "p99": np.percentile(durations, 99),
            "mean": np.mean(durations),
            "std": np.std(durations)
        }
```

### Latency Distribution
```python
class LatencyDistribution:
    def __init__(self, bin_size: float = 0.1):
        self.bin_size = bin_size
        self.distribution = {}
    
    def add_latency(self, duration: float) -> None:
        bin_index = int(duration / self.bin_size)
        self.distribution[bin_index] = self.distribution.get(bin_index, 0) + 1
    
    def get_distribution(self) -> Dict[float, int]:
        return {
            bin_index * self.bin_size: count
            for bin_index, count in sorted(self.distribution.items())
        }
    
    def get_percentile(self, percentile: float) -> float:
        if not self.distribution:
            return 0.0
        
        total = sum(self.distribution.values())
        target = total * percentile / 100
        
        current = 0
        for bin_index, count in sorted(self.distribution.items()):
            current += count
            if current >= target:
                return bin_index * self.bin_size
        
        return 0.0
```

## Latency Optimization

### Caching Implementation
```python
from functools import lru_cache
import time

class LatencyOptimizer:
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    @lru_cache(maxsize=1000)
    def cached_operation(self, operation_id: str, *args, **kwargs):
        return self._expensive_operation(operation_id, *args, **kwargs)
    
    def _expensive_operation(self, operation_id: str, *args, **kwargs):
        # Simulate expensive operation
        time.sleep(0.1)
        return f"Result for {operation_id}"
    
    def get_cache_stats(self) -> Dict[str, int]:
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0
            else 0
        }
```

### Batch Processing
```python
class BatchProcessor:
    def __init__(self, batch_size: int = 10, max_wait_time: float = 0.1):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.batch = []
        self.last_process_time = time.time()
    
    def add_to_batch(self, item: Any) -> None:
        self.batch.append(item)
        
        if (
            len(self.batch) >= self.batch_size or
            time.time() - self.last_process_time >= self.max_wait_time
        ):
            self.process_batch()
    
    def process_batch(self) -> None:
        if not self.batch:
            return
        
        # Process batch
        results = self._process_items(self.batch)
        
        # Clear batch
        self.batch = []
        self.last_process_time = time.time()
        
        return results
    
    def _process_items(self, items: List[Any]) -> List[Any]:
        # Implement batch processing logic
        return [f"Processed {item}" for item in items]
```

## Best Practices

1. **Measurement Strategy**:
   - Use appropriate granularity
   - Consider system load
   - Account for overhead

2. **Optimization Techniques**:
   - Implement caching
   - Use batch processing
   - Optimize critical paths

3. **Monitoring Setup**:
   - Set up alerts
   - Track trends
   - Monitor outliers

4. **Performance Tuning**:
   - Profile code
   - Identify bottlenecks
   - Optimize resources

## Common Patterns

1. **Latency Pipeline**:
```python
class LatencyPipeline:
    def __init__(self):
        self.monitor = LatencyMonitor()
        self.analyzer = LatencyAnalyzer()
        self.optimizer = LatencyOptimizer()
    
    def process_operation(self, operation: str, metadata: Optional[Dict] = None) -> Any:
        with LatencyContext(self.monitor, operation, metadata):
            # Perform operation
            result = self.optimizer.cached_operation(operation)
            
            # Analyze latency
            analysis = self.analyzer.analyze_latency(operation)
            
            return {
                "result": result,
                "latency_analysis": analysis
            }
```

2. **Latency Dashboard**:
```python
class LatencyDashboard:
    def __init__(self):
        self.pipeline = LatencyPipeline()
        self.alerts = []
    
    def process_operation(self, operation: str, metadata: Optional[Dict] = None) -> None:
        results = self.pipeline.process_operation(operation, metadata)
        
        # Check for latency alerts
        if results["latency_analysis"]["p95"] > 1.0:  # 1 second threshold
            self.alerts.append({
                "timestamp": datetime.now(),
                "operation": operation,
                "latency": results["latency_analysis"]["p95"]
            })
    
    def get_dashboard_data(self) -> Dict:
        return {
            "current_metrics": self.pipeline.monitor.get_average_latency(),
            "analysis": self.pipeline.analyzer.analyze_latency(),
            "alerts": self.alerts[-10:]  # Last 10 alerts
        }
```

## Further Reading

- [Latency Measurement Techniques](https://arxiv.org/abs/2004.07213)
- [Performance Optimization](https://arxiv.org/abs/2004.07213)
- [Caching Strategies](https://arxiv.org/abs/2004.07213)
- [Batch Processing](https://arxiv.org/abs/2004.07213)
- [Latency Monitoring](https://arxiv.org/abs/2004.07213) 