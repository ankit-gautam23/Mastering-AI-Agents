# Metrics for AI Agents

This guide covers the fundamental concepts and implementations of metrics for evaluating AI agents, including performance metrics, monitoring systems, and analysis techniques.

## Basic Metrics Implementation

### Performance Metrics
```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

@dataclass
class AgentMetrics:
    response_time: float
    accuracy: float
    token_usage: int
    error_rate: float
    timestamp: datetime

class MetricsCollector:
    def __init__(self):
        self.metrics_history: List[AgentMetrics] = []
    
    def record_metrics(self, metrics: AgentMetrics) -> None:
        self.metrics_history.append(metrics)
    
    def get_average_metrics(self) -> Dict[str, float]:
        if not self.metrics_history:
            return {}
        
        return {
            "avg_response_time": np.mean([m.response_time for m in self.metrics_history]),
            "avg_accuracy": np.mean([m.accuracy for m in self.metrics_history]),
            "avg_token_usage": np.mean([m.token_usage for m in self.metrics_history]),
            "avg_error_rate": np.mean([m.error_rate for m in self.metrics_history])
        }
```

### Advanced Metrics
```python
class AdvancedMetricsCollector:
    def __init__(self):
        self.metrics_history: List[AgentMetrics] = []
        self.alert_thresholds = {
            "response_time": 1.0,  # seconds
            "error_rate": 0.1,     # 10%
            "token_usage": 1000    # tokens
        }
    
    def record_metrics(self, metrics: AgentMetrics) -> Dict[str, bool]:
        self.metrics_history.append(metrics)
        return self._check_alerts(metrics)
    
    def _check_alerts(self, metrics: AgentMetrics) -> Dict[str, bool]:
        return {
            "response_time_alert": metrics.response_time > self.alert_thresholds["response_time"],
            "error_rate_alert": metrics.error_rate > self.alert_thresholds["error_rate"],
            "token_usage_alert": metrics.token_usage > self.alert_thresholds["token_usage"]
        }
    
    def get_percentile_metrics(self, percentile: float = 95) -> Dict[str, float]:
        if not self.metrics_history:
            return {}
        
        return {
            f"p{percentile}_response_time": np.percentile(
                [m.response_time for m in self.metrics_history], percentile
            ),
            f"p{percentile}_token_usage": np.percentile(
                [m.token_usage for m in self.metrics_history], percentile
            )
        }
```

## Monitoring Systems

### Real-time Monitoring
```python
from threading import Thread
import time
from queue import Queue

class RealTimeMonitor:
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.metrics_queue = Queue()
        self.is_running = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        self.is_running = True
        self.monitor_thread = Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        while self.is_running:
            metrics = self._collect_metrics()
            self.metrics_queue.put(metrics)
            time.sleep(self.update_interval)
    
    def _collect_metrics(self) -> AgentMetrics:
        # Implement actual metrics collection
        return AgentMetrics(
            response_time=0.0,
            accuracy=0.0,
            token_usage=0,
            error_rate=0.0,
            timestamp=datetime.now()
        )
```

### Metrics Aggregation
```python
class MetricsAggregator:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_buffer = []
    
    def add_metrics(self, metrics: AgentMetrics) -> None:
        self.metrics_buffer.append(metrics)
        if len(self.metrics_buffer) > self.window_size:
            self.metrics_buffer.pop(0)
    
    def get_aggregated_metrics(self) -> Dict[str, Dict[str, float]]:
        if not self.metrics_buffer:
            return {}
        
        return {
            "response_time": {
                "mean": np.mean([m.response_time for m in self.metrics_buffer]),
                "std": np.std([m.response_time for m in self.metrics_buffer]),
                "max": max(m.response_time for m in self.metrics_buffer)
            },
            "accuracy": {
                "mean": np.mean([m.accuracy for m in self.metrics_buffer]),
                "std": np.std([m.accuracy for m in self.metrics_buffer]),
                "min": min(m.accuracy for m in self.metrics_buffer)
            },
            "token_usage": {
                "mean": np.mean([m.token_usage for m in self.metrics_buffer]),
                "std": np.std([m.token_usage for m in self.metrics_buffer]),
                "total": sum(m.token_usage for m in self.metrics_buffer)
            }
        }
```

## Analysis Techniques

### Trend Analysis
```python
class TrendAnalyzer:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics_history = []
    
    def add_metrics(self, metrics: AgentMetrics) -> None:
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.window_size:
            self.metrics_history.pop(0)
    
    def analyze_trends(self) -> Dict[str, Dict[str, float]]:
        if len(self.metrics_history) < 2:
            return {}
        
        return {
            "response_time": self._calculate_trend(
                [m.response_time for m in self.metrics_history]
            ),
            "accuracy": self._calculate_trend(
                [m.accuracy for m in self.metrics_history]
            ),
            "error_rate": self._calculate_trend(
                [m.error_rate for m in self.metrics_history]
            )
        }
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, float]:
        if len(values) < 2:
            return {}
        
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": self._calculate_r_squared(values, slope, intercept)
        }
    
    def _calculate_r_squared(self, values: List[float], slope: float, intercept: float) -> float:
        y_pred = slope * np.arange(len(values)) + intercept
        ss_res = np.sum((np.array(values) - y_pred) ** 2)
        ss_tot = np.sum((np.array(values) - np.mean(values)) ** 2)
        return 1 - (ss_res / ss_tot)
```

### Anomaly Detection
```python
class AnomalyDetector:
    def __init__(self, threshold_std: float = 2.0):
        self.threshold_std = threshold_std
        self.metrics_history = []
    
    def add_metrics(self, metrics: AgentMetrics) -> Dict[str, bool]:
        self.metrics_history.append(metrics)
        return self._detect_anomalies(metrics)
    
    def _detect_anomalies(self, metrics: AgentMetrics) -> Dict[str, bool]:
        if len(self.metrics_history) < 2:
            return {}
        
        return {
            "response_time_anomaly": self._is_anomaly(
                [m.response_time for m in self.metrics_history[:-1]],
                metrics.response_time
            ),
            "error_rate_anomaly": self._is_anomaly(
                [m.error_rate for m in self.metrics_history[:-1]],
                metrics.error_rate
            ),
            "token_usage_anomaly": self._is_anomaly(
                [m.token_usage for m in self.metrics_history[:-1]],
                metrics.token_usage
            )
        }
    
    def _is_anomaly(self, history: List[float], current: float) -> bool:
        mean = np.mean(history)
        std = np.std(history)
        return abs(current - mean) > self.threshold_std * std
```

## Best Practices

1. **Metric Selection**:
   - Choose relevant metrics
   - Define clear thresholds
   - Monitor key indicators

2. **Data Collection**:
   - Implement efficient sampling
   - Ensure data quality
   - Handle missing data

3. **Analysis Strategy**:
   - Use appropriate statistical methods
   - Implement trend analysis
   - Detect anomalies

4. **Monitoring Setup**:
   - Set up real-time monitoring
   - Configure alerts
   - Maintain historical data

## Common Patterns

1. **Metrics Pipeline**:
```python
class MetricsPipeline:
    def __init__(self):
        self.collector = MetricsCollector()
        self.aggregator = MetricsAggregator()
        self.analyzer = TrendAnalyzer()
        self.detector = AnomalyDetector()
    
    def process_metrics(self, metrics: AgentMetrics) -> Dict:
        # Collect metrics
        self.collector.record_metrics(metrics)
        
        # Aggregate metrics
        self.aggregator.add_metrics(metrics)
        
        # Analyze trends
        self.analyzer.add_metrics(metrics)
        
        # Detect anomalies
        anomalies = self.detector.add_metrics(metrics)
        
        return {
            "aggregated": self.aggregator.get_aggregated_metrics(),
            "trends": self.analyzer.analyze_trends(),
            "anomalies": anomalies
        }
```

2. **Metrics Dashboard**:
```python
class MetricsDashboard:
    def __init__(self):
        self.pipeline = MetricsPipeline()
        self.alerts = []
    
    def update(self, metrics: AgentMetrics) -> None:
        results = self.pipeline.process_metrics(metrics)
        
        # Check for alerts
        if any(results["anomalies"].values()):
            self.alerts.append({
                "timestamp": metrics.timestamp,
                "metrics": metrics,
                "anomalies": results["anomalies"]
            })
    
    def get_dashboard_data(self) -> Dict:
        return {
            "current_metrics": self.pipeline.collector.get_average_metrics(),
            "trends": self.pipeline.analyzer.analyze_trends(),
            "alerts": self.alerts[-10:]  # Last 10 alerts
        }
```

## Further Reading

- [Metrics Collection Best Practices](https://arxiv.org/abs/2004.07213)
- [Anomaly Detection in Time Series](https://arxiv.org/abs/2004.07213)
- [Statistical Analysis Methods](https://arxiv.org/abs/2004.07213)
- [Monitoring Systems Design](https://arxiv.org/abs/2004.07213)
- [Performance Metrics](https://arxiv.org/abs/2004.07213) 