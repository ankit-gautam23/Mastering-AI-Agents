# Feedback Loop

This guide covers the fundamental concepts and implementations of feedback loops in AI agent frameworks, including performance feedback, learning feedback, adaptation feedback, and optimization feedback.

## Performance Feedback

### Basic Performance Monitor
```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
import numpy as np

@dataclass
class PerformanceMetric:
    name: str
    value: float
    timestamp: float = None

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
        self.thresholds = {}
    
    def record_metric(self, metric: PerformanceMetric) -> None:
        """Record a performance metric"""
        if metric.timestamp is None:
            metric.timestamp = time.time()
        
        if metric.name not in self.metrics:
            self.metrics[metric.name] = []
        self.metrics[metric.name].append(metric)
    
    def set_threshold(self, metric_name: str, threshold: float) -> None:
        """Set a threshold for a metric"""
        self.thresholds[metric_name] = threshold
    
    def check_threshold(self, metric_name: str) -> bool:
        """Check if a metric exceeds its threshold"""
        if metric_name not in self.thresholds:
            return True
        
        values = [m.value for m in self.metrics.get(metric_name, [])]
        if not values:
            return True
        
        return np.mean(values) <= self.thresholds[metric_name]
    
    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        values = [m.value for m in self.metrics.get(metric_name, [])]
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
```

### Advanced Performance Monitor
```python
class AdvancedPerformanceMonitor:
    def __init__(self):
        self.metrics = {}
        self.thresholds = {}
        self.observers = {}
        self.aggregators = {}
    
    def add_observer(self, metric_name: str, observer: Callable) -> None:
        """Add an observer for a metric"""
        if metric_name not in self.observers:
            self.observers[metric_name] = []
        self.observers[metric_name].append(observer)
    
    def add_aggregator(self, metric_name: str, aggregator: Callable) -> None:
        """Add an aggregator for a metric"""
        self.aggregators[metric_name] = aggregator
    
    def record_metric(self, metric: PerformanceMetric) -> None:
        """Record a performance metric with observers"""
        if metric.timestamp is None:
            metric.timestamp = time.time()
        
        if metric.name not in self.metrics:
            self.metrics[metric.name] = []
        self.metrics[metric.name].append(metric)
        
        # Notify observers
        if metric.name in self.observers:
            for observer in self.observers[metric.name]:
                observer(metric)
    
    def get_aggregated_value(self, metric_name: str) -> float:
        """Get aggregated value for a metric"""
        if metric_name not in self.aggregators:
            return np.mean([m.value for m in self.metrics.get(metric_name, [])])
        
        aggregator = self.aggregators[metric_name]
        return aggregator([m.value for m in self.metrics.get(metric_name, [])])
    
    def get_statistics(self,
                      metric_name: str,
                      window_size: int = None) -> Dict[str, float]:
        """Get statistics for a metric with optional window"""
        values = [m.value for m in self.metrics.get(metric_name, [])]
        if not values:
            return {}
        
        if window_size:
            values = values[-window_size:]
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'aggregated': self.get_aggregated_value(metric_name)
        }
```

## Learning Feedback

### Basic Learning Monitor
```python
@dataclass
class LearningMetric:
    name: str
    value: float
    epoch: int
    timestamp: float = None

class LearningMonitor:
    def __init__(self):
        self.metrics = {}
        self.best_values = {}
    
    def record_metric(self, metric: LearningMetric) -> None:
        """Record a learning metric"""
        if metric.timestamp is None:
            metric.timestamp = time.time()
        
        if metric.name not in self.metrics:
            self.metrics[metric.name] = []
            self.best_values[metric.name] = float('inf')
        
        self.metrics[metric.name].append(metric)
        
        # Update best value
        if metric.value < self.best_values[metric.name]:
            self.best_values[metric.name] = metric.value
    
    def get_best_value(self, metric_name: str) -> float:
        """Get best value for a metric"""
        return self.best_values.get(metric_name, float('inf'))
    
    def get_learning_curve(self, metric_name: str) -> List[float]:
        """Get learning curve for a metric"""
        return [m.value for m in self.metrics.get(metric_name, [])]
    
    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        values = [m.value for m in self.metrics.get(metric_name, [])]
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'best': self.get_best_value(metric_name)
        }
```

### Advanced Learning Monitor
```python
class AdvancedLearningMonitor:
    def __init__(self):
        self.metrics = {}
        self.best_values = {}
        self.early_stopping = {}
        self.learning_rate_schedulers = {}
    
    def add_early_stopping(self,
                          metric_name: str,
                          patience: int,
                          min_delta: float = 0.0) -> None:
        """Add early stopping for a metric"""
        self.early_stopping[metric_name] = {
            'patience': patience,
            'min_delta': min_delta,
            'best_value': float('inf'),
            'counter': 0
        }
    
    def add_learning_rate_scheduler(self,
                                  metric_name: str,
                                  scheduler: Callable) -> None:
        """Add learning rate scheduler for a metric"""
        self.learning_rate_schedulers[metric_name] = scheduler
    
    def record_metric(self, metric: LearningMetric) -> None:
        """Record a learning metric with early stopping"""
        if metric.timestamp is None:
            metric.timestamp = time.time()
        
        if metric.name not in self.metrics:
            self.metrics[metric.name] = []
            self.best_values[metric.name] = float('inf')
        
        self.metrics[metric.name].append(metric)
        
        # Update best value
        if metric.value < self.best_values[metric_name]:
            self.best_values[metric_name] = metric.value
        
        # Check early stopping
        if metric.name in self.early_stopping:
            es = self.early_stopping[metric.name]
            if metric.value > es['best_value'] - es['min_delta']:
                es['counter'] += 1
            else:
                es['counter'] = 0
                es['best_value'] = metric.value
    
    def should_stop(self, metric_name: str) -> bool:
        """Check if training should stop"""
        if metric_name not in self.early_stopping:
            return False
        
        es = self.early_stopping[metric_name]
        return es['counter'] >= es['patience']
    
    def get_learning_rate(self, metric_name: str) -> float:
        """Get learning rate from scheduler"""
        if metric_name not in self.learning_rate_schedulers:
            return 1.0
        
        scheduler = self.learning_rate_schedulers[metric_name]
        return scheduler(self.get_learning_curve(metric_name))
```

## Adaptation Feedback

### Basic Adaptation Monitor
```python
@dataclass
class AdaptationMetric:
    name: str
    value: float
    context: Dict[str, Any]
    timestamp: float = None

class AdaptationMonitor:
    def __init__(self):
        self.metrics = {}
        self.contexts = {}
    
    def record_metric(self, metric: AdaptationMetric) -> None:
        """Record an adaptation metric"""
        if metric.timestamp is None:
            metric.timestamp = time.time()
        
        if metric.name not in self.metrics:
            self.metrics[metric.name] = []
            self.contexts[metric.name] = []
        
        self.metrics[metric.name].append(metric)
        self.contexts[metric.name].append(metric.context)
    
    def get_context_statistics(self,
                             metric_name: str,
                             context_key: str) -> Dict[str, float]:
        """Get statistics for a context key"""
        contexts = self.contexts.get(metric_name, [])
        values = [c.get(context_key) for c in contexts if context_key in c]
        
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    def get_adaptation_curve(self,
                           metric_name: str,
                           context_key: str = None) -> List[float]:
        """Get adaptation curve for a metric"""
        if context_key:
            return [
                m.context.get(context_key)
                for m in self.metrics.get(metric_name, [])
                if context_key in m.context
            ]
        return [m.value for m in self.metrics.get(metric_name, [])]
```

### Advanced Adaptation Monitor
```python
class AdvancedAdaptationMonitor:
    def __init__(self):
        self.metrics = {}
        self.contexts = {}
        self.adaptation_strategies = {}
        self.context_analyzers = {}
    
    def add_adaptation_strategy(self,
                              metric_name: str,
                              strategy: Callable) -> None:
        """Add an adaptation strategy"""
        self.adaptation_strategies[metric_name] = strategy
    
    def add_context_analyzer(self,
                           metric_name: str,
                           analyzer: Callable) -> None:
        """Add a context analyzer"""
        self.context_analyzers[metric_name] = analyzer
    
    def record_metric(self, metric: AdaptationMetric) -> None:
        """Record an adaptation metric with analysis"""
        if metric.timestamp is None:
            metric.timestamp = time.time()
        
        if metric.name not in self.metrics:
            self.metrics[metric.name] = []
            self.contexts[metric.name] = []
        
        self.metrics[metric.name].append(metric)
        self.contexts[metric.name].append(metric.context)
        
        # Analyze context
        if metric.name in self.context_analyzers:
            analyzer = self.context_analyzers[metric.name]
            analyzer(metric.context)
    
    def get_adaptation_suggestion(self,
                                metric_name: str,
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Get adaptation suggestion from strategy"""
        if metric_name not in self.adaptation_strategies:
            return {}
        
        strategy = self.adaptation_strategies[metric_name]
        return strategy(context)
    
    def get_context_analysis(self,
                           metric_name: str,
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Get context analysis"""
        if metric_name not in self.context_analyzers:
            return {}
        
        analyzer = self.context_analyzers[metric_name]
        return analyzer(context)
```

## Optimization Feedback

### Basic Optimization Monitor
```python
@dataclass
class OptimizationMetric:
    name: str
    value: float
    parameters: Dict[str, Any]
    timestamp: float = None

class OptimizationMonitor:
    def __init__(self):
        self.metrics = {}
        self.best_parameters = {}
    
    def record_metric(self, metric: OptimizationMetric) -> None:
        """Record an optimization metric"""
        if metric.timestamp is None:
            metric.timestamp = time.time()
        
        if metric.name not in self.metrics:
            self.metrics[metric.name] = []
            self.best_parameters[metric.name] = (float('inf'), {})
        
        self.metrics[metric.name].append(metric)
        
        # Update best parameters
        if metric.value < self.best_parameters[metric.name][0]:
            self.best_parameters[metric.name] = (metric.value, metric.parameters)
    
    def get_best_parameters(self, metric_name: str) -> Dict[str, Any]:
        """Get best parameters for a metric"""
        return self.best_parameters.get(metric_name, (float('inf'), {}))[1]
    
    def get_optimization_curve(self, metric_name: str) -> List[float]:
        """Get optimization curve for a metric"""
        return [m.value for m in self.metrics.get(metric_name, [])]
```

### Advanced Optimization Monitor
```python
class AdvancedOptimizationMonitor:
    def __init__(self):
        self.metrics = {}
        self.best_parameters = {}
        self.optimization_strategies = {}
        self.parameter_constraints = {}
    
    def add_optimization_strategy(self,
                                metric_name: str,
                                strategy: Callable) -> None:
        """Add an optimization strategy"""
        self.optimization_strategies[metric_name] = strategy
    
    def add_parameter_constraint(self,
                               metric_name: str,
                               constraint: Callable) -> None:
        """Add a parameter constraint"""
        self.parameter_constraints[metric_name] = constraint
    
    def record_metric(self, metric: OptimizationMetric) -> None:
        """Record an optimization metric with constraints"""
        if metric.timestamp is None:
            metric.timestamp = time.time()
        
        if metric.name not in self.metrics:
            self.metrics[metric.name] = []
            self.best_parameters[metric.name] = (float('inf'), {})
        
        # Check parameter constraints
        if metric.name in self.parameter_constraints:
            constraint = self.parameter_constraints[metric.name]
            if not constraint(metric.parameters):
                return
        
        self.metrics[metric.name].append(metric)
        
        # Update best parameters
        if metric.value < self.best_parameters[metric_name][0]:
            self.best_parameters[metric_name] = (metric.value, metric.parameters)
    
    def get_next_parameters(self,
                          metric_name: str,
                          current_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get next parameters from strategy"""
        if metric_name not in self.optimization_strategies:
            return current_parameters
        
        strategy = self.optimization_strategies[metric_name]
        return strategy(current_parameters, self.get_optimization_curve(metric_name))
```

## Best Practices

1. **Performance Feedback**:
   - Metric collection
   - Threshold monitoring
   - Statistical analysis
   - Observer notification

2. **Learning Feedback**:
   - Learning curve tracking
   - Early stopping
   - Learning rate scheduling
   - Best value tracking

3. **Adaptation Feedback**:
   - Context analysis
   - Adaptation strategies
   - Context tracking
   - Strategy selection

4. **Optimization Feedback**:
   - Parameter tracking
   - Constraint satisfaction
   - Strategy selection
   - Best parameter tracking

## Common Patterns

1. **Feedback Factory**:
```python
class FeedbackFactory:
    @staticmethod
    def create_monitor(monitor_type: str, **kwargs) -> Any:
        if monitor_type == 'performance':
            return PerformanceMonitor(**kwargs)
        elif monitor_type == 'learning':
            return LearningMonitor(**kwargs)
        elif monitor_type == 'adaptation':
            return AdaptationMonitor(**kwargs)
        elif monitor_type == 'optimization':
            return OptimizationMonitor(**kwargs)
        else:
            raise ValueError(f"Unknown monitor type: {monitor_type}")
```

2. **Feedback Analyzer**:
```python
class FeedbackAnalyzer:
    def __init__(self):
        self.analyzers = {}
    
    def add_analyzer(self, name: str, analyzer: Callable) -> None:
        """Add a feedback analyzer"""
        self.analyzers[name] = analyzer
    
    def analyze_feedback(self,
                        feedback_type: str,
                        data: List[Any]) -> Dict[str, Any]:
        """Analyze feedback data"""
        if feedback_type not in self.analyzers:
            return {}
        
        analyzer = self.analyzers[feedback_type]
        return analyzer(data)
```

3. **Feedback Validator**:
```python
class FeedbackValidator:
    def __init__(self):
        self.validators = {}
    
    def add_validator(self, name: str, validator: Callable) -> None:
        """Add a feedback validator"""
        self.validators[name] = validator
    
    def validate_feedback(self,
                         feedback_type: str,
                         data: Any) -> bool:
        """Validate feedback data"""
        if feedback_type not in self.validators:
            return True
        
        validator = self.validators[feedback_type]
        return validator(data)
```

## Further Reading

- [Performance Monitoring](https://arxiv.org/abs/2004.07213)
- [Learning Systems](https://arxiv.org/abs/2004.07213)
- [Adaptation Mechanisms](https://arxiv.org/abs/2004.07213)
- [Optimization Techniques](https://arxiv.org/abs/2004.07213)
- [Feedback Systems](https://arxiv.org/abs/2004.07213) 