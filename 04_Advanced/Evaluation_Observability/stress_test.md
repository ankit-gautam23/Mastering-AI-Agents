# Stress Testing AI Agents

This guide covers the fundamental concepts and implementations of stress testing for AI agents, including load testing, performance testing, and failure scenario analysis.

## Basic Stress Testing

### Load Generator
```python
import asyncio
import time
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime

@dataclass
class LoadTestResult:
    start_time: float
    end_time: float
    duration: float
    success: bool
    error: Optional[str]
    metadata: Dict[str, Any]

class LoadGenerator:
    def __init__(self, max_concurrent: int = 100):
        self.max_concurrent = max_concurrent
        self.results: List[LoadTestResult] = []
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def generate_load(
        self,
        operation: Callable,
        num_requests: int,
        metadata: Optional[Dict] = None
    ) -> List[LoadTestResult]:
        tasks = []
        for _ in range(num_requests):
            task = asyncio.create_task(
                self._execute_operation(operation, metadata)
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    async def _execute_operation(
        self,
        operation: Callable,
        metadata: Optional[Dict] = None
    ) -> LoadTestResult:
        async with self.semaphore:
            start_time = time.time()
            try:
                await operation()
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
            
            end_time = time.time()
            
            result = LoadTestResult(
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                success=success,
                error=error,
                metadata=metadata or {}
            )
            
            self.results.append(result)
            return result
```

### Performance Monitor
```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {
            "response_time": [],
            "throughput": [],
            "error_rate": [],
            "resource_usage": []
        }
    
    def record_metrics(self, results: List[LoadTestResult]) -> None:
        if not results:
            return
        
        # Calculate response time
        response_times = [r.duration for r in results]
        self.metrics["response_time"].extend(response_times)
        
        # Calculate throughput (requests per second)
        total_time = max(r.end_time for r in results) - min(r.start_time for r in results)
        throughput = len(results) / total_time if total_time > 0 else 0
        self.metrics["throughput"].append(throughput)
        
        # Calculate error rate
        error_count = sum(1 for r in results if not r.success)
        error_rate = error_count / len(results) if results else 0
        self.metrics["error_rate"].append(error_rate)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for metric, values in self.metrics.items():
            if not values:
                continue
            
            summary[metric] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "p95": sorted(values)[int(len(values) * 0.95)],
                "p99": sorted(values)[int(len(values) * 0.99)]
            }
        
        return summary
```

## Advanced Stress Testing

### Failure Scenario Generator
```python
class FailureScenarioGenerator:
    def __init__(self):
        self.scenarios = {
            "network_latency": self._simulate_network_latency,
            "timeout": self._simulate_timeout,
            "rate_limit": self._simulate_rate_limit,
            "resource_exhaustion": self._simulate_resource_exhaustion
        }
    
    async def _simulate_network_latency(self, operation: Callable) -> Any:
        await asyncio.sleep(0.5)  # Simulate network delay
        return await operation()
    
    async def _simulate_timeout(self, operation: Callable) -> Any:
        try:
            return await asyncio.wait_for(operation(), timeout=0.1)
        except asyncio.TimeoutError:
            raise Exception("Operation timed out")
    
    async def _simulate_rate_limit(self, operation: Callable) -> Any:
        # Simulate rate limiting
        if random.random() < 0.3:  # 30% chance of rate limit
            raise Exception("Rate limit exceeded")
        return await operation()
    
    async def _simulate_resource_exhaustion(self, operation: Callable) -> Any:
        # Simulate resource exhaustion
        if random.random() < 0.2:  # 20% chance of resource exhaustion
            raise Exception("Resource exhaustion")
        return await operation()
    
    async def run_scenario(
        self,
        scenario_name: str,
        operation: Callable,
        num_requests: int
    ) -> List[LoadTestResult]:
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = self.scenarios[scenario_name]
        generator = LoadGenerator()
        
        return await generator.generate_load(
            lambda: scenario(operation),
            num_requests
        )
```

### Resource Monitor
```python
import psutil
import threading
from typing import Dict, List

class ResourceMonitor:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.metrics: Dict[str, List[float]] = {
            "cpu_percent": [],
            "memory_percent": [],
            "disk_io": [],
            "network_io": []
        }
        self._stop_event = threading.Event()
        self._monitor_thread = None
    
    def start_monitoring(self) -> None:
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_resources)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def _monitor_resources(self) -> None:
        while not self._stop_event.is_set():
            # CPU usage
            self.metrics["cpu_percent"].append(psutil.cpu_percent())
            
            # Memory usage
            self.metrics["memory_percent"].append(psutil.virtual_memory().percent)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            self.metrics["disk_io"].append(disk_io.read_bytes + disk_io.write_bytes)
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self.metrics["network_io"].append(net_io.bytes_sent + net_io.bytes_recv)
            
            time.sleep(self.interval)
    
    def get_resource_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for metric, values in self.metrics.items():
            if not values:
                continue
            
            summary[metric] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "p95": sorted(values)[int(len(values) * 0.95)],
                "p99": sorted(values)[int(len(values) * 0.99)]
            }
        
        return summary
```

## Best Practices

1. **Test Planning**:
   - Define clear objectives
   - Set up monitoring
   - Prepare test data

2. **Load Generation**:
   - Start with small loads
   - Gradually increase load
   - Monitor system behavior

3. **Failure Testing**:
   - Test various failure scenarios
   - Monitor recovery behavior
   - Document failure patterns

4. **Resource Monitoring**:
   - Track system resources
   - Set up alerts
   - Analyze bottlenecks

## Common Patterns

1. **Stress Test Pipeline**:
```python
class StressTestPipeline:
    def __init__(self):
        self.load_generator = LoadGenerator()
        self.performance_monitor = PerformanceMonitor()
        self.resource_monitor = ResourceMonitor()
        self.failure_generator = FailureScenarioGenerator()
    
    async def run_stress_test(
        self,
        operation: Callable,
        num_requests: int,
        scenario: Optional[str] = None
    ) -> Dict[str, Any]:
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        try:
            # Generate load
            if scenario:
                results = await self.failure_generator.run_scenario(
                    scenario,
                    operation,
                    num_requests
                )
            else:
                results = await self.load_generator.generate_load(
                    operation,
                    num_requests
                )
            
            # Record performance metrics
            self.performance_monitor.record_metrics(results)
            
            return {
                "performance_summary": self.performance_monitor.get_summary(),
                "resource_summary": self.resource_monitor.get_resource_summary()
            }
        finally:
            self.resource_monitor.stop_monitoring()
```

2. **Stress Test Dashboard**:
```python
class StressTestDashboard:
    def __init__(self):
        self.pipeline = StressTestPipeline()
        self.test_history = []
    
    async def run_test(
        self,
        operation: Callable,
        num_requests: int,
        scenario: Optional[str] = None
    ) -> None:
        results = await self.pipeline.run_stress_test(
            operation,
            num_requests,
            scenario
        )
        
        self.test_history.append({
            "timestamp": datetime.now(),
            "scenario": scenario,
            "num_requests": num_requests,
            "results": results
        })
    
    def get_dashboard_data(self) -> Dict:
        return {
            "latest_test": self.test_history[-1] if self.test_history else None,
            "test_history": self.test_history[-10:]  # Last 10 tests
        }
```

## Further Reading

- [Load Testing Techniques](https://arxiv.org/abs/2004.07213)
- [Performance Testing](https://arxiv.org/abs/2004.07213)
- [Failure Testing](https://arxiv.org/abs/2004.07213)
- [Resource Monitoring](https://arxiv.org/abs/2004.07213)
- [Stress Testing Best Practices](https://arxiv.org/abs/2004.07213) 