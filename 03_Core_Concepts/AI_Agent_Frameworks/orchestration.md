# Orchestration

This guide covers the fundamental concepts and implementations of orchestration in AI agent frameworks, including task orchestration, workflow management, state management, and error handling.

## Task Orchestration

### Basic Task Orchestrator
```python
from typing import List, Dict, Any, Callable
import asyncio
from dataclasses import dataclass

@dataclass
class Task:
    name: str
    func: Callable
    dependencies: List[str] = None
    timeout: float = None
    retries: int = 0

class TaskOrchestrator:
    def __init__(self):
        self.tasks = {}
        self.results = {}
    
    def add_task(self, task: Task) -> None:
        """Add a task to the orchestrator"""
        self.tasks[task.name] = task
    
    async def execute_task(self, task: Task) -> Any:
        """Execute a single task"""
        try:
            if task.dependencies:
                # Wait for dependencies
                for dep in task.dependencies:
                    if dep not in self.results:
                        raise ValueError(f"Dependency {dep} not found")
            
            # Execute task
            result = await asyncio.wait_for(
                task.func(),
                timeout=task.timeout
            )
            self.results[task.name] = result
            return result
            
        except asyncio.TimeoutError:
            if task.retries > 0:
                task.retries -= 1
                return await self.execute_task(task)
            raise
    
    async def execute_all(self) -> Dict[str, Any]:
        """Execute all tasks in dependency order"""
        # Sort tasks by dependencies
        sorted_tasks = self._topological_sort()
        
        # Execute tasks
        for task_name in sorted_tasks:
            task = self.tasks[task_name]
            await self.execute_task(task)
        
        return self.results
    
    def _topological_sort(self) -> List[str]:
        """Sort tasks by dependencies"""
        visited = set()
        temp = set()
        order = []
        
        def visit(task_name):
            if task_name in temp:
                raise ValueError("Circular dependency detected")
            if task_name in visited:
                return
            
            temp.add(task_name)
            task = self.tasks[task_name]
            
            if task.dependencies:
                for dep in task.dependencies:
                    visit(dep)
            
            temp.remove(task_name)
            visited.add(task_name)
            order.append(task_name)
        
        for task_name in self.tasks:
            if task_name not in visited:
                visit(task_name)
        
        return order
```

### Advanced Task Orchestrator
```python
class AdvancedTaskOrchestrator:
    def __init__(self):
        self.tasks = {}
        self.results = {}
        self.monitors = {}
        self.error_handlers = {}
    
    def add_task(self, task: Task) -> None:
        """Add a task with monitoring and error handling"""
        self.tasks[task.name] = task
        self.monitors[task.name] = []
        self.error_handlers[task.name] = []
    
    def add_monitor(self, task_name: str, monitor: Callable) -> None:
        """Add a monitor for a task"""
        if task_name not in self.monitors:
            raise ValueError(f"Task {task_name} not found")
        self.monitors[task_name].append(monitor)
    
    def add_error_handler(self, task_name: str, handler: Callable) -> None:
        """Add an error handler for a task"""
        if task_name not in self.error_handlers:
            raise ValueError(f"Task {task_name} not found")
        self.error_handlers[task_name].append(handler)
    
    async def execute_task(self, task: Task) -> Any:
        """Execute a task with monitoring and error handling"""
        try:
            # Execute task
            result = await super().execute_task(task)
            
            # Run monitors
            for monitor in self.monitors[task.name]:
                await monitor(result)
            
            return result
            
        except Exception as e:
            # Run error handlers
            for handler in self.error_handlers[task.name]:
                await handler(e)
            raise
```

## Workflow Management

### Basic Workflow Manager
```python
from enum import Enum
from typing import List, Dict, Any, Optional

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class Workflow:
    def __init__(self, name: str):
        self.name = name
        self.tasks = []
        self.status = WorkflowStatus.PENDING
        self.results = {}
    
    def add_task(self, task: Task) -> None:
        """Add a task to the workflow"""
        self.tasks.append(task)
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the workflow"""
        self.status = WorkflowStatus.RUNNING
        
        try:
            orchestrator = TaskOrchestrator()
            for task in self.tasks:
                orchestrator.add_task(task)
            
            results = await orchestrator.execute_all()
            self.results = results
            self.status = WorkflowStatus.COMPLETED
            return results
            
        except Exception as e:
            self.status = WorkflowStatus.FAILED
            raise
```

### Advanced Workflow Manager
```python
class AdvancedWorkflow:
    def __init__(self, name: str):
        self.name = name
        self.tasks = []
        self.status = WorkflowStatus.PENDING
        self.results = {}
        self.metadata = {}
        self.hooks = {
            'pre_execute': [],
            'post_execute': [],
            'on_error': []
        }
    
    def add_task(self, task: Task) -> None:
        """Add a task to the workflow"""
        self.tasks.append(task)
    
    def add_hook(self, hook_type: str, hook: Callable) -> None:
        """Add a hook to the workflow"""
        if hook_type not in self.hooks:
            raise ValueError(f"Unknown hook type: {hook_type}")
        self.hooks[hook_type].append(hook)
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the workflow with hooks"""
        self.status = WorkflowStatus.RUNNING
        
        try:
            # Run pre-execute hooks
            for hook in self.hooks['pre_execute']:
                await hook(self)
            
            # Execute workflow
            orchestrator = AdvancedTaskOrchestrator()
            for task in self.tasks:
                orchestrator.add_task(task)
            
            results = await orchestrator.execute_all()
            self.results = results
            
            # Run post-execute hooks
            for hook in self.hooks['post_execute']:
                await hook(self)
            
            self.status = WorkflowStatus.COMPLETED
            return results
            
        except Exception as e:
            self.status = WorkflowStatus.FAILED
            
            # Run error hooks
            for hook in self.hooks['on_error']:
                await hook(self, e)
            
            raise
```

## State Management

### Basic State Manager
```python
class StateManager:
    def __init__(self):
        self.states = {}
        self.history = []
    
    def set_state(self, key: str, value: Any) -> None:
        """Set a state value"""
        self.states[key] = value
        self.history.append({
            'key': key,
            'value': value,
            'timestamp': time.time()
        })
    
    def get_state(self, key: str) -> Any:
        """Get a state value"""
        return self.states.get(key)
    
    def get_history(self, key: str) -> List[Dict[str, Any]]:
        """Get state history for a key"""
        return [
            entry for entry in self.history
            if entry['key'] == key
        ]
```

### Advanced State Manager
```python
class AdvancedStateManager:
    def __init__(self):
        self.states = {}
        self.history = []
        self.validators = {}
        self.subscribers = {}
    
    def add_validator(self, key: str, validator: Callable) -> None:
        """Add a validator for a state key"""
        self.validators[key] = validator
    
    def add_subscriber(self, key: str, subscriber: Callable) -> None:
        """Add a subscriber for state changes"""
        if key not in self.subscribers:
            self.subscribers[key] = []
        self.subscribers[key].append(subscriber)
    
    def set_state(self, key: str, value: Any) -> None:
        """Set a state value with validation and notification"""
        # Validate value
        if key in self.validators:
            if not self.validators[key](value):
                raise ValueError(f"Invalid value for key {key}")
        
        # Update state
        self.states[key] = value
        entry = {
            'key': key,
            'value': value,
            'timestamp': time.time()
        }
        self.history.append(entry)
        
        # Notify subscribers
        if key in self.subscribers:
            for subscriber in self.subscribers[key]:
                subscriber(entry)
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a state value with default"""
        return self.states.get(key, default)
    
    def get_history(self,
                   key: str,
                   start_time: float = None,
                   end_time: float = None) -> List[Dict[str, Any]]:
        """Get filtered state history"""
        history = [
            entry for entry in self.history
            if entry['key'] == key
        ]
        
        if start_time:
            history = [
                entry for entry in history
                if entry['timestamp'] >= start_time
            ]
        
        if end_time:
            history = [
                entry for entry in history
                if entry['timestamp'] <= end_time
            ]
        
        return history
```

## Error Handling

### Basic Error Handler
```python
class ErrorHandler:
    def __init__(self):
        self.handlers = {}
        self.error_log = []
    
    def add_handler(self, error_type: type, handler: Callable) -> None:
        """Add an error handler"""
        self.handlers[error_type] = handler
    
    def handle_error(self, error: Exception) -> Any:
        """Handle an error"""
        # Log error
        self.error_log.append({
            'error': error,
            'timestamp': time.time()
        })
        
        # Find and execute handler
        for error_type, handler in self.handlers.items():
            if isinstance(error, error_type):
                return handler(error)
        
        # Default handling
        raise error
```

### Advanced Error Handler
```python
class AdvancedErrorHandler:
    def __init__(self):
        self.handlers = {}
        self.error_log = []
        self.recovery_strategies = {}
        self.monitors = []
    
    def add_handler(self,
                   error_type: type,
                   handler: Callable,
                   recovery_strategy: Callable = None) -> None:
        """Add an error handler with recovery strategy"""
        self.handlers[error_type] = handler
        if recovery_strategy:
            self.recovery_strategies[error_type] = recovery_strategy
    
    def add_monitor(self, monitor: Callable) -> None:
        """Add an error monitor"""
        self.monitors.append(monitor)
    
    def handle_error(self, error: Exception) -> Any:
        """Handle an error with monitoring and recovery"""
        # Log error
        error_entry = {
            'error': error,
            'timestamp': time.time()
        }
        self.error_log.append(error_entry)
        
        # Notify monitors
        for monitor in self.monitors:
            monitor(error_entry)
        
        # Find and execute handler
        for error_type, handler in self.handlers.items():
            if isinstance(error, error_type):
                try:
                    result = handler(error)
                    
                    # Try recovery if available
                    if error_type in self.recovery_strategies:
                        recovery_strategy = self.recovery_strategies[error_type]
                        recovery_strategy(error)
                    
                    return result
                    
                except Exception as e:
                    # Log recovery failure
                    self.error_log.append({
                        'error': e,
                        'timestamp': time.time(),
                        'original_error': error
                    })
                    raise
        
        # Default handling
        raise error
```

## Best Practices

1. **Task Orchestration**:
   - Dependency management
   - Task scheduling
   - Resource allocation
   - Error handling

2. **Workflow Management**:
   - Workflow design
   - State tracking
   - Error recovery
   - Monitoring

3. **State Management**:
   - State validation
   - Change tracking
   - Event notification
   - History management

4. **Error Handling**:
   - Error classification
   - Recovery strategies
   - Error monitoring
   - Logging

## Common Patterns

1. **Orchestrator Factory**:
```python
class OrchestratorFactory:
    @staticmethod
    def create_orchestrator(orchestrator_type: str, **kwargs) -> Any:
        if orchestrator_type == 'basic':
            return TaskOrchestrator(**kwargs)
        elif orchestrator_type == 'advanced':
            return AdvancedTaskOrchestrator(**kwargs)
        else:
            raise ValueError(f"Unknown orchestrator type: {orchestrator_type}")
```

2. **Orchestrator Monitor**:
```python
class OrchestratorMonitor:
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

3. **Orchestrator Validator**:
```python
class OrchestratorValidator:
    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
    
    def validate_task(self, task: Task) -> bool:
        """Validate task before execution"""
        if not isinstance(task, Task):
            return False
        if not task.name or not task.func:
            return False
        return True
    
    def validate_workflow(self, workflow: Workflow) -> bool:
        """Validate workflow before execution"""
        if not isinstance(workflow, Workflow):
            return False
        if not workflow.name or not workflow.tasks:
            return False
        return True
```

## Further Reading

- [Task Orchestration](https://arxiv.org/abs/2004.07213)
- [Workflow Management](https://arxiv.org/abs/2004.07213)
- [State Management](https://arxiv.org/abs/2004.07213)
- [Error Handling](https://arxiv.org/abs/2004.07213)
- [Distributed Systems](https://arxiv.org/abs/2004.07213) 