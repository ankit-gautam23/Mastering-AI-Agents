# Agent Patterns

This guide covers common patterns used in AI Agent development, including design patterns, implementation patterns, integration patterns, and testing patterns.

## Design Patterns

### Observer Pattern
```python
class Observable:
    def __init__(self):
        self.observers = []
    
    def add_observer(self, observer: callable) -> None:
        """Add an observer"""
        self.observers.append(observer)
    
    def remove_observer(self, observer: callable) -> None:
        """Remove an observer"""
        if observer in self.observers:
            self.observers.remove(observer)
    
    def notify_observers(self, *args, **kwargs) -> None:
        """Notify all observers"""
        for observer in self.observers:
            observer(*args, **kwargs)
```

### Strategy Pattern
```python
class ActionStrategy(ABC):
    @abstractmethod
    def select_action(self, state: Dict[str, Any]) -> Any:
        """Select an action based on state"""
        pass

class RandomStrategy(ActionStrategy):
    def select_action(self, state: Dict[str, Any]) -> Any:
        """Select a random action"""
        return random.choice(list(self.actions.values()))

class GreedyStrategy(ActionStrategy):
    def select_action(self, state: Dict[str, Any]) -> Any:
        """Select the best action"""
        return max(self.actions.values(), key=lambda x: x.evaluate(state))
```

### Command Pattern
```python
class Command(ABC):
    @abstractmethod
    def execute(self) -> Any:
        """Execute the command"""
        pass
    
    @abstractmethod
    def undo(self) -> None:
        """Undo the command"""
        pass

class ActionCommand(Command):
    def __init__(self, action: callable, *args, **kwargs):
        self.action = action
        self.args = args
        self.kwargs = kwargs
        self.result = None
    
    def execute(self) -> Any:
        """Execute the action"""
        self.result = self.action(*self.args, **self.kwargs)
        return self.result
    
    def undo(self) -> None:
        """Undo the action"""
        if hasattr(self.action, 'undo'):
            self.action.undo(*self.args, **self.kwargs)
```

## Implementation Patterns

### Factory Pattern
```python
class AgentFactory:
    @staticmethod
    def create_agent(agent_type: str, **kwargs) -> Any:
        """Create an agent of specified type"""
        if agent_type == 'simple_reflex':
            return SimpleReflexAgent(**kwargs)
        elif agent_type == 'model_based':
            return ModelBasedAgent(**kwargs)
        elif agent_type == 'goal_based':
            return GoalBasedAgent(**kwargs)
        elif agent_type == 'utility_based':
            return UtilityBasedAgent(**kwargs)
        elif agent_type == 'learning':
            return LearningAgent(**kwargs)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
```

### Builder Pattern
```python
class AgentBuilder:
    def __init__(self):
        self.agent = None
    
    def create_agent(self, agent_type: str) -> 'AgentBuilder':
        """Create a new agent"""
        self.agent = AgentFactory.create_agent(agent_type)
        return self
    
    def add_capability(self, capability: Any) -> 'AgentBuilder':
        """Add a capability to the agent"""
        self.agent.add_capability(capability)
        return self
    
    def set_state(self, state: Dict[str, Any]) -> 'AgentBuilder':
        """Set agent state"""
        self.agent.state = state
        return self
    
    def build(self) -> Any:
        """Build the agent"""
        return self.agent
```

### Singleton Pattern
```python
class AgentManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.agents = {}
        return cls._instance
    
    def register_agent(self, name: str, agent: Any) -> None:
        """Register an agent"""
        self.agents[name] = agent
    
    def get_agent(self, name: str) -> Any:
        """Get a registered agent"""
        return self.agents.get(name)
```

## Integration Patterns

### Adapter Pattern
```python
class AgentAdapter:
    def __init__(self, agent: Any):
        self.agent = agent
    
    def adapt_input(self, input_data: Any) -> Dict[str, Any]:
        """Adapt input for agent"""
        return {
            'state': input_data,
            'timestamp': time.time()
        }
    
    def adapt_output(self, output_data: Any) -> Any:
        """Adapt output from agent"""
        return {
            'result': output_data,
            'timestamp': time.time()
        }
    
    def process(self, input_data: Any) -> Any:
        """Process data through agent"""
        adapted_input = self.adapt_input(input_data)
        result = self.agent.run(adapted_input)
        return self.adapt_output(result)
```

### Facade Pattern
```python
class AgentFacade:
    def __init__(self):
        self.agent = None
        self.adapter = None
        self.monitor = None
    
    def initialize(self, agent_type: str, **kwargs) -> None:
        """Initialize the facade"""
        self.agent = AgentFactory.create_agent(agent_type, **kwargs)
        self.adapter = AgentAdapter(self.agent)
        self.monitor = AgentMonitor()
    
    def process(self, input_data: Any) -> Any:
        """Process data through the facade"""
        start_time = time.time()
        try:
            result = self.adapter.process(input_data)
            self.monitor.record_metric('success', 1.0)
        except Exception as e:
            result = None
            self.monitor.record_metric('error', 1.0)
        
        self.monitor.record_metric('processing_time', time.time() - start_time)
        return result
```

### Mediator Pattern
```python
class AgentMediator:
    def __init__(self):
        self.agents = {}
        self.routes = {}
    
    def register_agent(self, name: str, agent: Any) -> None:
        """Register an agent"""
        self.agents[name] = agent
    
    def add_route(self, source: str, target: str, condition: callable) -> None:
        """Add a routing rule"""
        if source not in self.routes:
            self.routes[source] = []
        self.routes[source].append({
            'target': target,
            'condition': condition
        })
    
    def route(self, source: str, data: Any) -> List[Any]:
        """Route data between agents"""
        results = []
        if source in self.routes:
            for route in self.routes[source]:
                if route['condition'](data):
                    target = route['target']
                    if target in self.agents:
                        result = self.agents[target].process(data)
                        results.append(result)
        return results
```

## Testing Patterns

### Test Case Pattern
```python
class AgentTestCase:
    def __init__(self, agent: Any):
        self.agent = agent
        self.test_cases = []
    
    def add_test_case(self, input_data: Any, expected_output: Any) -> None:
        """Add a test case"""
        self.test_cases.append({
            'input': input_data,
            'expected': expected_output
        })
    
    def run_tests(self) -> Dict[str, Any]:
        """Run all test cases"""
        results = {
            'total': len(self.test_cases),
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        for test_case in self.test_cases:
            try:
                result = self.agent.run(test_case['input'])
                if result == test_case['expected']:
                    results['passed'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append({
                        'input': test_case['input'],
                        'expected': test_case['expected'],
                        'actual': result
                    })
            except Exception as e:
                results['failed'] += 1
                results['errors'].append({
                    'input': test_case['input'],
                    'error': str(e)
                })
        
        return results
```

### Mock Pattern
```python
class MockAgent:
    def __init__(self):
        self.responses = {}
        self.calls = []
    
    def set_response(self, input_data: Any, response: Any) -> None:
        """Set response for input"""
        self.responses[str(input_data)] = response
    
    def run(self, input_data: Any) -> Any:
        """Run with mock response"""
        self.calls.append({
            'input': input_data,
            'timestamp': time.time()
        })
        return self.responses.get(str(input_data))
    
    def get_calls(self) -> List[Dict[str, Any]]:
        """Get call history"""
        return self.calls
```

### Performance Test Pattern
```python
class PerformanceTest:
    def __init__(self, agent: Any):
        self.agent = agent
        self.metrics = {}
    
    def run_test(self, input_data: Any, iterations: int = 1000) -> Dict[str, float]:
        """Run performance test"""
        times = []
        for _ in range(iterations):
            start_time = time.time()
            self.agent.run(input_data)
            times.append(time.time() - start_time)
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }
```

## Best Practices

1. **Design Patterns**:
   - Choose appropriate patterns
   - Keep patterns simple
   - Document pattern usage
   - Consider maintainability

2. **Implementation**:
   - Follow SOLID principles
   - Use dependency injection
   - Implement error handling
   - Add logging

3. **Integration**:
   - Define clear interfaces
   - Handle versioning
   - Implement fallbacks
   - Monitor performance

4. **Testing**:
   - Write unit tests
   - Add integration tests
   - Perform load testing
   - Monitor coverage

## Common Patterns

1. **Pattern Registry**:
```python
class PatternRegistry:
    def __init__(self):
        self.patterns = {}
    
    def register_pattern(self, name: str, pattern: Any) -> None:
        self.patterns[name] = pattern
    
    def get_pattern(self, name: str) -> Any:
        return self.patterns.get(name)
    
    def list_patterns(self) -> List[str]:
        return list(self.patterns.keys())
```

2. **Pattern Composition**:
```python
class PatternComposer:
    def __init__(self):
        self.patterns = []
    
    def add_pattern(self, pattern: Any) -> None:
        self.patterns.append(pattern)
    
    def compose(self, *args, **kwargs) -> Any:
        result = None
        for pattern in self.patterns:
            result = pattern.apply(result or args[0], *args[1:], **kwargs)
        return result
```

3. **Pattern Validator**:
```python
class PatternValidator:
    def __init__(self):
        self.rules = {}
    
    def add_rule(self, pattern: str, rule: callable) -> None:
        self.rules[pattern] = rule
    
    def validate(self, pattern: str, implementation: Any) -> bool:
        if pattern in self.rules:
            return self.rules[pattern](implementation)
        return True
```

## Further Reading

- [Design Patterns](https://refactoring.guru/design-patterns)
- [Implementation Patterns](https://martinfowler.com/books/implementation-patterns.html)
- [Integration Patterns](https://www.enterpriseintegrationpatterns.com/)
- [Testing Patterns](https://martinfowler.com/bliki/TestDouble.html)
- [Pattern Languages](https://www.hillside.net/plop/plop98/final_submissions/P51.pdf) 