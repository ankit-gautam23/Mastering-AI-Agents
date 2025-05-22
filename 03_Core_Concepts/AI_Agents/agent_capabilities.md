# Agent Capabilities

This guide covers the core capabilities of AI Agents, including perception, reasoning, planning, learning, and communication.

## Perception

### Sensor System
```python
class SensorSystem:
    def __init__(self):
        self.sensors = {}
        self.data_buffer = {}
    
    def add_sensor(self, name: str, sensor: callable) -> None:
        """Add a sensor to the system"""
        self.sensors[name] = sensor
        self.data_buffer[name] = []
    
    def sense(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Collect data from all sensors"""
        observations = {}
        for name, sensor in self.sensors.items():
            try:
                data = sensor(environment)
                self.data_buffer[name].append(data)
                observations[name] = data
            except Exception as e:
                print(f"Error in sensor {name}: {e}")
        return observations
    
    def get_sensor_history(self, name: str, n: int = 1) -> List[Any]:
        """Get recent data from a sensor"""
        return self.data_buffer[name][-n:]
```

### Perception Processor
```python
class PerceptionProcessor:
    def __init__(self):
        self.processors = {}
    
    def add_processor(self, name: str, processor: callable) -> None:
        """Add a data processor"""
        self.processors[name] = processor
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process sensor data"""
        processed_data = {}
        for name, processor in self.processors.items():
            if name in data:
                processed_data[name] = processor(data[name])
        return processed_data
```

## Reasoning

### Rule Engine
```python
class RuleEngine:
    def __init__(self):
        self.rules = []
        self.facts = set()
    
    def add_rule(self, condition: callable, action: callable) -> None:
        """Add a rule to the engine"""
        self.rules.append({
            'condition': condition,
            'action': action
        })
    
    def add_fact(self, fact: Any) -> None:
        """Add a fact to the engine"""
        self.facts.add(fact)
    
    def reason(self) -> List[Any]:
        """Apply rules to facts"""
        results = []
        for rule in self.rules:
            if rule['condition'](self.facts):
                result = rule['action'](self.facts)
                results.append(result)
        return results
```

### Inference Engine
```python
class InferenceEngine:
    def __init__(self):
        self.knowledge_base = {}
        self.inference_rules = []
    
    def add_knowledge(self, key: str, value: Any) -> None:
        """Add knowledge to the base"""
        self.knowledge_base[key] = value
    
    def add_rule(self, rule: callable) -> None:
        """Add an inference rule"""
        self.inference_rules.append(rule)
    
    def infer(self, query: Any) -> List[Any]:
        """Make inferences from knowledge base"""
        results = []
        for rule in self.inference_rules:
            try:
                result = rule(self.knowledge_base, query)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error in inference rule: {e}")
        return results
```

## Planning

### Planner
```python
class Planner:
    def __init__(self):
        self.actions = {}
        self.preconditions = {}
        self.effects = {}
    
    def add_action(self, name: str, preconditions: List[str], 
                  effects: List[str]) -> None:
        """Add an action to the planner"""
        self.actions[name] = True
        self.preconditions[name] = set(preconditions)
        self.effects[name] = set(effects)
    
    def plan(self, initial_state: Set[str], goal_state: Set[str]) -> List[str]:
        """Generate a plan to achieve goal"""
        current_state = initial_state.copy()
        plan = []
        
        while not goal_state.issubset(current_state):
            applicable_actions = self.get_applicable_actions(current_state)
            if not applicable_actions:
                return None
            
            action = self.select_action(applicable_actions, goal_state)
            if not action:
                return None
            
            plan.append(action)
            current_state.update(self.effects[action])
        
        return plan
    
    def get_applicable_actions(self, state: Set[str]) -> List[str]:
        """Get actions whose preconditions are satisfied"""
        return [action for action, preconds in self.preconditions.items()
                if preconds.issubset(state)]
```

### Goal Manager
```python
class GoalManager:
    def __init__(self):
        self.goals = []
        self.current_goal = None
    
    def add_goal(self, goal: Dict[str, Any], priority: float = 1.0) -> None:
        """Add a goal with priority"""
        self.goals.append({
            'goal': goal,
            'priority': priority
        })
        self.goals.sort(key=lambda x: x['priority'], reverse=True)
    
    def select_goal(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Select most appropriate goal for current state"""
        for goal_info in self.goals:
            if self.is_goal_relevant(goal_info['goal'], state):
                return goal_info['goal']
        return None
```

## Learning

### Experience Collector
```python
class ExperienceCollector:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.experiences = []
    
    def add_experience(self, state: Dict[str, Any], action: str,
                      reward: float, next_state: Dict[str, Any]) -> None:
        """Add an experience to memory"""
        self.experiences.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state
        })
        
        if len(self.experiences) > self.capacity:
            self.experiences.pop(0)
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a batch of experiences"""
        return random.sample(self.experiences, min(batch_size, len(self.experiences)))
```

### Learning System
```python
class LearningSystem:
    def __init__(self):
        self.model = None
        self.experience_collector = ExperienceCollector()
    
    def update(self, state: Dict[str, Any], action: str,
              reward: float, next_state: Dict[str, Any]) -> None:
        """Update learning system with new experience"""
        self.experience_collector.add_experience(state, action, reward, next_state)
        self.learn()
    
    def learn(self) -> None:
        """Learn from collected experiences"""
        if len(self.experience_collector.experiences) > 0:
            batch = self.experience_collector.sample(32)
            self.update_model(batch)
    
    def update_model(self, batch: List[Dict[str, Any]]) -> None:
        """Update model with batch of experiences"""
        pass  # Override in subclasses
```

## Communication

### Message System
```python
class MessageSystem:
    def __init__(self):
        self.messages = []
        self.handlers = {}
    
    def send_message(self, sender: str, receiver: str,
                    content: Any, priority: float = 1.0) -> None:
        """Send a message"""
        self.messages.append({
            'sender': sender,
            'receiver': receiver,
            'content': content,
            'priority': priority,
            'timestamp': time.time()
        })
    
    def register_handler(self, message_type: str, handler: callable) -> None:
        """Register a message handler"""
        self.handlers[message_type] = handler
    
    def process_messages(self) -> None:
        """Process pending messages"""
        self.messages.sort(key=lambda x: x['priority'], reverse=True)
        for message in self.messages:
            if message['receiver'] in self.handlers:
                self.handlers[message['receiver']](message)
```

### Communication Protocol
```python
class CommunicationProtocol:
    def __init__(self):
        self.protocols = {}
    
    def register_protocol(self, name: str, protocol: Dict[str, callable]) -> None:
        """Register a communication protocol"""
        self.protocols[name] = protocol
    
    def encode_message(self, protocol: str, message: Any) -> str:
        """Encode message using protocol"""
        if protocol in self.protocols and 'encode' in self.protocols[protocol]:
            return self.protocols[protocol]['encode'](message)
        return str(message)
    
    def decode_message(self, protocol: str, message: str) -> Any:
        """Decode message using protocol"""
        if protocol in self.protocols and 'decode' in self.protocols[protocol]:
            return self.protocols[protocol]['decode'](message)
        return message
```

## Best Practices

1. **Perception**:
   - Multiple sensors
   - Data validation
   - Error handling
   - Data processing

2. **Reasoning**:
   - Clear rules
   - Efficient inference
   - Knowledge management
   - Conflict resolution

3. **Planning**:
   - Goal prioritization
   - Action validation
   - Plan monitoring
   - Adaptation

4. **Learning**:
   - Experience collection
   - Model updates
   - Performance evaluation
   - Knowledge transfer

5. **Communication**:
   - Message queuing
   - Protocol handling
   - Error recovery
   - Security

## Common Patterns

1. **Capability Manager**:
```python
class CapabilityManager:
    def __init__(self):
        self.capabilities = {}
    
    def register_capability(self, name: str, capability: Any) -> None:
        self.capabilities[name] = capability
    
    def get_capability(self, name: str) -> Any:
        return self.capabilities.get(name)
    
    def has_capability(self, name: str) -> bool:
        return name in self.capabilities
```

2. **Capability Composition**:
```python
class CompositeCapability:
    def __init__(self):
        self.capabilities = []
    
    def add_capability(self, capability: Any) -> None:
        self.capabilities.append(capability)
    
    def execute(self, *args, **kwargs) -> Any:
        results = []
        for capability in self.capabilities:
            result = capability.execute(*args, **kwargs)
            results.append(result)
        return self.combine_results(results)
```

3. **Capability Monitoring**:
```python
class CapabilityMonitor:
    def __init__(self):
        self.metrics = {}
    
    def record_metric(self, capability: str, metric: str, value: float) -> None:
        if capability not in self.metrics:
            self.metrics[capability] = {}
        if metric not in self.metrics[capability]:
            self.metrics[capability][metric] = []
        self.metrics[capability][metric].append(value)
    
    def get_statistics(self, capability: str, metric: str) -> Dict[str, float]:
        values = self.metrics.get(capability, {}).get(metric, [])
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

- [Agent Perception](https://www.sciencedirect.com/science/article/pii/S0004370201001012)
- [Reasoning Systems](https://www.cs.cmu.edu/~tom/pubs/MachineLearning.pdf)
- [Planning in AI](https://www.sciencedirect.com/science/article/pii/S0004370201001012)
- [Learning Systems](https://www.cs.cmu.edu/~tom/pubs/MachineLearning.pdf)
- [Agent Communication](https://www.sciencedirect.com/science/article/pii/S0004370201001012) 