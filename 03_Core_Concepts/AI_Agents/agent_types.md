# Agent Types

This guide covers different types of AI Agents, their characteristics, and implementations.

## Simple Reflex Agents

### Basic Implementation
```python
class SimpleReflexAgent:
    def __init__(self):
        self.rules = {}
    
    def add_rule(self, condition: callable, action: callable) -> None:
        """Add a condition-action rule"""
        self.rules[condition] = action
    
    def perceive(self, environment: Dict[str, Any]) -> None:
        """Process environment information"""
        self.current_state = environment
    
    def think(self) -> None:
        """Match conditions to rules"""
        self.selected_action = None
        for condition, action in self.rules.items():
            if condition(self.current_state):
                self.selected_action = action
                break
    
    def act(self) -> Any:
        """Execute selected action"""
        if self.selected_action:
            return self.selected_action(self.current_state)
        return None
```

### Rule-based System
```python
class RuleBasedSystem:
    def __init__(self):
        self.rules = []
        self.facts = set()
    
    def add_rule(self, antecedents: List[str], consequent: str) -> None:
        """Add a rule with antecedents and consequent"""
        self.rules.append({
            'antecedents': set(antecedents),
            'consequent': consequent
        })
    
    def add_fact(self, fact: str) -> None:
        """Add a fact to the system"""
        self.facts.add(fact)
    
    def infer(self) -> Set[str]:
        """Apply rules to derive new facts"""
        new_facts = set()
        for rule in self.rules:
            if rule['antecedents'].issubset(self.facts):
                new_facts.add(rule['consequent'])
        return new_facts
```

## Model-based Agents

### World Model
```python
class WorldModel:
    def __init__(self):
        self.model = {}
        self.transitions = {}
    
    def update_model(self, state: Dict[str, Any], action: str, next_state: Dict[str, Any]) -> None:
        """Update model with observed transition"""
        state_key = str(state)
        if state_key not in self.transitions:
            self.transitions[state_key] = {}
        if action not in self.transitions[state_key]:
            self.transitions[state_key][action] = []
        self.transitions[state_key][action].append(next_state)
    
    def predict_next_state(self, state: Dict[str, Any], action: str) -> Dict[str, Any]:
        """Predict next state given current state and action"""
        state_key = str(state)
        if state_key in self.transitions and action in self.transitions[state_key]:
            transitions = self.transitions[state_key][action]
            return random.choice(transitions)
        return state
```

### Model-based Agent
```python
class ModelBasedAgent:
    def __init__(self):
        self.world_model = WorldModel()
        self.current_state = {}
    
    def perceive(self, environment: Dict[str, Any]) -> None:
        """Update current state and world model"""
        self.current_state = environment
    
    def think(self) -> None:
        """Use world model to plan actions"""
        possible_actions = self.get_possible_actions()
        best_action = None
        best_value = float('-inf')
        
        for action in possible_actions:
            next_state = self.world_model.predict_next_state(self.current_state, action)
            value = self.evaluate_state(next_state)
            if value > best_value:
                best_value = value
                best_action = action
        
        self.selected_action = best_action
    
    def act(self) -> Any:
        """Execute selected action"""
        if self.selected_action:
            return self.selected_action(self.current_state)
        return None
```

## Goal-based Agents

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
    
    def is_goal_relevant(self, goal: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """Check if goal is relevant to current state"""
        return True  # Override in subclasses
```

### Goal-based Agent
```python
class GoalBasedAgent:
    def __init__(self):
        self.goal_manager = GoalManager()
        self.planner = Planner()
    
    def perceive(self, environment: Dict[str, Any]) -> None:
        """Update current state and select goal"""
        self.current_state = environment
        self.current_goal = self.goal_manager.select_goal(environment)
    
    def think(self) -> None:
        """Plan actions to achieve goal"""
        if self.current_goal:
            self.plan = self.planner.plan(
                self.current_state,
                self.current_goal
            )
            self.selected_action = self.plan[0] if self.plan else None
        else:
            self.selected_action = None
    
    def act(self) -> Any:
        """Execute selected action"""
        if self.selected_action:
            return self.selected_action(self.current_state)
        return None
```

## Utility-based Agents

### Utility Function
```python
class UtilityFunction:
    def __init__(self):
        self.weights = {}
    
    def add_feature(self, name: str, weight: float) -> None:
        """Add a feature with weight"""
        self.weights[name] = weight
    
    def evaluate(self, state: Dict[str, Any]) -> float:
        """Evaluate state utility"""
        utility = 0.0
        for feature, weight in self.weights.items():
            if feature in state:
                utility += weight * state[feature]
        return utility
```

### Utility-based Agent
```python
class UtilityBasedAgent:
    def __init__(self):
        self.utility_function = UtilityFunction()
        self.current_state = {}
    
    def perceive(self, environment: Dict[str, Any]) -> None:
        """Update current state"""
        self.current_state = environment
    
    def think(self) -> None:
        """Select action with highest utility"""
        possible_actions = self.get_possible_actions()
        best_action = None
        best_utility = float('-inf')
        
        for action in possible_actions:
            next_state = self.predict_next_state(action)
            utility = self.utility_function.evaluate(next_state)
            if utility > best_utility:
                best_utility = utility
                best_action = action
        
        self.selected_action = best_action
    
    def act(self) -> Any:
        """Execute selected action"""
        if self.selected_action:
            return self.selected_action(self.current_state)
        return None
```

## Learning Agents

### Learning Component
```python
class LearningComponent:
    def __init__(self):
        self.experiences = []
        self.model = None
    
    def add_experience(self, state: Dict[str, Any], action: str, 
                      reward: float, next_state: Dict[str, Any]) -> None:
        """Add experience to memory"""
        self.experiences.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state
        })
    
    def learn(self) -> None:
        """Update model based on experiences"""
        if len(self.experiences) > 0:
            self.update_model(self.experiences)
    
    def update_model(self, experiences: List[Dict[str, Any]]) -> None:
        """Update model with experiences"""
        pass  # Override in subclasses
```

### Learning Agent
```python
class LearningAgent:
    def __init__(self):
        self.learning_component = LearningComponent()
        self.current_state = {}
    
    def perceive(self, environment: Dict[str, Any]) -> None:
        """Update current state"""
        self.current_state = environment
    
    def think(self) -> None:
        """Select action using learned model"""
        self.selected_action = self.learning_component.select_action(self.current_state)
    
    def act(self) -> Any:
        """Execute action and learn from result"""
        if self.selected_action:
            result = self.selected_action(self.current_state)
            reward = self.evaluate_result(result)
            next_state = self.get_next_state(result)
            
            self.learning_component.add_experience(
                self.current_state,
                self.selected_action,
                reward,
                next_state
            )
            
            self.learning_component.learn()
            return result
        return None
```

## Best Practices

1. **Agent Selection**:
   - Match agent type to problem
   - Consider complexity
   - Evaluate requirements
   - Plan for scalability

2. **Implementation**:
   - Clear interfaces
   - Modular design
   - Error handling
   - Testing strategy

3. **Learning**:
   - Experience collection
   - Model updates
   - Performance monitoring
   - Adaptation strategy

4. **Integration**:
   - Component communication
   - State management
   - Action coordination
   - Resource sharing

## Common Patterns

1. **Agent Factory**:
```python
class AgentFactory:
    @staticmethod
    def create_agent(agent_type: str, **kwargs) -> Any:
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

2. **Agent Composition**:
```python
class CompositeAgent:
    def __init__(self):
        self.agents = []
    
    def add_agent(self, agent: Any) -> None:
        self.agents.append(agent)
    
    def run(self, environment: Dict[str, Any]) -> Any:
        results = []
        for agent in self.agents:
            result = agent.run(environment)
            results.append(result)
        return self.combine_results(results)
```

3. **Agent Monitoring**:
```python
class AgentMonitor:
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

- [Agent Types in AI](https://www.sciencedirect.com/science/article/pii/S0004370201001012)
- [Learning Agents](https://www.cs.cmu.edu/~tom/pubs/MachineLearning.pdf)
- [Goal-based Agents](https://www.sciencedirect.com/science/article/pii/S0004370201001012)
- [Utility-based Agents](https://www.cs.cmu.edu/~tom/pubs/MachineLearning.pdf)
- [Model-based Agents](https://www.sciencedirect.com/science/article/pii/S0004370201001012) 