# Planning

This guide covers the fundamental concepts and implementations of planning in AI agent frameworks, including action planning, goal planning, resource planning, and planning optimization.

## Action Planning

### Basic Action Planner
```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Action:
    name: str
    preconditions: List[str]
    effects: List[str]
    cost: float = 1.0

class ActionPlanner:
    def __init__(self):
        self.actions = {}
        self.state = set()
    
    def add_action(self, action: Action) -> None:
        """Add an action to the planner"""
        self.actions[action.name] = action
    
    def set_state(self, state: List[str]) -> None:
        """Set the current state"""
        self.state = set(state)
    
    def is_applicable(self, action: Action) -> bool:
        """Check if an action is applicable in current state"""
        return all(pre in self.state for pre in action.preconditions)
    
    def apply_action(self, action: Action) -> None:
        """Apply an action to the current state"""
        if not self.is_applicable(action):
            raise ValueError(f"Action {action.name} not applicable")
        
        # Remove preconditions
        for pre in action.preconditions:
            self.state.remove(pre)
        
        # Add effects
        for effect in action.effects:
            self.state.add(effect)
    
    def plan(self, goal: List[str]) -> List[str]:
        """Generate a plan to achieve the goal"""
        goal_set = set(goal)
        plan = []
        
        while not goal_set.issubset(self.state):
            # Find applicable actions
            applicable_actions = [
                action for action in self.actions.values()
                if self.is_applicable(action)
            ]
            
            if not applicable_actions:
                raise ValueError("No applicable actions found")
            
            # Select best action
            best_action = min(
                applicable_actions,
                key=lambda a: len(set(a.effects) & goal_set)
            )
            
            # Apply action
            self.apply_action(best_action)
            plan.append(best_action.name)
        
        return plan
```

### Advanced Action Planner
```python
class AdvancedActionPlanner:
    def __init__(self):
        self.actions = {}
        self.state = set()
        self.heuristics = {}
        self.constraints = []
    
    def add_action(self, action: Action) -> None:
        """Add an action to the planner"""
        self.actions[action.name] = action
    
    def add_heuristic(self, name: str, heuristic: Callable) -> None:
        """Add a heuristic function"""
        self.heuristics[name] = heuristic
    
    def add_constraint(self, constraint: Callable) -> None:
        """Add a planning constraint"""
        self.constraints.append(constraint)
    
    def is_applicable(self, action: Action) -> bool:
        """Check if an action is applicable in current state"""
        if not all(pre in self.state for pre in action.preconditions):
            return False
        
        # Check constraints
        for constraint in self.constraints:
            if not constraint(action, self.state):
                return False
        
        return True
    
    def apply_action(self, action: Action) -> None:
        """Apply an action to the current state"""
        if not self.is_applicable(action):
            raise ValueError(f"Action {action.name} not applicable")
        
        # Remove preconditions
        for pre in action.preconditions:
            self.state.remove(pre)
        
        # Add effects
        for effect in action.effects:
            self.state.add(effect)
    
    def plan(self, goal: List[str], heuristic_name: str = None) -> List[str]:
        """Generate a plan to achieve the goal using heuristics"""
        goal_set = set(goal)
        plan = []
        
        while not goal_set.issubset(self.state):
            # Find applicable actions
            applicable_actions = [
                action for action in self.actions.values()
                if self.is_applicable(action)
            ]
            
            if not applicable_actions:
                raise ValueError("No applicable actions found")
            
            # Select best action using heuristic
            if heuristic_name and heuristic_name in self.heuristics:
                heuristic = self.heuristics[heuristic_name]
                best_action = min(
                    applicable_actions,
                    key=lambda a: heuristic(a, self.state, goal_set)
                )
            else:
                best_action = min(
                    applicable_actions,
                    key=lambda a: len(set(a.effects) & goal_set)
                )
            
            # Apply action
            self.apply_action(best_action)
            plan.append(best_action.name)
        
        return plan
```

## Goal Planning

### Basic Goal Planner
```python
@dataclass
class Goal:
    name: str
    conditions: List[str]
    priority: int = 0

class GoalPlanner:
    def __init__(self):
        self.goals = []
        self.state = set()
    
    def add_goal(self, goal: Goal) -> None:
        """Add a goal to the planner"""
        self.goals.append(goal)
        self.goals.sort(key=lambda g: g.priority, reverse=True)
    
    def set_state(self, state: List[str]) -> None:
        """Set the current state"""
        self.state = set(state)
    
    def is_achieved(self, goal: Goal) -> bool:
        """Check if a goal is achieved"""
        return all(cond in self.state for cond in goal.conditions)
    
    def get_active_goals(self) -> List[Goal]:
        """Get goals that are not yet achieved"""
        return [
            goal for goal in self.goals
            if not self.is_achieved(goal)
        ]
    
    def update_state(self, new_state: List[str]) -> None:
        """Update the current state"""
        self.state.update(new_state)
```

### Advanced Goal Planner
```python
class AdvancedGoalPlanner:
    def __init__(self):
        self.goals = []
        self.state = set()
        self.goal_handlers = {}
        self.state_observers = []
    
    def add_goal(self, goal: Goal) -> None:
        """Add a goal to the planner"""
        self.goals.append(goal)
        self.goals.sort(key=lambda g: g.priority, reverse=True)
    
    def add_goal_handler(self, goal_name: str, handler: Callable) -> None:
        """Add a handler for goal achievement"""
        self.goal_handlers[goal_name] = handler
    
    def add_state_observer(self, observer: Callable) -> None:
        """Add a state observer"""
        self.state_observers.append(observer)
    
    def is_achieved(self, goal: Goal) -> bool:
        """Check if a goal is achieved"""
        return all(cond in self.state for cond in goal.conditions)
    
    def get_active_goals(self) -> List[Goal]:
        """Get goals that are not yet achieved"""
        return [
            goal for goal in self.goals
            if not self.is_achieved(goal)
        ]
    
    def update_state(self, new_state: List[str]) -> None:
        """Update the current state and notify observers"""
        self.state.update(new_state)
        
        # Notify observers
        for observer in self.state_observers:
            observer(self.state)
        
        # Check goals and run handlers
        for goal in self.goals:
            if self.is_achieved(goal) and goal.name in self.goal_handlers:
                self.goal_handlers[goal.name](goal)
```

## Resource Planning

### Basic Resource Planner
```python
@dataclass
class Resource:
    name: str
    capacity: float
    current: float = 0.0

class ResourcePlanner:
    def __init__(self):
        self.resources = {}
    
    def add_resource(self, resource: Resource) -> None:
        """Add a resource to the planner"""
        self.resources[resource.name] = resource
    
    def allocate(self, resource_name: str, amount: float) -> bool:
        """Allocate a resource"""
        if resource_name not in self.resources:
            raise ValueError(f"Resource {resource_name} not found")
        
        resource = self.resources[resource_name]
        if resource.current + amount > resource.capacity:
            return False
        
        resource.current += amount
        return True
    
    def deallocate(self, resource_name: str, amount: float) -> None:
        """Deallocate a resource"""
        if resource_name not in self.resources:
            raise ValueError(f"Resource {resource_name} not found")
        
        resource = self.resources[resource_name]
        if resource.current < amount:
            raise ValueError(f"Not enough {resource_name}")
        
        resource.current -= amount
    
    def get_available(self, resource_name: str) -> float:
        """Get available amount of a resource"""
        if resource_name not in self.resources:
            raise ValueError(f"Resource {resource_name} not found")
        
        resource = self.resources[resource_name]
        return resource.capacity - resource.current
```

### Advanced Resource Planner
```python
class AdvancedResourcePlanner:
    def __init__(self):
        self.resources = {}
        self.allocation_history = []
        self.resource_observers = {}
    
    def add_resource(self, resource: Resource) -> None:
        """Add a resource to the planner"""
        self.resources[resource.name] = resource
        self.resource_observers[resource.name] = []
    
    def add_resource_observer(self,
                            resource_name: str,
                            observer: Callable) -> None:
        """Add an observer for a resource"""
        if resource_name not in self.resource_observers:
            raise ValueError(f"Resource {resource_name} not found")
        self.resource_observers[resource_name].append(observer)
    
    def allocate(self,
                resource_name: str,
                amount: float,
                requester: str = None) -> bool:
        """Allocate a resource with tracking"""
        if resource_name not in self.resources:
            raise ValueError(f"Resource {resource_name} not found")
        
        resource = self.resources[resource_name]
        if resource.current + amount > resource.capacity:
            return False
        
        resource.current += amount
        
        # Record allocation
        self.allocation_history.append({
            'resource': resource_name,
            'amount': amount,
            'requester': requester,
            'timestamp': time.time()
        })
        
        # Notify observers
        for observer in self.resource_observers[resource_name]:
            observer(resource)
        
        return True
    
    def deallocate(self,
                  resource_name: str,
                  amount: float,
                  requester: str = None) -> None:
        """Deallocate a resource with tracking"""
        if resource_name not in self.resources:
            raise ValueError(f"Resource {resource_name} not found")
        
        resource = self.resources[resource_name]
        if resource.current < amount:
            raise ValueError(f"Not enough {resource_name}")
        
        resource.current -= amount
        
        # Record deallocation
        self.allocation_history.append({
            'resource': resource_name,
            'amount': -amount,
            'requester': requester,
            'timestamp': time.time()
        })
        
        # Notify observers
        for observer in self.resource_observers[resource_name]:
            observer(resource)
    
    def get_allocation_history(self,
                             resource_name: str = None,
                             start_time: float = None,
                             end_time: float = None) -> List[Dict[str, Any]]:
        """Get resource allocation history"""
        history = self.allocation_history
        
        if resource_name:
            history = [
                entry for entry in history
                if entry['resource'] == resource_name
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

## Planning Optimization

### Basic Planning Optimizer
```python
class PlanningOptimizer:
    def __init__(self):
        self.metrics = {}
    
    def record_metric(self, name: str, value: float) -> None:
        """Record a planning metric"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_statistics(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        values = self.metrics.get(name, [])
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    def optimize_plan(self, plan: List[str]) -> List[str]:
        """Optimize a plan"""
        # Remove redundant actions
        optimized = []
        for action in plan:
            if action not in optimized:
                optimized.append(action)
        
        return optimized
```

### Advanced Planning Optimizer
```python
class AdvancedPlanningOptimizer:
    def __init__(self):
        self.metrics = {}
        self.optimization_strategies = {}
        self.constraints = []
    
    def add_optimization_strategy(self,
                                name: str,
                                strategy: Callable) -> None:
        """Add an optimization strategy"""
        self.optimization_strategies[name] = strategy
    
    def add_constraint(self, constraint: Callable) -> None:
        """Add an optimization constraint"""
        self.constraints.append(constraint)
    
    def record_metric(self, name: str, value: float) -> None:
        """Record a planning metric"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_statistics(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        values = self.metrics.get(name, [])
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    def optimize_plan(self,
                     plan: List[str],
                     strategy_name: str = None) -> List[str]:
        """Optimize a plan using strategies"""
        optimized = plan
        
        # Apply optimization strategy
        if strategy_name and strategy_name in self.optimization_strategies:
            strategy = self.optimization_strategies[strategy_name]
            optimized = strategy(optimized)
        
        # Check constraints
        for constraint in self.constraints:
            if not constraint(optimized):
                raise ValueError("Optimized plan violates constraints")
        
        return optimized
```

## Best Practices

1. **Action Planning**:
   - Action representation
   - State management
   - Goal achievement
   - Plan generation

2. **Goal Planning**:
   - Goal prioritization
   - State tracking
   - Goal achievement
   - Goal handling

3. **Resource Planning**:
   - Resource allocation
   - Capacity management
   - History tracking
   - Resource monitoring

4. **Planning Optimization**:
   - Metric collection
   - Strategy selection
   - Constraint satisfaction
   - Plan validation

## Common Patterns

1. **Planner Factory**:
```python
class PlannerFactory:
    @staticmethod
    def create_planner(planner_type: str, **kwargs) -> Any:
        if planner_type == 'action':
            return ActionPlanner(**kwargs)
        elif planner_type == 'goal':
            return GoalPlanner(**kwargs)
        elif planner_type == 'resource':
            return ResourcePlanner(**kwargs)
        else:
            raise ValueError(f"Unknown planner type: {planner_type}")
```

2. **Planner Monitor**:
```python
class PlannerMonitor:
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

3. **Planner Validator**:
```python
class PlannerValidator:
    def __init__(self, planner: Any):
        self.planner = planner
    
    def validate_plan(self, plan: List[str]) -> bool:
        """Validate a plan"""
        if not plan:
            return False
        
        # Check plan structure
        for action in plan:
            if not isinstance(action, str):
                return False
        
        return True
    
    def validate_goal(self, goal: Goal) -> bool:
        """Validate a goal"""
        if not isinstance(goal, Goal):
            return False
        if not goal.name or not goal.conditions:
            return False
        return True
```

## Further Reading

- [Action Planning](https://arxiv.org/abs/2004.07213)
- [Goal Planning](https://arxiv.org/abs/2004.07213)
- [Resource Planning](https://arxiv.org/abs/2004.07213)
- [Planning Optimization](https://arxiv.org/abs/2004.07213)
- [Automated Planning](https://arxiv.org/abs/2004.07213) 