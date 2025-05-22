# Hand-offs in Multi-Agent Systems

This guide covers the fundamental concepts and implementations of hand-offs in Multi-Agent Systems (MAS), including task transfer, state management, and coordination mechanisms.

## Basic Hand-off Implementation

### Task Transfer
```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    HANDED_OFF = "handed_off"

@dataclass
class Task:
    id: str
    description: str
    status: TaskStatus
    assigned_agent: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]

class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.hand_off_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def create_task(self, task_id: str, description: str, assigned_agent: str) -> Task:
        task = Task(
            id=task_id,
            description=description,
            status=TaskStatus.PENDING,
            assigned_agent=assigned_agent,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={}
        )
        self.tasks[task_id] = task
        return task
    
    def hand_off_task(
        self,
        task_id: str,
        from_agent: str,
        to_agent: str,
        reason: str
    ) -> Task:
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        if task.assigned_agent != from_agent:
            raise ValueError(f"Task {task_id} not assigned to agent {from_agent}")
        
        # Update task
        task.assigned_agent = to_agent
        task.status = TaskStatus.HANDED_OFF
        task.updated_at = datetime.now()
        
        # Record hand-off history
        if task_id not in self.hand_off_history:
            self.hand_off_history[task_id] = []
        
        self.hand_off_history[task_id].append({
            "from_agent": from_agent,
            "to_agent": to_agent,
            "timestamp": datetime.now(),
            "reason": reason
        })
        
        return task
    
    def get_task_history(self, task_id: str) -> List[Dict[str, Any]]:
        return self.hand_off_history.get(task_id, [])
```

### State Management
```python
class StateManager:
    def __init__(self):
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        self.state_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def update_agent_state(self, agent_id: str, state: Dict[str, Any]) -> None:
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = {}
        
        # Record state history
        if agent_id not in self.state_history:
            self.state_history[agent_id] = []
        
        self.state_history[agent_id].append({
            "timestamp": datetime.now(),
            "state": state.copy()
        })
        
        # Update current state
        self.agent_states[agent_id].update(state)
    
    def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        return self.agent_states.get(agent_id, {})
    
    def get_state_history(self, agent_id: str) -> List[Dict[str, Any]]:
        return self.state_history.get(agent_id, [])
    
    def transfer_state(self, from_agent: str, to_agent: str) -> None:
        if from_agent not in self.agent_states:
            raise ValueError(f"Agent {from_agent} not found")
        
        # Copy state to new agent
        self.agent_states[to_agent] = self.agent_states[from_agent].copy()
        
        # Record state transfer
        if to_agent not in self.state_history:
            self.state_history[to_agent] = []
        
        self.state_history[to_agent].append({
            "timestamp": datetime.now(),
            "state": self.agent_states[to_agent].copy(),
            "transferred_from": from_agent
        })
```

## Advanced Hand-off Mechanisms

### Coordinated Hand-off
```python
class CoordinatedHandoff:
    def __init__(self, task_manager: TaskManager, state_manager: StateManager):
        self.task_manager = task_manager
        self.state_manager = state_manager
        self.hand_off_coordinations: Dict[str, Dict[str, Any]] = {}
    
    def initiate_hand_off(
        self,
        task_id: str,
        from_agent: str,
        to_agent: str,
        reason: str
    ) -> Dict[str, Any]:
        # Create hand-off coordination
        coordination_id = f"handoff_{task_id}_{datetime.now().timestamp()}"
        self.hand_off_coordinations[coordination_id] = {
            "task_id": task_id,
            "from_agent": from_agent,
            "to_agent": to_agent,
            "status": "initiated",
            "reason": reason,
            "steps": []
        }
        
        return self.hand_off_coordinations[coordination_id]
    
    def execute_hand_off(self, coordination_id: str) -> Dict[str, Any]:
        if coordination_id not in self.hand_off_coordinations:
            raise ValueError(f"Coordination {coordination_id} not found")
        
        coordination = self.hand_off_coordinations[coordination_id]
        
        try:
            # Step 1: Transfer state
            self.state_manager.transfer_state(
                coordination["from_agent"],
                coordination["to_agent"]
            )
            coordination["steps"].append({
                "step": "state_transfer",
                "status": "completed",
                "timestamp": datetime.now()
            })
            
            # Step 2: Hand off task
            self.task_manager.hand_off_task(
                coordination["task_id"],
                coordination["from_agent"],
                coordination["to_agent"],
                coordination["reason"]
            )
            coordination["steps"].append({
                "step": "task_transfer",
                "status": "completed",
                "timestamp": datetime.now()
            })
            
            coordination["status"] = "completed"
            
        except Exception as e:
            coordination["status"] = "failed"
            coordination["error"] = str(e)
            coordination["steps"].append({
                "step": "hand_off",
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now()
            })
        
        return coordination
```

### Hand-off Validation
```python
class HandoffValidator:
    def __init__(self):
        self.validation_rules: Dict[str, List[callable]] = {}
    
    def add_validation_rule(self, rule_name: str, rule_func: callable) -> None:
        if rule_name not in self.validation_rules:
            self.validation_rules[rule_name] = []
        self.validation_rules[rule_name].append(rule_func)
    
    def validate_hand_off(
        self,
        task: Task,
        from_agent: str,
        to_agent: str,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Run all validation rules
        for rule_name, rules in self.validation_rules.items():
            for rule in rules:
                try:
                    result = rule(task, from_agent, to_agent, state)
                    if not result["valid"]:
                        validation_results["valid"] = False
                        validation_results["errors"].extend(result["errors"])
                    if result.get("warnings"):
                        validation_results["warnings"].extend(result["warnings"])
                except Exception as e:
                    validation_results["valid"] = False
                    validation_results["errors"].append(f"Rule {rule_name} failed: {str(e)}")
        
        return validation_results
```

## Best Practices

1. **Task Management**:
   - Maintain task history
   - Track hand-off reasons
   - Monitor task progress

2. **State Management**:
   - Keep state history
   - Ensure state consistency
   - Handle state conflicts

3. **Hand-off Coordination**:
   - Implement validation
   - Handle failures
   - Maintain audit trail

4. **Performance Optimization**:
   - Minimize state transfer
   - Optimize validation
   - Handle concurrent hand-offs

## Common Patterns

1. **Hand-off Pipeline**:
```python
class HandoffPipeline:
    def __init__(self):
        self.task_manager = TaskManager()
        self.state_manager = StateManager()
        self.coordinator = CoordinatedHandoff(self.task_manager, self.state_manager)
        self.validator = HandoffValidator()
    
    def execute_hand_off(
        self,
        task_id: str,
        from_agent: str,
        to_agent: str,
        reason: str
    ) -> Dict[str, Any]:
        # Get current task and state
        task = self.task_manager.tasks[task_id]
        state = self.state_manager.get_agent_state(from_agent)
        
        # Validate hand-off
        validation = self.validator.validate_hand_off(task, from_agent, to_agent, state)
        if not validation["valid"]:
            return {
                "status": "failed",
                "reason": "validation_failed",
                "errors": validation["errors"]
            }
        
        # Initiate and execute hand-off
        coordination = self.coordinator.initiate_hand_off(task_id, from_agent, to_agent, reason)
        result = self.coordinator.execute_hand_off(coordination["coordination_id"])
        
        return {
            "status": result["status"],
            "coordination": result,
            "validation": validation
        }
```

2. **Hand-off Monitor**:
```python
class HandoffMonitor:
    def __init__(self, pipeline: HandoffPipeline):
        self.pipeline = pipeline
        self.metrics = {
            "hand_off_count": [],
            "success_rate": [],
            "validation_errors": [],
            "hand_off_duration": []
        }
    
    def record_metrics(self, hand_off_result: Dict[str, Any]) -> None:
        # Record hand-off count
        self.metrics["hand_off_count"].append(1)
        
        # Record success rate
        success = hand_off_result["status"] == "completed"
        self.metrics["success_rate"].append(1 if success else 0)
        
        # Record validation errors
        if "validation" in hand_off_result:
            self.metrics["validation_errors"].append(
                len(hand_off_result["validation"]["errors"])
            )
        
        # Record hand-off duration
        if "coordination" in hand_off_result:
            steps = hand_off_result["coordination"]["steps"]
            if len(steps) >= 2:
                duration = (steps[-1]["timestamp"] - steps[0]["timestamp"]).total_seconds()
                self.metrics["hand_off_duration"].append(duration)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for metric, values in self.metrics.items():
            if not values:
                continue
            
            summary[metric] = {
                "total": sum(values),
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values)
            }
        
        return summary
```

## Further Reading

- [Task Transfer](https://arxiv.org/abs/2004.07213)
- [State Management](https://arxiv.org/abs/2004.07213)
- [Hand-off Coordination](https://arxiv.org/abs/2004.07213)
- [Validation Strategies](https://arxiv.org/abs/2004.07213)
- [Hand-off Patterns](https://arxiv.org/abs/2004.07213) 