# Types of Multi-Agent Systems

This guide covers the fundamental concepts and implementations of different types of Multi-Agent Systems (MAS), including their architectures, characteristics, and use cases.

## Basic MAS Types

### Hierarchical MAS
```python
from typing import List, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Agent:
    id: str
    role: str
    capabilities: List[str]
    parent: Optional[str] = None
    children: List[str] = None

class HierarchicalMAS:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.hierarchy_levels: Dict[str, List[str]] = {}
    
    def add_agent(self, agent: Agent) -> None:
        self.agents[agent.id] = agent
        
        # Update hierarchy
        if agent.parent:
            if agent.parent not in self.agents:
                raise ValueError(f"Parent agent {agent.parent} not found")
            
            parent = self.agents[agent.parent]
            if parent.children is None:
                parent.children = []
            parent.children.append(agent.id)
        
        # Update hierarchy levels
        level = self._get_agent_level(agent.id)
        if level not in self.hierarchy_levels:
            self.hierarchy_levels[level] = []
        self.hierarchy_levels[level].append(agent.id)
    
    def _get_agent_level(self, agent_id: str) -> int:
        level = 0
        current = self.agents[agent_id]
        while current.parent:
            level += 1
            current = self.agents[current.parent]
        return level
    
    def get_subordinates(self, agent_id: str) -> List[str]:
        agent = self.agents[agent_id]
        if not agent.children:
            return []
        
        subordinates = agent.children.copy()
        for child_id in agent.children:
            subordinates.extend(self.get_subordinates(child_id))
        
        return subordinates
```

### Peer-to-Peer MAS
```python
class PeerToPeerMAS:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.connections: Dict[str, List[str]] = {}
    
    def add_agent(self, agent: Agent) -> None:
        self.agents[agent.id] = agent
        self.connections[agent.id] = []
    
    def connect_agents(self, agent1_id: str, agent2_id: str) -> None:
        if agent1_id not in self.agents or agent2_id not in self.agents:
            raise ValueError("One or both agents not found")
        
        if agent2_id not in self.connections[agent1_id]:
            self.connections[agent1_id].append(agent2_id)
        if agent1_id not in self.connections[agent2_id]:
            self.connections[agent2_id].append(agent1_id)
    
    def get_peers(self, agent_id: str) -> List[str]:
        return self.connections.get(agent_id, [])
    
    def is_connected(self, agent1_id: str, agent2_id: str) -> bool:
        return (
            agent2_id in self.connections.get(agent1_id, []) and
            agent1_id in self.connections.get(agent2_id, [])
        )
```

## Advanced MAS Types

### Market-Based MAS
```python
from enum import Enum
from typing import Dict, List, Optional, Tuple

class ResourceType(Enum):
    COMPUTATION = "computation"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"

@dataclass
class Resource:
    type: ResourceType
    amount: float
    unit: str

@dataclass
class Bid:
    agent_id: str
    resource_type: ResourceType
    amount: float
    price: float

class MarketBasedMAS:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.resources: Dict[str, Dict[ResourceType, Resource]] = {}
        self.bids: Dict[ResourceType, List[Bid]] = {
            rt: [] for rt in ResourceType
        }
    
    def add_agent(self, agent: Agent) -> None:
        self.agents[agent.id] = agent
        self.resources[agent.id] = {}
    
    def add_resource(self, agent_id: str, resource: Resource) -> None:
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        self.resources[agent_id][resource.type] = resource
    
    def place_bid(self, bid: Bid) -> None:
        if bid.agent_id not in self.agents:
            raise ValueError(f"Agent {bid.agent_id} not found")
        
        self.bids[bid.resource_type].append(bid)
        self.bids[bid.resource_type].sort(key=lambda x: x.price, reverse=True)
    
    def match_bids(self, resource_type: ResourceType) -> List[Tuple[Bid, str]]:
        matches = []
        available_bids = self.bids[resource_type].copy()
        
        for agent_id, resources in self.resources.items():
            if resource_type not in resources:
                continue
            
            resource = resources[resource_type]
            while available_bids and resource.amount > 0:
                bid = available_bids.pop(0)
                if bid.amount <= resource.amount:
                    matches.append((bid, agent_id))
                    resource.amount -= bid.amount
        
        return matches
```

### Swarm MAS
```python
import numpy as np
from typing import List, Tuple, Optional

@dataclass
class Position:
    x: float
    y: float
    z: float

class SwarmAgent(Agent):
    def __init__(self, id: str, position: Position):
        super().__init__(id=id, role="swarm", capabilities=["movement"])
        self.position = position
        self.velocity = Position(0, 0, 0)
        self.best_position = position
        self.best_score = float('inf')

class SwarmMAS:
    def __init__(self, num_agents: int, bounds: Tuple[float, float]):
        self.agents: List[SwarmAgent] = []
        self.bounds = bounds
        self.global_best_position = None
        self.global_best_score = float('inf')
        
        # Initialize agents
        for i in range(num_agents):
            position = Position(
                x=np.random.uniform(bounds[0], bounds[1]),
                y=np.random.uniform(bounds[0], bounds[1]),
                z=np.random.uniform(bounds[0], bounds[1])
            )
            self.agents.append(SwarmAgent(f"agent_{i}", position))
    
    def update_positions(self, fitness_function) -> None:
        for agent in self.agents:
            # Update velocity
            agent.velocity = Position(
                x=agent.velocity.x + np.random.random() * (agent.best_position.x - agent.position.x),
                y=agent.velocity.y + np.random.random() * (agent.best_position.y - agent.position.y),
                z=agent.velocity.z + np.random.random() * (agent.best_position.z - agent.position.z)
            )
            
            # Update position
            agent.position = Position(
                x=agent.position.x + agent.velocity.x,
                y=agent.position.y + agent.velocity.y,
                z=agent.position.z + agent.velocity.z
            )
            
            # Enforce bounds
            agent.position = Position(
                x=max(min(agent.position.x, self.bounds[1]), self.bounds[0]),
                y=max(min(agent.position.y, self.bounds[1]), self.bounds[0]),
                z=max(min(agent.position.z, self.bounds[1]), self.bounds[0])
            )
            
            # Update best positions
            score = fitness_function(agent.position)
            if score < agent.best_score:
                agent.best_score = score
                agent.best_position = agent.position
                
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = agent.position
```

## Best Practices

1. **Architecture Selection**:
   - Consider system requirements
   - Evaluate scalability needs
   - Assess communication patterns

2. **Agent Design**:
   - Define clear roles
   - Implement robust communication
   - Handle failures gracefully

3. **System Management**:
   - Monitor agent behavior
   - Track resource usage
   - Implement recovery mechanisms

4. **Performance Optimization**:
   - Optimize communication
   - Balance load distribution
   - Minimize resource contention

## Common Patterns

1. **MAS Factory**:
```python
class MASFactory:
    @staticmethod
    def create_mas(mas_type: str, **kwargs) -> Any:
        if mas_type == "hierarchical":
            return HierarchicalMAS()
        elif mas_type == "peer_to_peer":
            return PeerToPeerMAS()
        elif mas_type == "market_based":
            return MarketBasedMAS()
        elif mas_type == "swarm":
            return SwarmMAS(**kwargs)
        else:
            raise ValueError(f"Unknown MAS type: {mas_type}")
```

2. **MAS Monitor**:
```python
class MASMonitor:
    def __init__(self, mas: Any):
        self.mas = mas
        self.metrics = {
            "agent_count": [],
            "communication_count": [],
            "resource_usage": [],
            "performance_metrics": []
        }
    
    def record_metrics(self) -> None:
        # Record agent count
        self.metrics["agent_count"].append(len(self.mas.agents))
        
        # Record communication count (if applicable)
        if hasattr(self.mas, "connections"):
            total_connections = sum(len(conns) for conns in self.mas.connections.values())
            self.metrics["communication_count"].append(total_connections)
        
        # Record resource usage (if applicable)
        if hasattr(self.mas, "resources"):
            total_resources = sum(
                sum(r.amount for r in resources.values())
                for resources in self.mas.resources.values()
            )
            self.metrics["resource_usage"].append(total_resources)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for metric, values in self.metrics.items():
            if not values:
                continue
            
            summary[metric] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "current": values[-1]
            }
        
        return summary
```

## Further Reading

- [Multi-Agent Systems](https://arxiv.org/abs/2004.07213)
- [Hierarchical MAS](https://arxiv.org/abs/2004.07213)
- [Market-Based MAS](https://arxiv.org/abs/2004.07213)
- [Swarm Intelligence](https://arxiv.org/abs/2004.07213)
- [MAS Architectures](https://arxiv.org/abs/2004.07213) 