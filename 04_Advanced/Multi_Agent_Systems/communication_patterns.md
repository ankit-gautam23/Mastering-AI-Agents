# Communication Patterns in Multi-Agent Systems

This guide covers the fundamental concepts and implementations of communication patterns in Multi-Agent Systems (MAS), including message passing, protocols, and coordination mechanisms.

## Basic Communication Patterns

### Message Passing
```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    INFORM = "inform"
    QUERY = "query"
    PROPOSE = "propose"
    ACCEPT = "accept"
    REJECT = "reject"

@dataclass
class Message:
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: Any
    timestamp: datetime
    conversation_id: str
    metadata: Dict[str, Any]

class MessageBroker:
    def __init__(self):
        self.messages: Dict[str, List[Message]] = {}
        self.conversations: Dict[str, List[Message]] = {}
    
    def send_message(self, message: Message) -> None:
        # Store message in receiver's inbox
        if message.receiver_id not in self.messages:
            self.messages[message.receiver_id] = []
        self.messages[message.receiver_id].append(message)
        
        # Store message in conversation history
        if message.conversation_id not in self.conversations:
            self.conversations[message.conversation_id] = []
        self.conversations[message.conversation_id].append(message)
    
    def get_messages(self, agent_id: str) -> List[Message]:
        return self.messages.get(agent_id, [])
    
    def get_conversation(self, conversation_id: str) -> List[Message]:
        return self.conversations.get(conversation_id, [])
    
    def clear_messages(self, agent_id: str) -> None:
        if agent_id in self.messages:
            del self.messages[agent_id]
```

### Protocol Implementation
```python
class CommunicationProtocol:
    def __init__(self, broker: MessageBroker):
        self.broker = broker
        self.protocols = {
            "request_response": self._handle_request_response,
            "publish_subscribe": self._handle_publish_subscribe,
            "contract_net": self._handle_contract_net
        }
    
    def _handle_request_response(self, message: Message) -> Optional[Message]:
        if message.message_type == MessageType.REQUEST:
            # Process request and generate response
            response = Message(
                sender_id=message.receiver_id,
                receiver_id=message.sender_id,
                message_type=MessageType.RESPONSE,
                content={"status": "processed", "result": "response_data"},
                timestamp=datetime.now(),
                conversation_id=message.conversation_id,
                metadata={}
            )
            self.broker.send_message(response)
            return response
        return None
    
    def _handle_publish_subscribe(self, message: Message) -> None:
        if message.message_type == MessageType.INFORM:
            # Broadcast message to all subscribers
            for agent_id in self.broker.messages.keys():
                if agent_id != message.sender_id:
                    broadcast = Message(
                        sender_id=message.sender_id,
                        receiver_id=agent_id,
                        message_type=MessageType.INFORM,
                        content=message.content,
                        timestamp=datetime.now(),
                        conversation_id=message.conversation_id,
                        metadata=message.metadata
                    )
                    self.broker.send_message(broadcast)
    
    def _handle_contract_net(self, message: Message) -> Optional[Message]:
        if message.message_type == MessageType.PROPOSE:
            # Handle contract proposal
            response = Message(
                sender_id=message.receiver_id,
                receiver_id=message.sender_id,
                message_type=MessageType.ACCEPT,
                content={"status": "accepted"},
                timestamp=datetime.now(),
                conversation_id=message.conversation_id,
                metadata={}
            )
            self.broker.send_message(response)
            return response
        return None
```

## Advanced Communication Patterns

### Coordination Mechanisms
```python
class CoordinationMechanism:
    def __init__(self, broker: MessageBroker):
        self.broker = broker
        self.coordination_states: Dict[str, Dict[str, Any]] = {}
    
    def start_coordination(self, coordination_id: str, agents: List[str]) -> None:
        self.coordination_states[coordination_id] = {
            "agents": agents,
            "state": "initializing",
            "results": {},
            "consensus": False
        }
    
    def update_coordination(self, coordination_id: str, agent_id: str, result: Any) -> None:
        if coordination_id not in self.coordination_states:
            raise ValueError(f"Coordination {coordination_id} not found")
        
        state = self.coordination_states[coordination_id]
        state["results"][agent_id] = result
        
        # Check if all agents have reported
        if len(state["results"]) == len(state["agents"]):
            state["state"] = "completed"
            state["consensus"] = self._check_consensus(state["results"])
    
    def _check_consensus(self, results: Dict[str, Any]) -> bool:
        if not results:
            return False
        
        # Check if all results are the same
        first_result = next(iter(results.values()))
        return all(result == first_result for result in results.values())
    
    def get_coordination_status(self, coordination_id: str) -> Dict[str, Any]:
        return self.coordination_states.get(coordination_id, {})
```

### Negotiation Protocol
```python
class NegotiationProtocol:
    def __init__(self, broker: MessageBroker):
        self.broker = broker
        self.negotiations: Dict[str, Dict[str, Any]] = {}
    
    def start_negotiation(
        self,
        negotiation_id: str,
        initiator_id: str,
        responder_id: str,
        proposal: Any
    ) -> None:
        self.negotiations[negotiation_id] = {
            "initiator": initiator_id,
            "responder": responder_id,
            "proposal": proposal,
            "counter_proposals": [],
            "status": "active",
            "round": 1
        }
        
        # Send initial proposal
        message = Message(
            sender_id=initiator_id,
            receiver_id=responder_id,
            message_type=MessageType.PROPOSE,
            content=proposal,
            timestamp=datetime.now(),
            conversation_id=negotiation_id,
            metadata={"round": 1}
        )
        self.broker.send_message(message)
    
    def handle_counter_proposal(
        self,
        negotiation_id: str,
        proposal: Any,
        is_accept: bool
    ) -> None:
        if negotiation_id not in self.negotiations:
            raise ValueError(f"Negotiation {negotiation_id} not found")
        
        negotiation = self.negotiations[negotiation_id]
        
        if is_accept:
            negotiation["status"] = "accepted"
            negotiation["final_proposal"] = proposal
        else:
            negotiation["counter_proposals"].append(proposal)
            negotiation["round"] += 1
            
            # Send counter proposal
            message = Message(
                sender_id=negotiation["responder"],
                receiver_id=negotiation["initiator"],
                message_type=MessageType.PROPOSE,
                content=proposal,
                timestamp=datetime.now(),
                conversation_id=negotiation_id,
                metadata={"round": negotiation["round"]}
            )
            self.broker.send_message(message)
    
    def get_negotiation_status(self, negotiation_id: str) -> Dict[str, Any]:
        return self.negotiations.get(negotiation_id, {})
```

## Best Practices

1. **Message Design**:
   - Use clear message types
   - Include necessary metadata
   - Implement proper error handling

2. **Protocol Implementation**:
   - Follow standard protocols
   - Handle timeouts
   - Implement retry mechanisms

3. **Coordination Strategy**:
   - Define clear roles
   - Implement consensus mechanisms
   - Handle failures gracefully

4. **Performance Optimization**:
   - Minimize message overhead
   - Implement message batching
   - Use efficient routing

## Common Patterns

1. **Communication Pipeline**:
```python
class CommunicationPipeline:
    def __init__(self):
        self.broker = MessageBroker()
        self.protocol = CommunicationProtocol(self.broker)
        self.coordination = CoordinationMechanism(self.broker)
        self.negotiation = NegotiationProtocol(self.broker)
    
    def process_message(self, message: Message) -> None:
        # Handle message based on type
        if message.message_type in [MessageType.REQUEST, MessageType.RESPONSE]:
            self.protocol._handle_request_response(message)
        elif message.message_type == MessageType.INFORM:
            self.protocol._handle_publish_subscribe(message)
        elif message.message_type in [MessageType.PROPOSE, MessageType.ACCEPT]:
            self.protocol._handle_contract_net(message)
    
    def get_communication_status(self) -> Dict[str, Any]:
        return {
            "active_conversations": len(self.broker.conversations),
            "pending_messages": sum(len(messages) for messages in self.broker.messages.values()),
            "active_coordinations": len(self.coordination.coordination_states),
            "active_negotiations": len(self.negotiation.negotiations)
        }
```

2. **Communication Monitor**:
```python
class CommunicationMonitor:
    def __init__(self, pipeline: CommunicationPipeline):
        self.pipeline = pipeline
        self.metrics = {
            "message_count": [],
            "response_time": [],
            "protocol_usage": [],
            "error_rate": []
        }
    
    def record_metrics(self) -> None:
        # Record message count
        total_messages = sum(len(messages) for messages in self.pipeline.broker.messages.values())
        self.metrics["message_count"].append(total_messages)
        
        # Record protocol usage
        protocol_usage = {
            "request_response": 0,
            "publish_subscribe": 0,
            "contract_net": 0
        }
        
        for conversation in self.pipeline.broker.conversations.values():
            for message in conversation:
                if message.message_type in [MessageType.REQUEST, MessageType.RESPONSE]:
                    protocol_usage["request_response"] += 1
                elif message.message_type == MessageType.INFORM:
                    protocol_usage["publish_subscribe"] += 1
                elif message.message_type in [MessageType.PROPOSE, MessageType.ACCEPT]:
                    protocol_usage["contract_net"] += 1
        
        self.metrics["protocol_usage"].append(protocol_usage)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for metric, values in self.metrics.items():
            if not values:
                continue
            
            if isinstance(values[0], dict):
                # Handle protocol usage metrics
                summary[metric] = {
                    protocol: sum(v[protocol] for v in values) / len(values)
                    for protocol in values[0].keys()
                }
            else:
                summary[metric] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "current": values[-1]
                }
        
        return summary
```

## Further Reading

- [MAS Communication](https://arxiv.org/abs/2004.07213)
- [Message Passing](https://arxiv.org/abs/2004.07213)
- [Coordination Protocols](https://arxiv.org/abs/2004.07213)
- [Negotiation Strategies](https://arxiv.org/abs/2004.07213)
- [Communication Patterns](https://arxiv.org/abs/2004.07213) 