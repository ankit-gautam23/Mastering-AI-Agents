# Agent-to-Agent (A2A) Protocol

This guide covers the fundamental concepts and implementations of Agent-to-Agent (A2A) Protocol in Multi-Agent Systems, including message formats, protocols, and interaction patterns.

## Basic A2A Protocol

### Message Format
```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    QUERY = "query"
    COMMAND = "command"
    ACKNOWLEDGMENT = "acknowledgment"

class MessagePriority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class A2AMessage:
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    priority: MessagePriority
    content: Any
    timestamp: datetime
    conversation_id: str
    metadata: Dict[str, Any]
    
    def to_json(self) -> str:
        return json.dumps({
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "conversation_id": self.conversation_id,
            "metadata": self.metadata
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'A2AMessage':
        data = json.loads(json_str)
        return cls(
            message_id=data["message_id"],
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            message_type=MessageType(data["message_type"]),
            priority=MessagePriority(data["priority"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            conversation_id=data["conversation_id"],
            metadata=data["metadata"]
        )
```

### Protocol Implementation
```python
class A2AProtocol:
    def __init__(self):
        self.message_handlers: Dict[MessageType, List[callable]] = {
            mt: [] for mt in MessageType
        }
        self.conversations: Dict[str, List[A2AMessage]] = {}
    
    def register_handler(self, message_type: MessageType, handler: callable) -> None:
        self.message_handlers[message_type].append(handler)
    
    def send_message(self, message: A2AMessage) -> None:
        # Store message in conversation history
        if message.conversation_id not in self.conversations:
            self.conversations[message.conversation_id] = []
        self.conversations[message.conversation_id].append(message)
        
        # Process message with registered handlers
        for handler in self.message_handlers[message.message_type]:
            try:
                handler(message)
            except Exception as e:
                # Log error and continue with other handlers
                print(f"Error in message handler: {str(e)}")
    
    def get_conversation(self, conversation_id: str) -> List[A2AMessage]:
        return self.conversations.get(conversation_id, [])
    
    def create_response(
        self,
        original_message: A2AMessage,
        content: Any,
        message_type: MessageType = MessageType.RESPONSE
    ) -> A2AMessage:
        return A2AMessage(
            message_id=f"resp_{original_message.message_id}",
            sender_id=original_message.receiver_id,
            receiver_id=original_message.sender_id,
            message_type=message_type,
            priority=original_message.priority,
            content=content,
            timestamp=datetime.now(),
            conversation_id=original_message.conversation_id,
            metadata={}
        )
```

## Advanced A2A Features

### Message Routing
```python
class MessageRouter:
    def __init__(self):
        self.routes: Dict[str, List[str]] = {}
        self.message_queue: List[A2AMessage] = []
        self.processed_messages: Dict[str, A2AMessage] = {}
    
    def add_route(self, source: str, destination: str) -> None:
        if source not in self.routes:
            self.routes[source] = []
        if destination not in self.routes[source]:
            self.routes[source].append(destination)
    
    def route_message(self, message: A2AMessage) -> List[A2AMessage]:
        if message.message_id in self.processed_messages:
            return []
        
        self.processed_messages[message.message_id] = message
        routed_messages = []
        
        # Get all possible routes
        destinations = self.routes.get(message.sender_id, [])
        for destination in destinations:
            routed_message = A2AMessage(
                message_id=f"route_{message.message_id}",
                sender_id=message.sender_id,
                receiver_id=destination,
                message_type=message.message_type,
                priority=message.priority,
                content=message.content,
                timestamp=datetime.now(),
                conversation_id=message.conversation_id,
                metadata=message.metadata
            )
            routed_messages.append(routed_message)
        
        return routed_messages
    
    def process_queue(self) -> None:
        while self.message_queue:
            message = self.message_queue.pop(0)
            routed_messages = self.route_message(message)
            self.message_queue.extend(routed_messages)
```

### Message Validation
```python
class MessageValidator:
    def __init__(self):
        self.validation_rules: Dict[MessageType, List[callable]] = {
            mt: [] for mt in MessageType
        }
    
    def add_validation_rule(self, message_type: MessageType, rule: callable) -> None:
        self.validation_rules[message_type].append(rule)
    
    def validate_message(self, message: A2AMessage) -> Dict[str, Any]:
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Run all validation rules for the message type
        for rule in self.validation_rules[message.message_type]:
            try:
                result = rule(message)
                if not result["valid"]:
                    validation_result["valid"] = False
                    validation_result["errors"].extend(result["errors"])
                if result.get("warnings"):
                    validation_result["warnings"].extend(result["warnings"])
            except Exception as e:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Validation rule failed: {str(e)}")
        
        return validation_result
```

## Best Practices

1. **Message Design**:
   - Use clear message types
   - Include necessary metadata
   - Implement proper validation

2. **Protocol Implementation**:
   - Follow standard protocols
   - Handle message routing
   - Implement error handling

3. **Message Processing**:
   - Validate messages
   - Handle priorities
   - Maintain conversation history

4. **Performance Optimization**:
   - Implement message queuing
   - Optimize routing
   - Handle message batching

## Common Patterns

1. **A2A Pipeline**:
```python
class A2APipeline:
    def __init__(self):
        self.protocol = A2AProtocol()
        self.router = MessageRouter()
        self.validator = MessageValidator()
    
    def process_message(self, message: A2AMessage) -> Dict[str, Any]:
        # Validate message
        validation = self.validator.validate_message(message)
        if not validation["valid"]:
            return {
                "status": "failed",
                "reason": "validation_failed",
                "errors": validation["errors"]
            }
        
        # Route message
        routed_messages = self.router.route_message(message)
        
        # Process messages
        results = []
        for routed_message in routed_messages:
            self.protocol.send_message(routed_message)
            results.append({
                "message_id": routed_message.message_id,
                "receiver_id": routed_message.receiver_id,
                "status": "sent"
            })
        
        return {
            "status": "success",
            "validation": validation,
            "routed_messages": results
        }
```

2. **A2A Monitor**:
```python
class A2AMonitor:
    def __init__(self, pipeline: A2APipeline):
        self.pipeline = pipeline
        self.metrics = {
            "message_count": [],
            "validation_errors": [],
            "routing_count": [],
            "processing_time": []
        }
    
    def record_metrics(self, result: Dict[str, Any]) -> None:
        # Record message count
        self.metrics["message_count"].append(1)
        
        # Record validation errors
        if "validation" in result:
            self.metrics["validation_errors"].append(
                len(result["validation"]["errors"])
            )
        
        # Record routing count
        if "routed_messages" in result:
            self.metrics["routing_count"].append(
                len(result["routed_messages"])
            )
    
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

- [A2A Protocol](https://arxiv.org/abs/2004.07213)
- [Message Routing](https://arxiv.org/abs/2004.07213)
- [Message Validation](https://arxiv.org/abs/2004.07213)
- [Protocol Design](https://arxiv.org/abs/2004.07213)
- [A2A Patterns](https://arxiv.org/abs/2004.07213) 