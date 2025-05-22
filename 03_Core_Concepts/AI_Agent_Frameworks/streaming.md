# Streaming

This guide covers the fundamental concepts and implementations of streaming in AI agent frameworks, including data streaming, event streaming, message streaming, and stream processing.

## Data Streaming

### Basic Data Stream
```python
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
import asyncio
import json

@dataclass
class DataChunk:
    data: Any
    timestamp: float
    metadata: Dict[str, Any] = None

class DataStream:
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.buffer = asyncio.Queue(maxsize=buffer_size)
        self.subscribers = []
    
    async def put(self, chunk: DataChunk) -> None:
        """Put a data chunk into the stream"""
        await self.buffer.put(chunk)
        
        # Notify subscribers
        for subscriber in self.subscribers:
            await subscriber(chunk)
    
    async def get(self) -> DataChunk:
        """Get a data chunk from the stream"""
        return await self.buffer.get()
    
    def subscribe(self, subscriber: callable) -> None:
        """Subscribe to the stream"""
        self.subscribers.append(subscriber)
    
    def unsubscribe(self, subscriber: callable) -> None:
        """Unsubscribe from the stream"""
        if subscriber in self.subscribers:
            self.subscribers.remove(subscriber)
    
    async def process(self, processor: callable) -> None:
        """Process the stream"""
        while True:
            chunk = await self.get()
            result = await processor(chunk)
            if result:
                await self.put(result)
```

### Advanced Data Stream
```python
class AdvancedDataStream:
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.buffer = asyncio.Queue(maxsize=buffer_size)
        self.subscribers = []
        self.filters = []
        self.transformers = []
    
    def add_filter(self, filter_func: callable) -> None:
        """Add a filter to the stream"""
        self.filters.append(filter_func)
    
    def add_transformer(self, transformer: callable) -> None:
        """Add a transformer to the stream"""
        self.transformers.append(transformer)
    
    async def put(self, chunk: DataChunk) -> None:
        """Put a data chunk into the stream with filtering and transformation"""
        # Apply filters
        for filter_func in self.filters:
            if not await filter_func(chunk):
                return
        
        # Apply transformers
        for transformer in self.transformers:
            chunk = await transformer(chunk)
        
        await self.buffer.put(chunk)
        
        # Notify subscribers
        for subscriber in self.subscribers:
            await subscriber(chunk)
    
    async def get(self, timeout: float = None) -> Optional[DataChunk]:
        """Get a data chunk from the stream with timeout"""
        try:
            return await asyncio.wait_for(self.buffer.get(), timeout)
        except asyncio.TimeoutError:
            return None
    
    def subscribe(self, subscriber: callable) -> None:
        """Subscribe to the stream"""
        self.subscribers.append(subscriber)
    
    def unsubscribe(self, subscriber: callable) -> None:
        """Unsubscribe from the stream"""
        if subscriber in self.subscribers:
            self.subscribers.remove(subscriber)
    
    async def process(self, processor: callable) -> None:
        """Process the stream with error handling"""
        while True:
            try:
                chunk = await self.get()
                if chunk is None:
                    continue
                
                result = await processor(chunk)
                if result:
                    await self.put(result)
            except Exception as e:
                # Handle processing error
                print(f"Error processing chunk: {e}")
```

## Event Streaming

### Basic Event Stream
```python
@dataclass
class Event:
    type: str
    data: Any
    timestamp: float
    source: str = None

class EventStream:
    def __init__(self):
        self.events = asyncio.Queue()
        self.handlers = {}
    
    async def emit(self, event: Event) -> None:
        """Emit an event to the stream"""
        await self.events.put(event)
        
        # Handle event
        if event.type in self.handlers:
            for handler in self.handlers[event.type]:
                await handler(event)
    
    def on(self, event_type: str, handler: callable) -> None:
        """Register an event handler"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    def off(self, event_type: str, handler: callable) -> None:
        """Unregister an event handler"""
        if event_type in self.handlers and handler in self.handlers[event_type]:
            self.handlers[event_type].remove(handler)
    
    async def process(self) -> None:
        """Process events in the stream"""
        while True:
            event = await self.events.get()
            if event.type in self.handlers:
                for handler in self.handlers[event.type]:
                    await handler(event)
```

### Advanced Event Stream
```python
class AdvancedEventStream:
    def __init__(self):
        self.events = asyncio.Queue()
        self.handlers = {}
        self.middleware = []
        self.error_handlers = {}
    
    def use(self, middleware: callable) -> None:
        """Add middleware to the stream"""
        self.middleware.append(middleware)
    
    def on_error(self, event_type: str, handler: callable) -> None:
        """Register an error handler"""
        self.error_handlers[event_type] = handler
    
    async def emit(self, event: Event) -> None:
        """Emit an event to the stream with middleware"""
        # Apply middleware
        for middleware in self.middleware:
            event = await middleware(event)
        
        await self.events.put(event)
        
        # Handle event
        if event.type in self.handlers:
            for handler in self.handlers[event.type]:
                try:
                    await handler(event)
                except Exception as e:
                    if event.type in self.error_handlers:
                        await self.error_handlers[event.type](e, event)
    
    def on(self, event_type: str, handler: callable) -> None:
        """Register an event handler"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    def off(self, event_type: str, handler: callable) -> None:
        """Unregister an event handler"""
        if event_type in self.handlers and handler in self.handlers[event_type]:
            self.handlers[event_type].remove(handler)
    
    async def process(self) -> None:
        """Process events in the stream with error handling"""
        while True:
            try:
                event = await self.events.get()
                if event.type in self.handlers:
                    for handler in self.handlers[event.type]:
                        try:
                            await handler(event)
                        except Exception as e:
                            if event.type in self.error_handlers:
                                await self.error_handlers[event.type](e, event)
            except Exception as e:
                print(f"Error processing event: {e}")
```

## Message Streaming

### Basic Message Stream
```python
@dataclass
class Message:
    content: Any
    timestamp: float
    sender: str
    receiver: str = None

class MessageStream:
    def __init__(self):
        self.messages = asyncio.Queue()
        self.receivers = {}
    
    async def send(self, message: Message) -> None:
        """Send a message to the stream"""
        await self.messages.put(message)
        
        # Deliver to receiver
        if message.receiver in self.receivers:
            for handler in self.receivers[message.receiver]:
                await handler(message)
    
    def register(self, receiver: str, handler: callable) -> None:
        """Register a message handler"""
        if receiver not in self.receivers:
            self.receivers[receiver] = []
        self.receivers[receiver].append(handler)
    
    def unregister(self, receiver: str, handler: callable) -> None:
        """Unregister a message handler"""
        if receiver in self.receivers and handler in self.receivers[receiver]:
            self.receivers[receiver].remove(handler)
    
    async def process(self) -> None:
        """Process messages in the stream"""
        while True:
            message = await self.messages.get()
            if message.receiver in self.receivers:
                for handler in self.receivers[message.receiver]:
                    await handler(message)
```

### Advanced Message Stream
```python
class AdvancedMessageStream:
    def __init__(self):
        self.messages = asyncio.Queue()
        self.receivers = {}
        self.middleware = []
        self.error_handlers = {}
    
    def use(self, middleware: callable) -> None:
        """Add middleware to the stream"""
        self.middleware.append(middleware)
    
    def on_error(self, receiver: str, handler: callable) -> None:
        """Register an error handler"""
        self.error_handlers[receiver] = handler
    
    async def send(self, message: Message) -> None:
        """Send a message to the stream with middleware"""
        # Apply middleware
        for middleware in self.middleware:
            message = await middleware(message)
        
        await self.messages.put(message)
        
        # Deliver to receiver
        if message.receiver in self.receivers:
            for handler in self.receivers[message.receiver]:
                try:
                    await handler(message)
                except Exception as e:
                    if message.receiver in self.error_handlers:
                        await self.error_handlers[message.receiver](e, message)
    
    def register(self, receiver: str, handler: callable) -> None:
        """Register a message handler"""
        if receiver not in self.receivers:
            self.receivers[receiver] = []
        self.receivers[receiver].append(handler)
    
    def unregister(self, receiver: str, handler: callable) -> None:
        """Unregister a message handler"""
        if receiver in self.receivers and handler in self.receivers[receiver]:
            self.receivers[receiver].remove(handler)
    
    async def process(self) -> None:
        """Process messages in the stream with error handling"""
        while True:
            try:
                message = await self.messages.get()
                if message.receiver in self.receivers:
                    for handler in self.receivers[message.receiver]:
                        try:
                            await handler(message)
                        except Exception as e:
                            if message.receiver in self.error_handlers:
                                await self.error_handlers[message.receiver](e, message)
            except Exception as e:
                print(f"Error processing message: {e}")
```

## Stream Processing

### Basic Stream Processor
```python
class StreamProcessor:
    def __init__(self):
        self.processors = {}
        self.pipelines = {}
    
    def add_processor(self, name: str, processor: callable) -> None:
        """Add a processor to the pipeline"""
        self.processors[name] = processor
    
    def create_pipeline(self, name: str, processors: List[str]) -> None:
        """Create a processing pipeline"""
        self.pipelines[name] = processors
    
    async def process(self, pipeline_name: str, data: Any) -> Any:
        """Process data through a pipeline"""
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_name} not found")
        
        result = data
        for processor_name in self.pipelines[pipeline_name]:
            if processor_name not in self.processors:
                raise ValueError(f"Processor {processor_name} not found")
            
            processor = self.processors[processor_name]
            result = await processor(result)
        
        return result
```

### Advanced Stream Processor
```python
class AdvancedStreamProcessor:
    def __init__(self):
        self.processors = {}
        self.pipelines = {}
        self.middleware = []
        self.error_handlers = {}
    
    def use(self, middleware: callable) -> None:
        """Add middleware to the processor"""
        self.middleware.append(middleware)
    
    def on_error(self, pipeline_name: str, handler: callable) -> None:
        """Register an error handler"""
        self.error_handlers[pipeline_name] = handler
    
    def add_processor(self, name: str, processor: callable) -> None:
        """Add a processor to the pipeline"""
        self.processors[name] = processor
    
    def create_pipeline(self, name: str, processors: List[str]) -> None:
        """Create a processing pipeline"""
        self.pipelines[name] = processors
    
    async def process(self, pipeline_name: str, data: Any) -> Any:
        """Process data through a pipeline with middleware and error handling"""
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_name} not found")
        
        # Apply middleware
        for middleware in self.middleware:
            data = await middleware(data)
        
        result = data
        for processor_name in self.pipelines[pipeline_name]:
            if processor_name not in self.processors:
                raise ValueError(f"Processor {processor_name} not found")
            
            try:
                processor = self.processors[processor_name]
                result = await processor(result)
            except Exception as e:
                if pipeline_name in self.error_handlers:
                    await self.error_handlers[pipeline_name](e, result)
                raise
        
        return result
```

## Best Practices

1. **Data Streaming**:
   - Buffer management
   - Subscriber notification
   - Data transformation
   - Error handling

2. **Event Streaming**:
   - Event handling
   - Middleware support
   - Error handling
   - Event filtering

3. **Message Streaming**:
   - Message delivery
   - Receiver registration
   - Middleware support
   - Error handling

4. **Stream Processing**:
   - Pipeline creation
   - Processor management
   - Middleware support
   - Error handling

## Common Patterns

1. **Stream Factory**:
```python
class StreamFactory:
    @staticmethod
    def create_stream(stream_type: str, **kwargs) -> Any:
        if stream_type == 'data':
            return DataStream(**kwargs)
        elif stream_type == 'event':
            return EventStream(**kwargs)
        elif stream_type == 'message':
            return MessageStream(**kwargs)
        else:
            raise ValueError(f"Unknown stream type: {stream_type}")
```

2. **Stream Monitor**:
```python
class StreamMonitor:
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

3. **Stream Validator**:
```python
class StreamValidator:
    def __init__(self):
        self.validators = {}
    
    def add_validator(self, name: str, validator: callable) -> None:
        """Add a stream validator"""
        self.validators[name] = validator
    
    def validate(self, name: str, data: Any) -> bool:
        """Validate stream data"""
        if name not in self.validators:
            return True
        
        validator = self.validators[name]
        return validator(data)
```

## Further Reading

- [Data Streaming](https://arxiv.org/abs/2004.07213)
- [Event Streaming](https://arxiv.org/abs/2004.07213)
- [Message Streaming](https://arxiv.org/abs/2004.07213)
- [Stream Processing](https://arxiv.org/abs/2004.07213)
- [Reactive Programming](https://arxiv.org/abs/2004.07213) 