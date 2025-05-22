# API Types

This guide covers different types of APIs and their implementations, focusing on REST, GraphQL, WebSocket, and Streaming APIs.

## REST APIs

### Basic REST Client
```python
import requests
from typing import Dict, Any, Optional

class RESTClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/{endpoint}", params=params)
        response.raise_for_status()
        return response.json()
    
    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        response = self.session.post(f"{self.base_url}/{endpoint}", json=data)
        response.raise_for_status()
        return response.json()
```

### Async REST Client
```python
import aiohttp
from typing import Dict, Any, Optional

class AsyncRESTClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(f"{self.base_url}/{endpoint}", params=params) as response:
                response.raise_for_status()
                return await response.json()
    
    async def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.post(f"{self.base_url}/{endpoint}", json=data) as response:
                response.raise_for_status()
                return await response.json()
```

## GraphQL APIs

### GraphQL Client
```python
import requests
from typing import Dict, Any, Optional

class GraphQLClient:
    def __init__(self, endpoint: str, api_key: Optional[str] = None):
        self.endpoint = endpoint
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else None
        }
    
    def query(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "query": query,
            "variables": variables or {}
        }
        response = requests.post(self.endpoint, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()
```

### Example GraphQL Query
```python
# Example usage
client = GraphQLClient("https://api.example.com/graphql", "your-api-key")

query = """
query GetUser($id: ID!) {
    user(id: $id) {
        id
        name
        email
        posts {
            id
            title
        }
    }
}
"""

variables = {"id": "123"}
result = client.query(query, variables)
```

## WebSocket APIs

### WebSocket Client
```python
import asyncio
import websockets
import json
from typing import Dict, Any, Callable

class WebSocketClient:
    def __init__(self, uri: str):
        self.uri = uri
        self.websocket = None
    
    async def connect(self):
        self.websocket = await websockets.connect(self.uri)
    
    async def send(self, message: Dict[str, Any]):
        if not self.websocket:
            raise ConnectionError("Not connected to WebSocket server")
        await self.websocket.send(json.dumps(message))
    
    async def receive(self) -> Dict[str, Any]:
        if not self.websocket:
            raise ConnectionError("Not connected to WebSocket server")
        message = await self.websocket.recv()
        return json.loads(message)
    
    async def listen(self, callback: Callable[[Dict[str, Any]], None]):
        if not self.websocket:
            raise ConnectionError("Not connected to WebSocket server")
        while True:
            message = await self.receive()
            callback(message)
```

## Streaming APIs

### Streaming Client
```python
import requests
from typing import Dict, Any, Generator, Optional

class StreamingClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    def stream(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Generator[Dict[str, Any], None, None]:
        response = requests.get(
            f"{self.base_url}/{endpoint}",
            params=params,
            headers=self.headers,
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                yield json.loads(line)
```

### Async Streaming Client
```python
import aiohttp
from typing import Dict, Any, AsyncGenerator, Optional

class AsyncStreamingClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    async def stream(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.get(f"{self.base_url}/{endpoint}", params=params) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line:
                        yield json.loads(line)
```

## Best Practices

1. **Error Handling**:
   - Implement proper exception handling
   - Add retry mechanisms
   - Handle rate limits
   - Log errors appropriately

2. **Performance**:
   - Use connection pooling
   - Implement caching
   - Handle timeouts
   - Use compression

3. **Security**:
   - Validate input data
   - Use HTTPS
   - Implement proper authentication
   - Handle sensitive data

## Common Patterns

1. **Retry Mechanism**:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def make_api_request():
    # API request implementation
    pass
```

2. **Circuit Breaker**:
```python
from pybreaker import CircuitBreaker

breaker = CircuitBreaker(fail_max=5, reset_timeout=60)

@breaker
def make_api_request():
    # API request implementation
    pass
```

3. **Request Batching**:
```python
class BatchClient:
    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
        self.batch = []
    
    def add_request(self, request: Dict[str, Any]):
        self.batch.append(request)
        if len(self.batch) >= self.batch_size:
            return self.process_batch()
        return None
    
    def process_batch(self) -> List[Dict[str, Any]]:
        # Process batch of requests
        results = process_requests(self.batch)
        self.batch = []
        return results
```

## Further Reading

- [REST API Best Practices](https://restfulapi.net/)
- [GraphQL Documentation](https://graphql.org/learn/)
- [WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)
- [HTTP/2 Server Push](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [aiohttp Documentation](https://docs.aiohttp.org/)
- [requests Documentation](https://docs.python-requests.org/) 