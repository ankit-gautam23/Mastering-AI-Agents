# Network Programming in Python and TypeScript

This guide covers network programming concepts and implementations in both Python and TypeScript, including HTTP clients, servers, and WebSocket communication.

## HTTP Clients

### Python HTTP Client
```python
import requests

# GET request
response = requests.get('https://api.example.com/data')
print(response.status_code)  # 200
print(response.json())  # {'key': 'value'}

# POST request
data = {'name': 'John', 'age': 30}
response = requests.post('https://api.example.com/users', json=data)
print(response.status_code)  # 201

# With headers
headers = {'Authorization': 'Bearer token123'}
response = requests.get('https://api.example.com/protected', headers=headers)

# Error handling
try:
    response = requests.get('https://api.example.com/data')
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
```

### TypeScript HTTP Client
```typescript
import axios from 'axios';

// GET request
async function getData() {
    try {
        const response = await axios.get('https://api.example.com/data');
        console.log(response.status);  // 200
        console.log(response.data);  // { key: 'value' }
    } catch (error) {
        console.error('Error:', error);
    }
}

// POST request
async function createUser() {
    const data = { name: 'John', age: 30 };
    try {
        const response = await axios.post('https://api.example.com/users', data);
        console.log(response.status);  // 201
    } catch (error) {
        console.error('Error:', error);
    }
}

// With headers
const headers = { Authorization: 'Bearer token123' };
axios.get('https://api.example.com/protected', { headers });
```

## HTTP Servers

### Python HTTP Server
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({'message': 'Hello, World!'})

@app.route('/api/users', methods=['POST'])
def create_user():
    data = request.get_json()
    # Process user data
    return jsonify({'id': 1, 'name': data['name']}), 201

if __name__ == '__main__':
    app.run(debug=True, port=3000)
```

### TypeScript HTTP Server
```typescript
import express from 'express';
import { Request, Response } from 'express';

const app = express();
app.use(express.json());

app.get('/api/data', (req: Request, res: Response) => {
    res.json({ message: 'Hello, World!' });
});

app.post('/api/users', (req: Request, res: Response) => {
    const userData = req.body;
    // Process user data
    res.status(201).json({ id: 1, name: userData.name });
});

app.listen(3000, () => {
    console.log('Server running on port 3000');
});
```

## WebSocket Communication

### Python WebSocket
```python
import asyncio
import websockets

async def websocket_server(websocket, path):
    try:
        async for message in websocket:
            print(f"Received: {message}")
            await websocket.send(f"Echo: {message}")
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    server = await websockets.serve(websocket_server, "localhost", 8765)
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())

# WebSocket Client
async def websocket_client():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        await websocket.send("Hello, Server!")
        response = await websocket.recv()
        print(f"Received: {response}")
```

### TypeScript WebSocket
```typescript
import WebSocket from 'ws';

// WebSocket Server
const wss = new WebSocket.Server({ port: 8765 });

wss.on('connection', (ws) => {
    console.log('Client connected');

    ws.on('message', (message) => {
        console.log('Received:', message.toString());
        ws.send(`Echo: ${message}`);
    });

    ws.on('close', () => {
        console.log('Client disconnected');
    });
});

// WebSocket Client
const ws = new WebSocket('ws://localhost:8765');

ws.on('open', () => {
    console.log('Connected to server');
    ws.send('Hello, Server!');
});

ws.on('message', (data) => {
    console.log('Received:', data.toString());
});

ws.on('close', () => {
    console.log('Disconnected from server');
});
```

## RESTful API Best Practices

1. **Resource Naming**:
   - Use nouns for resources
   - Use plural forms
   - Use lowercase letters
   - Use hyphens for multi-word resources

2. **HTTP Methods**:
   - GET: Retrieve resources
   - POST: Create resources
   - PUT: Update resources
   - DELETE: Remove resources
   - PATCH: Partial updates

3. **Status Codes**:
   - 200: Success
   - 201: Created
   - 400: Bad Request
   - 401: Unauthorized
   - 403: Forbidden
   - 404: Not Found
   - 500: Server Error

## Error Handling

### Python Error Handling
```python
from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

@app.errorhandler(Exception)
def handle_error(error):
    if isinstance(error, HTTPException):
        response = {
            "error": error.description,
            "status_code": error.code
        }
        return jsonify(response), error.code
    
    response = {
        "error": "Internal Server Error",
        "status_code": 500
    }
    return jsonify(response), 500
```

### TypeScript Error Handling
```typescript
import express, { Request, Response, NextFunction } from 'express';

const app = express();

app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
    console.error(err.stack);
    res.status(500).json({
        error: 'Internal Server Error',
        status_code: 500
    });
});

// Custom error class
class AppError extends Error {
    constructor(public statusCode: number, message: string) {
        super(message);
        this.name = 'AppError';
    }
}
```

## Security Best Practices

1. **Authentication**:
   - Use JWT tokens
   - Implement OAuth 2.0
   - Use secure password hashing
   - Implement rate limiting

2. **Data Protection**:
   - Use HTTPS
   - Implement CORS
   - Sanitize input data
   - Validate request data

3. **Error Handling**:
   - Don't expose sensitive information
   - Log errors appropriately
   - Use custom error messages
   - Implement proper error status codes

## Further Reading

- [Python Requests Library](https://docs.python-requests.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Express.js Documentation](https://expressjs.com/)
- [WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)
- [REST API Best Practices](https://restfulapi.net/)
- [HTTP Status Codes](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status) 