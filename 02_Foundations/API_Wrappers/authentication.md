# Authentication

This guide covers various authentication methods and best practices for securing API access.

## API Keys

### Basic API Key Authentication
```python
from typing import Optional
import os
from dotenv import load_dotenv

class APIKeyAuth:
    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        self.api_key = api_key or os.getenv('API_KEY')
        if not self.api_key:
            raise ValueError("API key not provided")
    
    def get_headers(self) -> dict:
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def validate_key(self) -> bool:
        # Implement key validation logic
        return bool(self.api_key and len(self.api_key) > 0)
```

### API Key Management
```python
from typing import Dict, Optional
import json
import os
from datetime import datetime

class APIKeyManager:
    def __init__(self, keys_file: str = 'api_keys.json'):
        self.keys_file = keys_file
        self.keys: Dict[str, Dict] = self._load_keys()
    
    def _load_keys(self) -> Dict[str, Dict]:
        if os.path.exists(self.keys_file):
            with open(self.keys_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_keys(self):
        with open(self.keys_file, 'w') as f:
            json.dump(self.keys, f, indent=4)
    
    def add_key(self, key: str, name: str, expires_at: Optional[datetime] = None):
        self.keys[key] = {
            'name': name,
            'created_at': datetime.now().isoformat(),
            'expires_at': expires_at.isoformat() if expires_at else None
        }
        self._save_keys()
    
    def remove_key(self, key: str):
        if key in self.keys:
            del self.keys[key]
            self._save_keys()
    
    def validate_key(self, key: str) -> bool:
        if key not in self.keys:
            return False
        
        key_data = self.keys[key]
        if key_data.get('expires_at'):
            expires_at = datetime.fromisoformat(key_data['expires_at'])
            if datetime.now() > expires_at:
                return False
        
        return True
```

## OAuth 2.0

### OAuth 2.0 Client
```python
from typing import Dict, Optional
import requests
from urllib.parse import urlencode

class OAuth2Client:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        auth_url: str,
        token_url: str
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.auth_url = auth_url
        self.token_url = token_url
        self.token: Optional[Dict] = None
    
    def get_auth_url(self, state: Optional[str] = None) -> str:
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'response_type': 'code',
            'scope': 'read write'
        }
        if state:
            params['state'] = state
        
        return f"{self.auth_url}?{urlencode(params)}"
    
    def get_token(self, code: str) -> Dict:
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': code,
            'redirect_uri': self.redirect_uri,
            'grant_type': 'authorization_code'
        }
        
        response = requests.post(self.token_url, data=data)
        response.raise_for_status()
        self.token = response.json()
        return self.token
    
    def refresh_token(self) -> Dict:
        if not self.token or 'refresh_token' not in self.token:
            raise ValueError("No refresh token available")
        
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'refresh_token': self.token['refresh_token'],
            'grant_type': 'refresh_token'
        }
        
        response = requests.post(self.token_url, data=data)
        response.raise_for_status()
        self.token = response.json()
        return self.token
```

## JWT Tokens

### JWT Token Handler
```python
from typing import Dict, Optional
import jwt
from datetime import datetime, timedelta

class JWTTokenHandler:
    def __init__(self, secret_key: str, algorithm: str = 'HS256'):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_token(
        self,
        data: Dict,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        
        to_encode.update({'exp': expire})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict:
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.JWTError:
            raise ValueError("Invalid token")
```

## Rate Limiting

### Rate Limiter
```python
from typing import Optional
import time
from collections import defaultdict

class RateLimiter:
    def __init__(
        self,
        max_requests: int,
        time_window: int,
        storage: Optional[dict] = None
    ):
        self.max_requests = max_requests
        self.time_window = time_window
        self.storage = storage or defaultdict(list)
    
    def is_allowed(self, key: str) -> bool:
        now = time.time()
        window_start = now - self.time_window
        
        # Clean old requests
        self.storage[key] = [
            timestamp for timestamp in self.storage[key]
            if timestamp > window_start
        ]
        
        # Check if under limit
        if len(self.storage[key]) < self.max_requests:
            self.storage[key].append(now)
            return True
        
        return False
    
    def get_remaining(self, key: str) -> int:
        now = time.time()
        window_start = now - self.time_window
        
        # Clean old requests
        self.storage[key] = [
            timestamp for timestamp in self.storage[key]
            if timestamp > window_start
        ]
        
        return max(0, self.max_requests - len(self.storage[key]))
```

### Redis-based Rate Limiter
```python
import redis
from typing import Optional
import time

class RedisRateLimiter:
    def __init__(
        self,
        redis_client: redis.Redis,
        max_requests: int,
        time_window: int
    ):
        self.redis = redis_client
        self.max_requests = max_requests
        self.time_window = time_window
    
    def is_allowed(self, key: str) -> bool:
        now = time.time()
        window_key = f"ratelimit:{key}"
        
        # Clean old requests
        self.redis.zremrangebyscore(
            window_key,
            0,
            now - self.time_window
        )
        
        # Count requests in window
        request_count = self.redis.zcard(window_key)
        
        if request_count < self.max_requests:
            self.redis.zadd(window_key, {str(now): now})
            self.redis.expire(window_key, self.time_window)
            return True
        
        return False
```

## Best Practices

1. **Security**:
   - Use HTTPS for all API calls
   - Store sensitive data securely
   - Implement proper key rotation
   - Use strong encryption

2. **Error Handling**:
   - Handle token expiration
   - Implement proper retry logic
   - Log authentication failures
   - Provide clear error messages

3. **Performance**:
   - Cache tokens when appropriate
   - Use efficient storage
   - Implement proper cleanup
   - Monitor rate limits

## Common Patterns

1. **Token Refresh**:
```python
class TokenManager:
    def __init__(self, token_handler: JWTTokenHandler):
        self.token_handler = token_handler
        self.token = None
    
    def get_valid_token(self) -> str:
        if not self.token or self._is_token_expired():
            self.token = self._refresh_token()
        return self.token
    
    def _is_token_expired(self) -> bool:
        try:
            self.token_handler.verify_token(self.token)
            return False
        except ValueError:
            return True
    
    def _refresh_token(self) -> str:
        # Implement token refresh logic
        pass
```

2. **API Key Rotation**:
```python
class APIKeyRotator:
    def __init__(self, key_manager: APIKeyManager):
        self.key_manager = key_manager
    
    def rotate_key(self, old_key: str) -> str:
        # Generate new key
        new_key = self._generate_key()
        
        # Add new key
        self.key_manager.add_key(new_key, "Rotated key")
        
        # Schedule old key removal
        self._schedule_key_removal(old_key)
        
        return new_key
    
    def _generate_key(self) -> str:
        # Implement key generation logic
        pass
    
    def _schedule_key_removal(self, key: str):
        # Implement key removal scheduling
        pass
```

## Further Reading

- [OAuth 2.0 Specification](https://oauth.net/2/)
- [JWT Documentation](https://jwt.io/introduction)
- [Rate Limiting Best Practices](https://cloud.google.com/architecture/rate-limiting-strategies-techniques)
- [API Security Best Practices](https://owasp.org/www-project-api-security/)
- [Python JWT Library](https://pyjwt.readthedocs.io/) 