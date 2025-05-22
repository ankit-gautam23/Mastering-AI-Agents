# GPT Wrappers

This guide covers the implementation of wrappers for various GPT and LLM APIs, including OpenAI, Azure OpenAI, and Anthropic Claude.

## OpenAI API Integration

### Basic OpenAI Client
```python
from openai import OpenAI
from typing import Dict, Any, List, Optional

class OpenAIClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7
    ) -> Generator[str, None, None]:
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

### Async OpenAI Client
```python
from openai import AsyncOpenAI
from typing import Dict, Any, List, Optional, AsyncGenerator

class AsyncOpenAIClient:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

## Azure OpenAI Integration

### Azure OpenAI Client
```python
from openai import AzureOpenAI
from typing import Dict, Any, List, Optional

class AzureOpenAIClient:
    def __init__(self, api_key: str, endpoint: str, api_version: str = "2023-05-15"):
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        deployment_name: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
```

## Anthropic Claude Integration

### Claude Client
```python
import anthropic
from typing import Dict, Any, List, Optional

class ClaudeClient:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        response = self.client.messages.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.content[0].text
    
    def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.7
    ) -> Generator[str, None, None]:
        stream = self.client.messages.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True
        )
        for chunk in stream:
            if chunk.type == "content_block_delta":
                yield chunk.delta.text
```

## Custom Model Integration

### Generic LLM Client
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Generator, AsyncGenerator

class LLMClient(ABC):
    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Generator[str, None, None]:
        pass

class CustomLLMClient(LLMClient):
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        # Implement custom API call
        pass
    
    def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Generator[str, None, None]:
        # Implement custom streaming
        pass
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

1. **Message History Management**:
```python
class MessageHistory:
    def __init__(self, max_tokens: int = 4000):
        self.messages = []
        self.max_tokens = max_tokens
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._trim_history()
    
    def _trim_history(self):
        # Implement token counting and trimming logic
        pass
```

2. **Response Caching**:
```python
from functools import lru_cache

class CachedLLMClient:
    def __init__(self, client: LLMClient):
        self.client = client
    
    @lru_cache(maxsize=100)
    def chat_completion(self, messages: str, **kwargs):
        return self.client.chat_completion(eval(messages), **kwargs)
```

3. **Rate Limiting**:
```python
from ratelimit import limits, sleep_and_retry

class RateLimitedLLMClient:
    def __init__(self, client: LLMClient, calls: int = 60, period: int = 60):
        self.client = client
        self.calls = calls
        self.period = period
    
    @sleep_and_retry
    @limits(calls=60, period=60)
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs):
        return self.client.chat_completion(messages, **kwargs)
```

## Further Reading

- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/)
- [Anthropic Claude Documentation](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
- [Python OpenAI Library](https://github.com/openai/openai-python)
- [Python Anthropic Library](https://github.com/anthropics/anthropic-sdk-python) 