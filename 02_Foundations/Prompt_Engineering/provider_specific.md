# Provider-Specific Prompt Engineering

This guide covers prompt engineering techniques and best practices for different LLM providers, including OpenAI GPT, Anthropic Claude, Azure OpenAI, and Custom Models.

## OpenAI GPT

### GPT-4 Specific
```python
class GPT4Prompt:
    def __init__(self):
        self.system_message = ""
        self.user_message = ""
        self.temperature = 0.7
        self.max_tokens = None
    
    def set_system_message(self, message: str):
        self.system_message = message
    
    def set_user_message(self, message: str):
        self.user_message = message
    
    def set_temperature(self, temperature: float):
        self.temperature = temperature
    
    def set_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens
    
    def format(self) -> Dict[str, Any]:
        messages = []
        
        if self.system_message:
            messages.append({
                "role": "system",
                "content": self.system_message
            })
        
        messages.append({
            "role": "user",
            "content": self.user_message
        })
        
        return {
            "model": "gpt-4",
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
```

### GPT-3.5 Specific
```python
class GPT35Prompt:
    def __init__(self):
        self.system_message = ""
        self.user_message = ""
        self.temperature = 0.7
        self.max_tokens = None
    
    def set_system_message(self, message: str):
        self.system_message = message
    
    def set_user_message(self, message: str):
        self.user_message = message
    
    def set_temperature(self, temperature: float):
        self.temperature = temperature
    
    def set_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens
    
    def format(self) -> Dict[str, Any]:
        messages = []
        
        if self.system_message:
            messages.append({
                "role": "system",
                "content": self.system_message
            })
        
        messages.append({
            "role": "user",
            "content": self.user_message
        })
        
        return {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
```

## Anthropic Claude

### Claude 3 Specific
```python
class Claude3Prompt:
    def __init__(self):
        self.system_message = ""
        self.user_message = ""
        self.temperature = 0.7
        self.max_tokens = None
    
    def set_system_message(self, message: str):
        self.system_message = message
    
    def set_user_message(self, message: str):
        self.user_message = message
    
    def set_temperature(self, temperature: float):
        self.temperature = temperature
    
    def set_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens
    
    def format(self) -> Dict[str, Any]:
        return {
            "model": "claude-3-opus-20240229",
            "system": self.system_message,
            "messages": [{
                "role": "user",
                "content": self.user_message
            }],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
```

### Claude 2 Specific
```python
class Claude2Prompt:
    def __init__(self):
        self.system_message = ""
        self.user_message = ""
        self.temperature = 0.7
        self.max_tokens = None
    
    def set_system_message(self, message: str):
        self.system_message = message
    
    def set_user_message(self, message: str):
        self.user_message = message
    
    def set_temperature(self, temperature: float):
        self.temperature = temperature
    
    def set_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens
    
    def format(self) -> Dict[str, Any]:
        return {
            "model": "claude-2",
            "system": self.system_message,
            "messages": [{
                "role": "user",
                "content": self.user_message
            }],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
```

## Azure OpenAI

### Azure GPT-4
```python
class AzureGPT4Prompt:
    def __init__(self, deployment_name: str):
        self.deployment_name = deployment_name
        self.system_message = ""
        self.user_message = ""
        self.temperature = 0.7
        self.max_tokens = None
    
    def set_system_message(self, message: str):
        self.system_message = message
    
    def set_user_message(self, message: str):
        self.user_message = message
    
    def set_temperature(self, temperature: float):
        self.temperature = temperature
    
    def set_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens
    
    def format(self) -> Dict[str, Any]:
        messages = []
        
        if self.system_message:
            messages.append({
                "role": "system",
                "content": self.system_message
            })
        
        messages.append({
            "role": "user",
            "content": self.user_message
        })
        
        return {
            "deployment_name": self.deployment_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
```

### Azure GPT-3.5
```python
class AzureGPT35Prompt:
    def __init__(self, deployment_name: str):
        self.deployment_name = deployment_name
        self.system_message = ""
        self.user_message = ""
        self.temperature = 0.7
        self.max_tokens = None
    
    def set_system_message(self, message: str):
        self.system_message = message
    
    def set_user_message(self, message: str):
        self.user_message = message
    
    def set_temperature(self, temperature: float):
        self.temperature = temperature
    
    def set_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens
    
    def format(self) -> Dict[str, Any]:
        messages = []
        
        if self.system_message:
            messages.append({
                "role": "system",
                "content": self.system_message
            })
        
        messages.append({
            "role": "user",
            "content": self.user_message
        })
        
        return {
            "deployment_name": self.deployment_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
```

## Custom Models

### Generic LLM Prompt
```python
class GenericLLMPrompt:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.system_message = ""
        self.user_message = ""
        self.temperature = 0.7
        self.max_tokens = None
        self.additional_params = {}
    
    def set_system_message(self, message: str):
        self.system_message = message
    
    def set_user_message(self, message: str):
        self.user_message = message
    
    def set_temperature(self, temperature: float):
        self.temperature = temperature
    
    def set_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens
    
    def set_additional_param(self, key: str, value: Any):
        self.additional_params[key] = value
    
    def format(self) -> Dict[str, Any]:
        base_params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        if self.system_message:
            base_params["system"] = self.system_message
        
        base_params["messages"] = [{
            "role": "user",
            "content": self.user_message
        }]
        
        return {**base_params, **self.additional_params}
```

### Custom Model Adapter
```python
class CustomModelAdapter:
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.prompt = GenericLLMPrompt(model_config["model_name"])
    
    def adapt_prompt(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        # Implement model-specific prompt adaptation
        adapted_prompt = prompt.copy()
        
        # Add model-specific parameters
        for key, value in self.model_config.get("default_params", {}).items():
            if key not in adapted_prompt:
                adapted_prompt[key] = value
        
        return adapted_prompt
    
    def format(self) -> Dict[str, Any]:
        base_prompt = self.prompt.format()
        return self.adapt_prompt(base_prompt)
```

## Best Practices

1. **OpenAI GPT**:
   - Use system messages effectively
   - Leverage few-shot examples
   - Control temperature carefully
   - Monitor token usage

2. **Anthropic Claude**:
   - Use clear instructions
   - Provide context
   - Set appropriate constraints
   - Use system prompts

3. **Azure OpenAI**:
   - Configure deployment settings
   - Use appropriate models
   - Monitor costs
   - Handle rate limits

4. **Custom Models**:
   - Understand model capabilities
   - Test thoroughly
   - Monitor performance
   - Adapt prompts

## Common Patterns

1. **Provider-Specific System Messages**:
```python
SYSTEM_MESSAGES = {
    "openai": "You are a helpful AI assistant.",
    "anthropic": "You are Claude, an AI assistant created by Anthropic.",
    "azure": "You are an AI assistant powered by Azure OpenAI.",
    "custom": "You are a custom AI assistant."
}
```

2. **Model-Specific Parameters**:
```python
MODEL_PARAMS = {
    "gpt-4": {
        "temperature": 0.7,
        "max_tokens": 2000
    },
    "claude-3": {
        "temperature": 0.7,
        "max_tokens": 4000
    },
    "azure-gpt-4": {
        "temperature": 0.7,
        "max_tokens": 2000
    }
}
```

3. **Provider-Specific Formatting**:
```python
def format_for_provider(provider: str, prompt: Dict[str, Any]) -> Dict[str, Any]:
    if provider == "openai":
        return format_openai_prompt(prompt)
    elif provider == "anthropic":
        return format_anthropic_prompt(prompt)
    elif provider == "azure":
        return format_azure_prompt(prompt)
    else:
        return format_custom_prompt(prompt)
```

## Further Reading

- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Anthropic Claude Documentation](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/)
- [Custom Model Integration Guide](https://www.promptingguide.ai/models)
- [Provider-Specific Best Practices](https://www.promptingguide.ai/providers) 