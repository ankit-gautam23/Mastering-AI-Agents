# Basic Prompt Engineering Concepts

This guide covers the fundamental concepts of prompt engineering, focusing on understanding prompts, their components, and effective structuring.

## Understanding Prompts

### What is a Prompt?
A prompt is a text input that guides an LLM to generate a desired response. It can include:
- Instructions
- Context
- Examples
- Constraints
- Output format

### Basic Prompt Structure
```python
from typing import List, Dict, Any

class BasicPrompt:
    def __init__(self):
        self.system_message = ""
        self.user_message = ""
        self.examples: List[Dict[str, str]] = []
    
    def set_system_message(self, message: str):
        self.system_message = message
    
    def set_user_message(self, message: str):
        self.user_message = message
    
    def add_example(self, input_text: str, output_text: str):
        self.examples.append({
            "input": input_text,
            "output": output_text
        })
    
    def format(self) -> List[Dict[str, str]]:
        messages = []
        
        if self.system_message:
            messages.append({
                "role": "system",
                "content": self.system_message
            })
        
        for example in self.examples:
            messages.append({
                "role": "user",
                "content": example["input"]
            })
            messages.append({
                "role": "assistant",
                "content": example["output"]
            })
        
        messages.append({
            "role": "user",
            "content": self.user_message
        })
        
        return messages
```

## Prompt Components

### System Message
```python
class SystemMessage:
    def __init__(self):
        self.role = "system"
        self.content = ""
    
    def set_role(self, role: str):
        self.role = role
    
    def set_content(self, content: str):
        self.content = content
    
    def format(self) -> Dict[str, str]:
        return {
            "role": self.role,
            "content": self.content
        }
```

### User Message
```python
class UserMessage:
    def __init__(self):
        self.role = "user"
        self.content = ""
    
    def set_content(self, content: str):
        self.content = content
    
    def format(self) -> Dict[str, str]:
        return {
            "role": self.role,
            "content": self.content
        }
```

### Assistant Message
```python
class AssistantMessage:
    def __init__(self):
        self.role = "assistant"
        self.content = ""
    
    def set_content(self, content: str):
        self.content = content
    
    def format(self) -> Dict[str, str]:
        return {
            "role": self.role,
            "content": self.content
        }
```

## Role and Context

### Role Definition
```python
class Role:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def format(self) -> str:
        return f"You are {self.name}. {self.description}"
```

### Context Management
```python
class Context:
    def __init__(self):
        self.background = ""
        self.constraints = []
        self.requirements = []
    
    def set_background(self, background: str):
        self.background = background
    
    def add_constraint(self, constraint: str):
        self.constraints.append(constraint)
    
    def add_requirement(self, requirement: str):
        self.requirements.append(requirement)
    
    def format(self) -> str:
        context = self.background + "\n\n"
        
        if self.constraints:
            context += "Constraints:\n"
            for constraint in self.constraints:
                context += f"- {constraint}\n"
        
        if self.requirements:
            context += "\nRequirements:\n"
            for requirement in self.requirements:
                context += f"- {requirement}\n"
        
        return context
```

## Format and Structure

### Output Format
```python
class OutputFormat:
    def __init__(self):
        self.format_type = ""
        self.schema = {}
    
    def set_format_type(self, format_type: str):
        self.format_type = format_type
    
    def set_schema(self, schema: Dict[str, Any]):
        self.schema = schema
    
    def format(self) -> str:
        if self.format_type == "json":
            return f"Respond in JSON format with the following schema:\n{json.dumps(self.schema, indent=2)}"
        elif self.format_type == "markdown":
            return "Respond in Markdown format"
        else:
            return "Respond in plain text"
```

### Prompt Builder
```python
class PromptBuilder:
    def __init__(self):
        self.system_message = SystemMessage()
        self.context = Context()
        self.output_format = OutputFormat()
        self.examples: List[Dict[str, str]] = []
    
    def set_system_role(self, role: Role):
        self.system_message.set_content(role.format())
    
    def set_context(self, context: Context):
        self.context = context
    
    def set_output_format(self, output_format: OutputFormat):
        self.output_format = output_format
    
    def add_example(self, input_text: str, output_text: str):
        self.examples.append({
            "input": input_text,
            "output": output_text
        })
    
    def build(self) -> List[Dict[str, str]]:
        messages = []
        
        # Add system message
        if self.system_message.content:
            messages.append(self.system_message.format())
        
        # Add context
        if self.context.background or self.context.constraints or self.context.requirements:
            messages.append({
                "role": "system",
                "content": self.context.format()
            })
        
        # Add examples
        for example in self.examples:
            messages.append({
                "role": "user",
                "content": example["input"]
            })
            messages.append({
                "role": "assistant",
                "content": example["output"]
            })
        
        # Add output format
        if self.output_format.format_type:
            messages.append({
                "role": "system",
                "content": self.output_format.format()
            })
        
        return messages
```

## Best Practices

1. **Clarity**:
   - Use clear and concise language
   - Be specific about requirements
   - Avoid ambiguity
   - Use proper formatting

2. **Context**:
   - Provide relevant background
   - Set clear constraints
   - Include necessary details
   - Maintain consistency

3. **Examples**:
   - Use relevant examples
   - Show input-output pairs
   - Demonstrate edge cases
   - Include error cases

## Common Patterns

1. **Role-Based Prompting**:
```python
def create_role_based_prompt(role: str, task: str) -> str:
    return f"""You are an expert {role}. Your task is to {task}.
Please provide a detailed and accurate response based on your expertise."""
```

2. **Context-Aware Prompting**:
```python
def create_context_aware_prompt(context: str, question: str) -> str:
    return f"""Given the following context:
{context}

Please answer the following question:
{question}"""
```

3. **Format-Specific Prompting**:
```python
def create_format_specific_prompt(format_type: str, content: str) -> str:
    return f"""Please provide the following information in {format_type} format:
{content}"""
```

## Further Reading

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Semantic Kernel Documentation](https://learn.microsoft.com/en-us/semantic-kernel/) 