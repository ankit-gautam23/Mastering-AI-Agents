# Prompt Templates

This guide covers the creation and management of prompt templates, including structure, variable substitution, dynamic prompts, and reusable patterns.

## Template Structure

### Basic Template
```python
from typing import Dict, Any
from string import Template

class BasicPromptTemplate:
    def __init__(self, template: str):
        self.template = Template(template)
    
    def format(self, **kwargs) -> str:
        return self.template.substitute(**kwargs)
```

### Structured Template
```python
class StructuredTemplate:
    def __init__(self):
        self.system_message = ""
        self.user_message = ""
        self.examples = []
        self.variables = {}
    
    def set_system_message(self, message: str):
        self.system_message = message
    
    def set_user_message(self, message: str):
        self.user_message = message
    
    def add_example(self, input_text: str, output_text: str):
        self.examples.append({
            "input": input_text,
            "output": output_text
        })
    
    def set_variables(self, variables: Dict[str, Any]):
        self.variables = variables
    
    def format(self) -> str:
        # Format system message
        system = Template(self.system_message).substitute(**self.variables)
        
        # Format examples
        examples = ""
        for example in self.examples:
            input_text = Template(example["input"]).substitute(**self.variables)
            output_text = Template(example["output"]).substitute(**self.variables)
            examples += f"Input: {input_text}\nOutput: {output_text}\n\n"
        
        # Format user message
        user = Template(self.user_message).substitute(**self.variables)
        
        return f"{system}\n\n{examples}{user}"
```

## Variable Substitution

### Template Variables
```python
class TemplateVariables:
    def __init__(self):
        self.variables = {}
    
    def add_variable(self, name: str, value: Any):
        self.variables[name] = value
    
    def remove_variable(self, name: str):
        if name in self.variables:
            del self.variables[name]
    
    def get_variable(self, name: str) -> Any:
        return self.variables.get(name)
    
    def format_template(self, template: str) -> str:
        return Template(template).substitute(**self.variables)
```

### Dynamic Variables
```python
class DynamicVariables:
    def __init__(self):
        self.variables = {}
        self.functions = {}
    
    def add_variable(self, name: str, value: Any):
        self.variables[name] = value
    
    def add_function(self, name: str, func: callable):
        self.functions[name] = func
    
    def format_template(self, template: str) -> str:
        # First substitute regular variables
        result = Template(template).substitute(**self.variables)
        
        # Then substitute function results
        for name, func in self.functions.items():
            result = result.replace(f"${{{name}}}", str(func()))
        
        return result
```

## Dynamic Prompts

### Dynamic Prompt Generator
```python
class DynamicPromptGenerator:
    def __init__(self):
        self.templates = {}
        self.variables = {}
    
    def add_template(self, name: str, template: str):
        self.templates[name] = template
    
    def set_variables(self, variables: Dict[str, Any]):
        self.variables = variables
    
    def generate_prompt(self, template_name: str) -> str:
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.templates[template_name]
        return Template(template).substitute(**self.variables)
```

### Context-Aware Prompts
```python
class ContextAwarePrompt:
    def __init__(self):
        self.context = {}
        self.template = ""
    
    def set_context(self, context: Dict[str, Any]):
        self.context = context
    
    def set_template(self, template: str):
        self.template = template
    
    def generate_prompt(self) -> str:
        # Add context to variables
        variables = self.context.copy()
        
        # Add derived variables
        variables["context_length"] = len(str(self.context))
        variables["has_context"] = bool(self.context)
        
        return Template(self.template).substitute(**variables)
```

## Reusable Patterns

### Pattern Library
```python
class PatternLibrary:
    def __init__(self):
        self.patterns = {}
    
    def add_pattern(self, name: str, pattern: str):
        self.patterns[name] = pattern
    
    def get_pattern(self, name: str) -> str:
        if name not in self.patterns:
            raise ValueError(f"Pattern '{name}' not found")
        return self.patterns[name]
    
    def combine_patterns(self, pattern_names: List[str], separator: str = "\n") -> str:
        patterns = [self.get_pattern(name) for name in pattern_names]
        return separator.join(patterns)
```

### Pattern Composer
```python
class PatternComposer:
    def __init__(self):
        self.patterns = {}
        self.variables = {}
    
    def add_pattern(self, name: str, pattern: str):
        self.patterns[name] = pattern
    
    def set_variables(self, variables: Dict[str, Any]):
        self.variables = variables
    
    def compose_prompt(self, pattern_sequence: List[str]) -> str:
        prompt_parts = []
        
        for pattern_name in pattern_sequence:
            if pattern_name not in self.patterns:
                raise ValueError(f"Pattern '{pattern_name}' not found")
            
            pattern = self.patterns[pattern_name]
            formatted_pattern = Template(pattern).substitute(**self.variables)
            prompt_parts.append(formatted_pattern)
        
        return "\n\n".join(prompt_parts)
```

## Best Practices

1. **Template Design**:
   - Use clear variable names
   - Include default values
   - Document template usage
   - Validate input data

2. **Variable Management**:
   - Type check variables
   - Handle missing values
   - Sanitize input
   - Cache results

3. **Pattern Organization**:
   - Group related patterns
   - Version control templates
   - Document patterns
   - Test templates

## Common Patterns

1. **System Message Pattern**:
```python
SYSTEM_MESSAGE_PATTERN = """You are an expert ${role}. Your task is to ${task}.
Please provide a detailed and accurate response based on your expertise."""
```

2. **Example Pattern**:
```python
EXAMPLE_PATTERN = """Input: ${input}
Output: ${output}"""
```

3. **Format Pattern**:
```python
FORMAT_PATTERN = """Please provide your response in the following format:
${format}"""
```

## Further Reading

- [Python String Template](https://docs.python.org/3/library/string.html#template-strings)
- [Jinja2 Documentation](https://jinja.palletsprojects.com/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [LangChain Templates](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates)
- [Semantic Kernel Templates](https://learn.microsoft.com/en-us/semantic-kernel/prompts/) 