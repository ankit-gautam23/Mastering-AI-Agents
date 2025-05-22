# Simple Chat Agent

A beginner-friendly project to build a simple chat agent that can engage in basic conversations and respond to user queries.

## 🎯 Project Overview

In this project, you'll build a chat agent that can:
- Respond to basic greetings
- Answer simple questions
- Maintain conversation context
- Handle basic error cases

## 📋 Requirements

### Functional Requirements
1. The agent should respond to basic greetings (hello, hi, hey)
2. The agent should be able to answer simple questions
3. The agent should maintain conversation history
4. The agent should handle unknown inputs gracefully

### Technical Requirements
1. Implement the `ChatAgent` class
2. Use proper error handling
3. Write unit tests
4. Document your code

## 🛠️ Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Complete the TODO items in the code
4. Run tests:
   ```bash
   pytest tests/
   ```

## 📝 Code Structure

```
chat_agent/
├── src/
│   ├── main.py
│   ├── agent.py
│   └── utils/
│       └── text_processor.py
├── tests/
│   └── test_agent.py
└── requirements.txt
```

## 🎯 Implementation Tasks

### 1. Complete the ChatAgent Class
```python
class ChatAgent:
    def __init__(self, name):
        # TODO: Initialize agent properties
        pass
    
    def process_message(self, message):
        # TODO: Process incoming messages
        pass
    
    def generate_response(self, message):
        # TODO: Generate appropriate responses
        pass
    
    def get_conversation_history(self):
        # TODO: Return conversation history
        pass
```

### 2. Implement Text Processing
```python
class TextProcessor:
    def __init__(self):
        # TODO: Initialize text processing components
        pass
    
    def preprocess(self, text):
        # TODO: Clean and normalize input text
        pass
    
    def extract_intent(self, text):
        # TODO: Determine user intent
        pass
```

### 3. Add Error Handling
```python
class ChatError(Exception):
    pass

def handle_error(error):
    # TODO: Implement error handling
    pass
```

## 📊 Expected Output

```
User: Hello
Agent: Hi! How can I help you today?

User: What's your name?
Agent: I'm ChatBot, nice to meet you!

User: Tell me a joke
Agent: Why don't scientists trust atoms? Because they make up everything!

User: Goodbye
Agent: Goodbye! Have a great day!
```

## 🎯 Learning Objectives

1. Basic Python programming
2. Object-oriented programming
3. Error handling
4. Unit testing
5. Documentation

## 📚 Resources

### Documentation
- [Python Documentation](https://docs.python.org/3/)
- [Unit Testing](https://docs.pytest.org/)

### Tutorials
- [Python OOP](https://www.example.com/python-oop)
- [Error Handling](https://www.example.com/error-handling)

## 🎯 Evaluation Criteria

Your implementation will be evaluated based on:
1. Code completeness
2. Error handling
3. Test coverage
4. Code documentation
5. Code organization
6. Response quality

## 📝 Submission

1. Complete all TODO items
2. Write unit tests
3. Document your code
4. Create a pull request

## 🎓 Next Steps

After completing this project, you can:
1. Add more advanced features
2. Implement natural language processing
3. Add a web interface
4. Create a multi-agent system 