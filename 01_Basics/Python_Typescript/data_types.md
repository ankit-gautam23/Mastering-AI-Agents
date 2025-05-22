# Data Types in Python and TypeScript

This guide covers the fundamental data types in both Python and TypeScript, with practical examples and use cases.

## Python Data Types

### 1. Numbers
```python
# Integers
x = 10
y = -5

# Floating-point numbers
pi = 3.14159
e = 2.71828

# Complex numbers
z = 1 + 2j
```

### 2. Strings
```python
# Single and double quotes
name = 'John'
message = "Hello, World!"

# String methods
print(name.upper())  # JOHN
print(message.lower())  # hello, world!
print(len(message))  # 13

# String formatting
age = 25
print(f"{name} is {age} years old")
```

### 3. Lists
```python
# Creating lists
fruits = ['apple', 'banana', 'orange']
numbers = [1, 2, 3, 4, 5]

# List operations
fruits.append('grape')
fruits.remove('banana')
print(fruits[0])  # apple
print(len(fruits))  # 3
```

### 4. Tuples
```python
# Immutable sequences
coordinates = (10, 20)
rgb = (255, 0, 0)

# Tuple unpacking
x, y = coordinates
r, g, b = rgb
```

### 5. Dictionaries
```python
# Key-value pairs
person = {
    'name': 'John',
    'age': 30,
    'city': 'New York'
}

# Dictionary operations
print(person['name'])  # John
person['job'] = 'Developer'
del person['age']
```

## TypeScript Data Types

### 1. Numbers
```typescript
// Numbers (all floating-point)
let x: number = 10;
let y: number = -5;
let pi: number = 3.14159;
```

### 2. Strings
```typescript
// String types
let name: string = 'John';
let message: string = "Hello, World!";

// Template literals
let age: number = 25;
console.log(`${name} is ${age} years old`);
```

### 3. Arrays
```typescript
// Array types
let fruits: string[] = ['apple', 'banana', 'orange'];
let numbers: number[] = [1, 2, 3, 4, 5];

// Array operations
fruits.push('grape');
fruits.splice(fruits.indexOf('banana'), 1);
console.log(fruits[0]);  // apple
console.log(fruits.length);  // 3
```

### 4. Tuples
```typescript
// Fixed-length arrays
let coordinates: [number, number] = [10, 20];
let rgb: [number, number, number] = [255, 0, 0];

// Tuple destructuring
let [x, y] = coordinates;
let [r, g, b] = rgb;
```

### 5. Objects
```typescript
// Object types
interface Person {
    name: string;
    age: number;
    city: string;
}

let person: Person = {
    name: 'John',
    age: 30,
    city: 'New York'
};

// Object operations
console.log(person.name);  // John
person.job = 'Developer';  // Error: Property 'job' does not exist
```

## Type Checking and Type Safety

### Python Type Hints
```python
from typing import List, Dict, Tuple

def greet(name: str) -> str:
    return f"Hello, {name}!"

def process_numbers(numbers: List[int]) -> int:
    return sum(numbers)

def get_person() -> Dict[str, str]:
    return {"name": "John", "city": "New York"}
```

### TypeScript Type Safety
```typescript
// Function type definitions
function greet(name: string): string {
    return `Hello, ${name}!`;
}

function processNumbers(numbers: number[]): number {
    return numbers.reduce((a, b) => a + b, 0);
}

function getPerson(): { name: string; city: string } {
    return { name: "John", city: "New York" };
}
```

## Best Practices

1. **Python**:
   - Use type hints for better code documentation
   - Prefer list comprehensions over loops when possible
   - Use dictionaries for key-value data structures
   - Leverage tuple unpacking for clean code

2. **TypeScript**:
   - Always define types for function parameters and return values
   - Use interfaces for object type definitions
   - Leverage union types for flexible type definitions
   - Use type guards for runtime type checking

## Common Use Cases

1. **Data Processing**:
   - Lists/Arrays for sequential data
   - Dictionaries/Objects for structured data
   - Tuples for fixed-length data

2. **API Development**:
   - Dictionaries/Objects for JSON responses
   - Lists/Arrays for collections
   - Strings for text processing

3. **Configuration**:
   - Dictionaries/Objects for settings
   - Tuples for immutable configurations
   - Lists/Arrays for ordered settings

## Further Reading

- [Python Type Hints Documentation](https://docs.python.org/3/library/typing.html)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/basic-types.html)
- [Python Data Structures](https://docs.python.org/3/tutorial/datastructures.html)
- [TypeScript Advanced Types](https://www.typescriptlang.org/docs/handbook/advanced-types.html) 