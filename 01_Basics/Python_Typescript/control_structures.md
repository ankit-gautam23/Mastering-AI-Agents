# Control Structures in Python and TypeScript

This guide covers the essential control structures in both Python and TypeScript, including conditionals, loops, and error handling.

## Conditional Statements

### Python Conditionals
```python
# Basic if-else
age = 18
if age >= 18:
    print("Adult")
else:
    print("Minor")

# if-elif-else
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"

# Ternary operator
status = "Adult" if age >= 18 else "Minor"
```

### TypeScript Conditionals
```typescript
// Basic if-else
let age: number = 18;
if (age >= 18) {
    console.log("Adult");
} else {
    console.log("Minor");
}

// if-else if-else
let score: number = 85;
let grade: string;
if (score >= 90) {
    grade = "A";
} else if (score >= 80) {
    grade = "B";
} else if (score >= 70) {
    grade = "C";
} else {
    grade = "F";
}

// Ternary operator
let status: string = age >= 18 ? "Adult" : "Minor";
```

## Loops

### Python Loops
```python
# For loop with range
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

# For loop with list
fruits = ['apple', 'banana', 'orange']
for fruit in fruits:
    print(fruit)

# While loop
count = 0
while count < 5:
    print(count)
    count += 1

# List comprehension
squares = [x**2 for x in range(5)]  # [0, 1, 4, 9, 16]
```

### TypeScript Loops
```typescript
// For loop
for (let i = 0; i < 5; i++) {
    console.log(i);  // 0, 1, 2, 3, 4
}

// For...of loop
const fruits: string[] = ['apple', 'banana', 'orange'];
for (const fruit of fruits) {
    console.log(fruit);
}

// While loop
let count: number = 0;
while (count < 5) {
    console.log(count);
    count++;
}

// Array methods
const squares: number[] = Array.from({length: 5}, (_, i) => i * i);
```

## Error Handling

### Python Error Handling
```python
# Try-except block
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
except Exception as e:
    print(f"An error occurred: {e}")
else:
    print("No errors occurred")
finally:
    print("This always executes")

# Custom exception
class CustomError(Exception):
    pass

def validate_age(age):
    if age < 0:
        raise CustomError("Age cannot be negative")
    return True
```

### TypeScript Error Handling
```typescript
// Try-catch block
try {
    const result: number = 10 / 0;
} catch (error) {
    if (error instanceof Error) {
        console.log(`An error occurred: ${error.message}`);
    }
} finally {
    console.log("This always executes");
}

// Custom error
class CustomError extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'CustomError';
    }
}

function validateAge(age: number): boolean {
    if (age < 0) {
        throw new CustomError("Age cannot be negative");
    }
    return true;
}
```

## Switch Statements

### Python Switch (Python 3.10+)
```python
# Match statement (Python 3.10+)
def get_day_name(day):
    match day:
        case 1:
            return "Monday"
        case 2:
            return "Tuesday"
        case 3:
            return "Wednesday"
        case _:
            return "Unknown"
```

### TypeScript Switch
```typescript
function getDayName(day: number): string {
    switch (day) {
        case 1:
            return "Monday";
        case 2:
            return "Tuesday";
        case 3:
            return "Wednesday";
        default:
            return "Unknown";
    }
}
```

## Best Practices

1. **Conditionals**:
   - Use early returns to reduce nesting
   - Keep conditions simple and readable
   - Use meaningful variable names
   - Avoid deep nesting of if statements

2. **Loops**:
   - Use appropriate loop type for the task
   - Avoid modifying loop variables
   - Use break and continue judiciously
   - Consider using list comprehensions in Python

3. **Error Handling**:
   - Catch specific exceptions
   - Don't catch all exceptions blindly
   - Use finally for cleanup
   - Log errors appropriately

## Common Patterns

1. **Guard Clauses**:
```python
# Python
def process_user(user):
    if not user.is_active:
        return
    if not user.has_permission:
        return
    # Process user
```

```typescript
// TypeScript
function processUser(user: User): void {
    if (!user.isActive) return;
    if (!user.hasPermission) return;
    // Process user
}
```

2. **Loop with Break**:
```python
# Python
for item in items:
    if item == target:
        found = True
        break
```

```typescript
// TypeScript
let found = false;
for (const item of items) {
    if (item === target) {
        found = true;
        break;
    }
}
```

## Further Reading

- [Python Control Flow](https://docs.python.org/3/tutorial/controlflow.html)
- [TypeScript Control Flow](https://www.typescriptlang.org/docs/handbook/control-flow-analysis.html)
- [Python Error Handling](https://docs.python.org/3/tutorial/errors.html)
- [TypeScript Error Handling](https://www.typescriptlang.org/docs/handbook/declaration-files/by-example.html#handling-errors) 