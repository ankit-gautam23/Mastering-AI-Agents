# File I/O Operations in Python and TypeScript

This guide covers file input/output operations in both Python and TypeScript, including reading, writing, and managing files.

## Basic File Operations

### Python File Operations
```python
# Writing to a file
with open('example.txt', 'w') as file:
    file.write('Hello, World!')

# Reading from a file
with open('example.txt', 'r') as file:
    content = file.read()
    print(content)  # Hello, World!

# Appending to a file
with open('example.txt', 'a') as file:
    file.write('\nNew line added')
```

### TypeScript File Operations
```typescript
import * as fs from 'fs';

// Writing to a file
fs.writeFileSync('example.txt', 'Hello, World!');

// Reading from a file
const content = fs.readFileSync('example.txt', 'utf-8');
console.log(content);  // Hello, World!

// Appending to a file
fs.appendFileSync('example.txt', '\nNew line added');
```

## Working with Different File Types

### Python File Types
```python
# JSON files
import json

# Writing JSON
data = {'name': 'John', 'age': 30}
with open('data.json', 'w') as file:
    json.dump(data, file, indent=4)

# Reading JSON
with open('data.json', 'r') as file:
    data = json.load(file)
    print(data)  # {'name': 'John', 'age': 30}

# CSV files
import csv

# Writing CSV
with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Age'])
    writer.writerow(['John', 30])

# Reading CSV
with open('data.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)
```

### TypeScript File Types
```typescript
import * as fs from 'fs';

// JSON files
// Writing JSON
const data = { name: 'John', age: 30 };
fs.writeFileSync('data.json', JSON.stringify(data, null, 4));

// Reading JSON
const jsonData = JSON.parse(fs.readFileSync('data.json', 'utf-8'));
console.log(jsonData);  // { name: 'John', age: 30 }

// CSV files
import { parse, stringify } from 'csv-parse/sync';

// Writing CSV
const csvData = [
    ['Name', 'Age'],
    ['John', '30']
];
fs.writeFileSync('data.csv', stringify(csvData));

// Reading CSV
const csvContent = fs.readFileSync('data.csv', 'utf-8');
const records = parse(csvContent, {
    columns: true,
    skip_empty_lines: true
});
```

## File System Operations

### Python File System
```python
import os
import shutil

# Create directory
os.makedirs('new_directory', exist_ok=True)

# List directory contents
files = os.listdir('.')
print(files)

# Check if file exists
if os.path.exists('example.txt'):
    print('File exists')

# Copy file
shutil.copy('source.txt', 'destination.txt')

# Move file
shutil.move('old_location.txt', 'new_location.txt')

# Delete file
os.remove('unwanted.txt')
```

### TypeScript File System
```typescript
import * as fs from 'fs';
import * as path from 'path';

// Create directory
fs.mkdirSync('new_directory', { recursive: true });

// List directory contents
const files = fs.readdirSync('.');
console.log(files);

// Check if file exists
if (fs.existsSync('example.txt')) {
    console.log('File exists');
}

// Copy file
fs.copyFileSync('source.txt', 'destination.txt');

// Move file
fs.renameSync('old_location.txt', 'new_location.txt');

// Delete file
fs.unlinkSync('unwanted.txt');
```

## Asynchronous File Operations

### Python Async File Operations
```python
import asyncio
import aiofiles

async def async_file_operations():
    # Async write
    async with aiofiles.open('async.txt', 'w') as file:
        await file.write('Async content')

    # Async read
    async with aiofiles.open('async.txt', 'r') as file:
        content = await file.read()
        print(content)
```

### TypeScript Async File Operations
```typescript
import * as fs from 'fs/promises';

async function asyncFileOperations() {
    // Async write
    await fs.writeFile('async.txt', 'Async content');

    // Async read
    const content = await fs.readFile('async.txt', 'utf-8');
    console.log(content);
}
```

## Best Practices

1. **File Handling**:
   - Always close files after use
   - Use context managers in Python
   - Handle exceptions appropriately
   - Use appropriate file modes

2. **Performance**:
   - Use buffered I/O for large files
   - Consider async operations for I/O-bound tasks
   - Use appropriate chunk sizes for reading

3. **Security**:
   - Validate file paths
   - Check file permissions
   - Sanitize file names
   - Handle sensitive data appropriately

## Common Patterns

1. **Reading Large Files**:
```python
# Python
def read_large_file(filename):
    with open(filename, 'r') as file:
        for line in file:
            yield line.strip()
```

```typescript
// TypeScript
async function* readLargeFile(filename: string) {
    const fileHandle = await fs.open(filename, 'r');
    for await (const line of fileHandle.readLines()) {
        yield line.trim();
    }
    await fileHandle.close();
}
```

2. **File Backup**:
```python
# Python
def backup_file(filename):
    backup_name = f"{filename}.backup"
    shutil.copy2(filename, backup_name)
```

```typescript
// TypeScript
async function backupFile(filename: string) {
    const backupName = `${filename}.backup`;
    await fs.copyFile(filename, backupName);
}
```

## Further Reading

- [Python File I/O](https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files)
- [Node.js File System](https://nodejs.org/api/fs.html)
- [Python aiofiles](https://github.com/Tinche/aiofiles)
- [TypeScript File System](https://www.typescriptlang.org/docs/handbook/declaration-files/by-example.html#node-js-modules) 