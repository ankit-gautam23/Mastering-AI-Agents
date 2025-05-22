# Advanced Prompt Engineering Techniques

This guide covers advanced techniques for crafting more effective prompts, including Chain-of-Thought, Few-Shot Learning, Self-Consistency, and Tree of Thoughts.

## Chain-of-Thought (CoT)

### Basic CoT Implementation
```python
from typing import List, Dict, Any

class ChainOfThought:
    def __init__(self):
        self.steps = []
        self.final_answer = ""
    
    def add_step(self, step: str):
        self.steps.append(step)
    
    def set_final_answer(self, answer: str):
        self.final_answer = answer
    
    def format(self) -> str:
        prompt = "Let's solve this step by step:\n\n"
        for i, step in enumerate(self.steps, 1):
            prompt += f"Step {i}: {step}\n"
        prompt += f"\nTherefore, the answer is: {self.final_answer}"
        return prompt
```

### CoT with Reasoning
```python
class CoTWithReasoning:
    def __init__(self):
        self.question = ""
        self.reasoning_steps = []
        self.answer = ""
    
    def set_question(self, question: str):
        self.question = question
    
    def add_reasoning_step(self, step: str, explanation: str):
        self.reasoning_steps.append({
            "step": step,
            "explanation": explanation
        })
    
    def set_answer(self, answer: str):
        self.answer = answer
    
    def format(self) -> str:
        prompt = f"Question: {self.question}\n\n"
        prompt += "Let's think through this carefully:\n\n"
        
        for i, step in enumerate(self.reasoning_steps, 1):
            prompt += f"Step {i}: {step['step']}\n"
            prompt += f"Reasoning: {step['explanation']}\n\n"
        
        prompt += f"Final Answer: {self.answer}"
        return prompt
```

## Few-Shot Learning

### Basic Few-Shot Template
```python
class FewShotTemplate:
    def __init__(self):
        self.examples = []
        self.query = ""
    
    def add_example(self, input_text: str, output_text: str):
        self.examples.append({
            "input": input_text,
            "output": output_text
        })
    
    def set_query(self, query: str):
        self.query = query
    
    def format(self) -> str:
        prompt = "Here are some examples:\n\n"
        
        for example in self.examples:
            prompt += f"Input: {example['input']}\n"
            prompt += f"Output: {example['output']}\n\n"
        
        prompt += f"Now, for the following input:\n{self.query}\n"
        prompt += "Output:"
        return prompt
```

### Dynamic Few-Shot Selection
```python
class DynamicFewShot:
    def __init__(self):
        self.example_pool = []
        self.max_examples = 3
    
    def add_to_pool(self, example: Dict[str, str], similarity_score: float):
        self.example_pool.append({
            "example": example,
            "score": similarity_score
        })
    
    def select_examples(self, query: str) -> List[Dict[str, str]]:
        # Sort examples by similarity score
        sorted_examples = sorted(
            self.example_pool,
            key=lambda x: x["score"],
            reverse=True
        )
        
        # Select top examples
        return [ex["example"] for ex in sorted_examples[:self.max_examples]]
```

## Self-Consistency

### Self-Consistency Implementation
```python
class SelfConsistency:
    def __init__(self, num_samples: int = 5):
        self.num_samples = num_samples
        self.samples = []
    
    def add_sample(self, reasoning: str, answer: str):
        self.samples.append({
            "reasoning": reasoning,
            "answer": answer
        })
    
    def get_majority_answer(self) -> str:
        # Count occurrences of each answer
        answer_counts = {}
        for sample in self.samples:
            answer = sample["answer"]
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        # Return most common answer
        return max(answer_counts.items(), key=lambda x: x[1])[0]
    
    def format(self) -> str:
        prompt = "Let's solve this problem multiple times to ensure consistency:\n\n"
        
        for i, sample in enumerate(self.samples, 1):
            prompt += f"Attempt {i}:\n"
            prompt += f"Reasoning: {sample['reasoning']}\n"
            prompt += f"Answer: {sample['answer']}\n\n"
        
        prompt += f"Final Answer (Majority): {self.get_majority_answer()}"
        return prompt
```

## Tree of Thoughts

### Tree of Thoughts Implementation
```python
class TreeOfThoughts:
    def __init__(self):
        self.root = None
        self.current_path = []
    
    class Node:
        def __init__(self, thought: str):
            self.thought = thought
            self.children = []
            self.score = 0.0
    
    def add_thought(self, thought: str, parent: Node = None):
        node = self.Node(thought)
        if parent:
            parent.children.append(node)
        else:
            self.root = node
        return node
    
    def evaluate_path(self, path: List[Node]) -> float:
        # Implement path evaluation logic
        return sum(node.score for node in path)
    
    def find_best_path(self) -> List[Node]:
        if not self.root:
            return []
        
        best_path = []
        best_score = float('-inf')
        
        def dfs(node: Node, current_path: List[Node]):
            nonlocal best_path, best_score
            
            current_path.append(node)
            
            if not node.children:
                score = self.evaluate_path(current_path)
                if score > best_score:
                    best_score = score
                    best_path = current_path.copy()
            else:
                for child in node.children:
                    dfs(child, current_path)
            
            current_path.pop()
        
        dfs(self.root, [])
        return best_path
    
    def format(self) -> str:
        if not self.root:
            return "No thoughts generated yet."
        
        best_path = self.find_best_path()
        prompt = "Let's explore different lines of reasoning:\n\n"
        
        for i, node in enumerate(best_path, 1):
            prompt += f"Thought {i}: {node.thought}\n"
        
        prompt += f"\nFinal Answer: {best_path[-1].thought}"
        return prompt
```

## Best Practices

1. **Chain-of-Thought**:
   - Break down complex problems
   - Show intermediate steps
   - Explain reasoning
   - Verify each step

2. **Few-Shot Learning**:
   - Choose relevant examples
   - Maintain consistency
   - Use diverse cases
   - Keep examples concise

3. **Self-Consistency**:
   - Generate multiple solutions
   - Compare approaches
   - Use majority voting
   - Validate results

4. **Tree of Thoughts**:
   - Explore alternatives
   - Evaluate paths
   - Prune invalid branches
   - Track best solutions

## Common Patterns

1. **Multi-Step Reasoning**:
```python
def create_multi_step_prompt(question: str, steps: List[str]) -> str:
    prompt = f"Question: {question}\n\n"
    prompt += "Let's solve this step by step:\n\n"
    
    for i, step in enumerate(steps, 1):
        prompt += f"Step {i}: {step}\n"
    
    prompt += "\nBased on these steps, the answer is:"
    return prompt
```

2. **Example-Based Learning**:
```python
def create_example_based_prompt(examples: List[Dict[str, str]], query: str) -> str:
    prompt = "Here are some examples of similar problems and their solutions:\n\n"
    
    for example in examples:
        prompt += f"Problem: {example['problem']}\n"
        prompt += f"Solution: {example['solution']}\n\n"
    
    prompt += f"Now, solve this problem:\n{query}\n"
    prompt += "Solution:"
    return prompt
```

3. **Consistency Checking**:
```python
def create_consistency_check_prompt(question: str, attempts: int) -> str:
    prompt = f"Question: {question}\n\n"
    prompt += f"Let's solve this {attempts} times to ensure consistency:\n\n"
    
    for i in range(attempts):
        prompt += f"Attempt {i+1}:\n"
        prompt += "Reasoning: [Your reasoning here]\n"
        prompt += "Answer: [Your answer here]\n\n"
    
    prompt += "Final Answer (Majority):"
    return prompt
```

## Further Reading

- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [Few-Shot Learning with Language Models](https://arxiv.org/abs/2005.14165)
- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171)
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)
- [Advanced Prompt Engineering Techniques](https://www.promptingguide.ai/techniques) 