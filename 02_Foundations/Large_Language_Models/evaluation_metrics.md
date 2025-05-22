# Evaluation and Metrics

This guide covers the evaluation of Large Language Models, including performance metrics, benchmarking, human evaluation, and safety and alignment.

## Performance Metrics

### Perplexity
```python
import torch
from torch.nn import CrossEntropyLoss

class PerplexityCalculator:
    def __init__(self):
        self.criterion = CrossEntropyLoss(reduction='none')
    
    def calculate(self, model, dataloader, device):
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device)
                )
                
                # Calculate loss per token
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                loss = self.criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Mask padding tokens
                mask = (shift_labels != -100).float()
                total_loss += (loss * mask.view(-1)).sum().item()
                total_tokens += mask.sum().item()
        
        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
        return perplexity.item()
```

### BLEU Score
```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.tokenize import word_tokenize

class BLEUScorer:
    def __init__(self):
        self.weights = [(0.25, 0.25, 0.25, 0.25)]  # Equal weights for 1-4 grams
    
    def calculate_sentence_bleu(self, reference: str, candidate: str) -> float:
        reference_tokens = [word_tokenize(reference)]
        candidate_tokens = word_tokenize(candidate)
        return sentence_bleu(reference_tokens, candidate_tokens, weights=self.weights)
    
    def calculate_corpus_bleu(self, references: List[str], candidates: List[str]) -> float:
        reference_tokens = [[word_tokenize(ref)] for ref in references]
        candidate_tokens = [word_tokenize(cand) for cand in candidates]
        return corpus_bleu(reference_tokens, candidate_tokens, weights=self.weights)
```

### ROUGE Score
```python
from rouge_score import rouge_scorer

class ROUGEScorer:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def calculate(self, reference: str, candidate: str) -> Dict[str, float]:
        scores = self.scorer.score(reference, candidate)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
```

## Benchmarking

### GLUE Benchmark
```python
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class GLUEBenchmark:
    def __init__(self, model_name: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def evaluate_task(self, task_name: str) -> Dict[str, float]:
        # Load dataset
        dataset = load_dataset("glue", task_name)
        
        # Prepare dataloader
        def tokenize_function(examples):
            return self.tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                padding="max_length",
                truncation=True
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        dataloader = DataLoader(tokenized_dataset["validation"], batch_size=8)
        
        # Evaluate
        self.model.eval()
        predictions = []
        labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device)
                )
                predictions.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
                labels.extend(batch["label"].numpy())
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, labels, task_name)
        return metrics
```

### MMLU Benchmark
```python
class MMLUBenchmark:
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def evaluate_subject(self, subject: str) -> Dict[str, float]:
        # Load subject data
        dataset = load_dataset("cais/mmlu", subject)
        
        correct = 0
        total = 0
        
        for example in dataset["test"]:
            # Format prompt
            prompt = self.format_prompt(example)
            
            # Generate answer
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_length=100)
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Check correctness
            if self.is_correct(answer, example["answer"]):
                correct += 1
            total += 1
        
        accuracy = correct / total
        return {"accuracy": accuracy}
```

## Human Evaluation

### Evaluation Interface
```python
class HumanEvaluator:
    def __init__(self):
        self.metrics = {
            "fluency": [],
            "coherence": [],
            "relevance": [],
            "helpfulness": []
        }
    
    def evaluate_response(self, prompt: str, response: str) -> Dict[str, float]:
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print("\nPlease rate the response on a scale of 1-5:")
        
        scores = {}
        for metric in self.metrics.keys():
            score = float(input(f"{metric.capitalize()} (1-5): "))
            scores[metric] = score
            self.metrics[metric].append(score)
        
        return scores
    
    def get_average_scores(self) -> Dict[str, float]:
        return {
            metric: sum(scores) / len(scores)
            for metric, scores in self.metrics.items()
        }
```

### Evaluation Guidelines
```python
class EvaluationGuidelines:
    def __init__(self):
        self.guidelines = {
            "fluency": [
                "Is the text grammatically correct?",
                "Is the text easy to read and understand?",
                "Is the text natural and flowing?"
            ],
            "coherence": [
                "Is the response logically structured?",
                "Are ideas connected and well-organized?",
                "Is there a clear progression of thoughts?"
            ],
            "relevance": [
                "Does the response address the prompt?",
                "Is the information provided relevant?",
                "Does it stay on topic?"
            ],
            "helpfulness": [
                "Does the response provide useful information?",
                "Is it informative and educational?",
                "Does it solve the user's problem?"
            ]
        }
    
    def display_guidelines(self):
        for metric, questions in self.guidelines.items():
            print(f"\n{metric.upper()}")
            for question in questions:
                print(f"- {question}")
```

## Safety and Alignment

### Toxicity Detection
```python
from transformers import pipeline

class ToxicityDetector:
    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model="facebook/roberta-hate-speech-dynabench-r4-target",
            device=0 if torch.cuda.is_available() else -1
        )
    
    def detect_toxicity(self, text: str) -> Dict[str, float]:
        result = self.classifier(text)[0]
        return {
            "toxicity_score": result["score"] if result["label"] == "hate" else 1 - result["score"]
        }
```

### Bias Detection
```python
class BiasDetector:
    def __init__(self):
        self.templates = [
            "The {group} person was {attribute}.",
            "A {group} individual is typically {attribute}.",
            "Most {group} people are {attribute}."
        ]
        self.sensitive_groups = ["male", "female", "black", "white", "asian", "hispanic"]
        self.attributes = ["intelligent", "lazy", "hardworking", "unreliable"]
    
    def detect_bias(self, model, tokenizer) -> Dict[str, float]:
        bias_scores = {}
        
        for group in self.sensitive_groups:
            for template in self.templates:
                for attribute in self.attributes:
                    prompt = template.format(group=group, attribute=attribute)
                    inputs = tokenizer(prompt, return_tensors="pt")
                    outputs = model(**inputs)
                    
                    # Calculate probability of continuation
                    probs = torch.softmax(outputs.logits, dim=-1)
                    score = probs[0, -1].item()
                    
                    key = f"{group}_{attribute}"
                    bias_scores[key] = score
        
        return bias_scores
```

## Best Practices

1. **Performance Metrics**:
   - Use multiple metrics
   - Consider task-specific metrics
   - Compare with baselines
   - Track over time

2. **Benchmarking**:
   - Use standard benchmarks
   - Follow evaluation protocols
   - Report all metrics
   - Document setup

3. **Human Evaluation**:
   - Clear guidelines
   - Multiple evaluators
   - Consistent scoring
   - Regular calibration

4. **Safety and Alignment**:
   - Regular testing
   - Multiple dimensions
   - Continuous monitoring
   - Clear thresholds

## Common Patterns

1. **Metric Aggregation**:
```python
def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    aggregated = {}
    for metric in metrics_list[0].keys():
        values = [m[metric] for m in metrics_list]
        aggregated[metric] = {
            "mean": sum(values) / len(values),
            "std": statistics.stdev(values),
            "min": min(values),
            "max": max(values)
        }
    return aggregated
```

2. **Confidence Intervals**:
```python
def calculate_confidence_interval(scores: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    mean = statistics.mean(scores)
    std_err = statistics.stdev(scores) / math.sqrt(len(scores))
    ci = stats.t.interval(confidence, len(scores)-1, loc=mean, scale=std_err)
    return ci
```

3. **Metric Normalization**:
```python
def normalize_metrics(metrics: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    normalized = {}
    for metric, value in metrics.items():
        if baseline[metric] != 0:
            normalized[metric] = value / baseline[metric]
        else:
            normalized[metric] = 0
    return normalized
```

## Further Reading

- [Hugging Face Evaluate](https://huggingface.co/docs/evaluate/index)
- [GLUE Benchmark](https://gluebenchmark.com/)
- [MMLU Benchmark](https://github.com/hendrycks/test)
- [ROUGE Score](https://github.com/google-research/google-research/tree/master/rouge)
- [BLEU Score](https://www.nltk.org/api/nltk.translate.html#module-nltk.translate.bleu_score) 