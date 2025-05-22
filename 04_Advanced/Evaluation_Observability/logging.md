# Logging for AI Agents

This guide covers the fundamental concepts and implementations of logging systems for AI agents, including structured logging, log levels, and best practices.

## Basic Logging Implementation

### Structured Logging
```python
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class LogEntry:
    timestamp: datetime
    level: str
    message: str
    agent_id: str
    metadata: Dict[str, Any]

class StructuredLogger:
    def __init__(self, agent_id: str, log_file: str):
        self.agent_id = agent_id
        self.logger = logging.getLogger(agent_id)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(message)s')
        )
        self.logger.addHandler(file_handler)
    
    def log(self, level: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            agent_id=self.agent_id,
            metadata=metadata or {}
        )
        
        self.logger.log(
            getattr(logging, level.upper()),
            json.dumps(asdict(entry), default=str)
        )
```

### Log Levels and Handlers
```python
class LogManager:
    def __init__(self, agent_id: str, log_dir: str):
        self.agent_id = agent_id
        self.logger = logging.getLogger(agent_id)
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler for INFO and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter('%(levelname)s: %(message)s')
        )
        self.logger.addHandler(console_handler)
        
        # File handler for DEBUG and above
        debug_handler = logging.FileHandler(f"{log_dir}/debug.log")
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(debug_handler)
        
        # Error handler for ERROR and above
        error_handler = logging.FileHandler(f"{log_dir}/error.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(error_handler)
```

## Advanced Logging Features

### Log Rotation
```python
from logging.handlers import RotatingFileHandler
import os

class RotatingLogger:
    def __init__(self, agent_id: str, log_dir: str, max_bytes: int = 1024*1024, backup_count: int = 5):
        self.agent_id = agent_id
        self.logger = logging.getLogger(agent_id)
        self.logger.setLevel(logging.INFO)
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Rotating file handler
        rotating_handler = RotatingFileHandler(
            f"{log_dir}/agent.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        rotating_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(rotating_handler)
```

### Log Aggregation
```python
class LogAggregator:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.log_files = {}
    
    def add_log_file(self, agent_id: str, log_file: str) -> None:
        self.log_files[agent_id] = log_file
    
    def get_agent_logs(self, agent_id: str, level: str = "INFO") -> List[Dict]:
        if agent_id not in self.log_files:
            return []
        
        logs = []
        with open(self.log_files[agent_id], 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    if log_entry["level"] == level:
                        logs.append(log_entry)
                except json.JSONDecodeError:
                    continue
        
        return logs
    
    def get_all_logs(self, level: str = "INFO") -> Dict[str, List[Dict]]:
        return {
            agent_id: self.get_agent_logs(agent_id, level)
            for agent_id in self.log_files
        }
```

## Log Analysis

### Log Parser
```python
class LogParser:
    def __init__(self):
        self.patterns = {
            "error": r"ERROR:.*",
            "warning": r"WARNING:.*",
            "info": r"INFO:.*"
        }
    
    def parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None
    
    def extract_patterns(self, log_entry: Dict[str, Any]) -> List[str]:
        patterns_found = []
        for pattern_name, pattern in self.patterns.items():
            if re.search(pattern, log_entry["message"]):
                patterns_found.append(pattern_name)
        return patterns_found
```

### Log Analyzer
```python
class LogAnalyzer:
    def __init__(self):
        self.parser = LogParser()
        self.log_entries = []
    
    def add_log_entry(self, log_entry: Dict[str, Any]) -> None:
        self.log_entries.append(log_entry)
    
    def analyze_logs(self) -> Dict[str, Any]:
        if not self.log_entries:
            return {}
        
        return {
            "total_entries": len(self.log_entries),
            "error_count": sum(1 for e in self.log_entries if e["level"] == "ERROR"),
            "warning_count": sum(1 for e in self.log_entries if e["level"] == "WARNING"),
            "info_count": sum(1 for e in self.log_entries if e["level"] == "INFO"),
            "patterns": self._analyze_patterns()
        }
    
    def _analyze_patterns(self) -> Dict[str, int]:
        pattern_counts = {}
        for entry in self.log_entries:
            patterns = self.parser.extract_patterns(entry)
            for pattern in patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        return pattern_counts
```

## Best Practices

1. **Log Structure**:
   - Use consistent format
   - Include timestamps
   - Add context information

2. **Log Levels**:
   - DEBUG: Detailed information
   - INFO: General information
   - WARNING: Potential issues
   - ERROR: Serious problems
   - CRITICAL: System failures

3. **Log Management**:
   - Implement log rotation
   - Set size limits
   - Archive old logs

4. **Security**:
   - Sanitize sensitive data
   - Control access
   - Encrypt logs

## Common Patterns

1. **Logging Factory**:
```python
class LoggingFactory:
    @staticmethod
    def create_logger(logger_type: str, **kwargs) -> logging.Logger:
        if logger_type == "structured":
            return StructuredLogger(**kwargs)
        elif logger_type == "rotating":
            return RotatingLogger(**kwargs)
        elif logger_type == "manager":
            return LogManager(**kwargs)
        else:
            raise ValueError(f"Unknown logger type: {logger_type}")
```

2. **Logging Pipeline**:
```python
class LoggingPipeline:
    def __init__(self, agent_id: str, log_dir: str):
        self.logger = LoggingFactory.create_logger("structured", agent_id=agent_id, log_dir=log_dir)
        self.analyzer = LogAnalyzer()
        self.aggregator = LogAggregator(log_dir)
    
    def process_log(self, level: str, message: str, metadata: Optional[Dict] = None) -> None:
        # Log the message
        self.logger.log(level, message, metadata)
        
        # Analyze the log
        log_entry = {
            "timestamp": datetime.now(),
            "level": level,
            "message": message,
            "metadata": metadata or {}
        }
        self.analyzer.add_log_entry(log_entry)
    
    def get_analysis(self) -> Dict:
        return self.analyzer.analyze_logs()
```

## Further Reading

- [Structured Logging Best Practices](https://arxiv.org/abs/2004.07213)
- [Log Analysis Techniques](https://arxiv.org/abs/2004.07213)
- [Log Management Systems](https://arxiv.org/abs/2004.07213)
- [Security in Logging](https://arxiv.org/abs/2004.07213)
- [Logging Patterns](https://arxiv.org/abs/2004.07213) 