# File I/O Operations

This guide covers file input/output operations for various file types commonly used in AI and LLM applications.

## Document Processing

### Text File Processing
```python
from typing import List, Generator
import json

class TextProcessor:
    def __init__(self, encoding: str = 'utf-8'):
        self.encoding = encoding
    
    def read_text(self, file_path: str) -> str:
        with open(file_path, 'r', encoding=self.encoding) as file:
            return file.read()
    
    def write_text(self, file_path: str, content: str):
        with open(file_path, 'w', encoding=self.encoding) as file:
            file.write(content)
    
    def read_lines(self, file_path: str) -> Generator[str, None, None]:
        with open(file_path, 'r', encoding=self.encoding) as file:
            for line in file:
                yield line.strip()
    
    def write_lines(self, file_path: str, lines: List[str]):
        with open(file_path, 'w', encoding=self.encoding) as file:
            file.write('\n'.join(lines))
```

### JSON File Processing
```python
from typing import Any, Dict, List
import json

class JSONProcessor:
    def read_json(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    
    def write_json(self, file_path: str, data: Dict[str, Any], indent: int = 4):
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=indent)
    
    def read_jsonl(self, file_path: str) -> Generator[Dict[str, Any], None, None]:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                yield json.loads(line.strip())
    
    def write_jsonl(self, file_path: str, data: List[Dict[str, Any]]):
        with open(file_path, 'w', encoding='utf-8') as file:
            for item in data:
                file.write(json.dumps(item) + '\n')
```

## Image Processing

### Image File Processing
```python
from PIL import Image
from typing import Tuple, Optional
import io

class ImageProcessor:
    def read_image(self, file_path: str) -> Image.Image:
        return Image.open(file_path)
    
    def write_image(self, image: Image.Image, file_path: str, format: Optional[str] = None):
        image.save(file_path, format=format)
    
    def resize_image(self, image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        return image.resize(size, Image.Resampling.LANCZOS)
    
    def convert_to_bytes(self, image: Image.Image, format: str = 'PNG') -> bytes:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=format)
        return img_byte_arr.getvalue()
    
    def read_from_bytes(self, image_bytes: bytes) -> Image.Image:
        return Image.open(io.BytesIO(image_bytes))
```

## Audio Processing

### Audio File Processing
```python
import librosa
import soundfile as sf
from typing import Tuple, Optional
import numpy as np

class AudioProcessor:
    def read_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        return librosa.load(file_path)
    
    def write_audio(self, file_path: str, audio: np.ndarray, sample_rate: int):
        sf.write(file_path, audio, sample_rate)
    
    def get_duration(self, file_path: str) -> float:
        return librosa.get_duration(path=file_path)
    
    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
```

## Video Processing

### Video File Processing
```python
import cv2
from typing import Generator, Tuple, Optional
import numpy as np

class VideoProcessor:
    def __init__(self):
        self.cap = None
    
    def open_video(self, file_path: str):
        self.cap = cv2.VideoCapture(file_path)
    
    def close_video(self):
        if self.cap:
            self.cap.release()
    
    def read_frames(self) -> Generator[np.ndarray, None, None]:
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame
    
    def write_video(
        self,
        file_path: str,
        frames: Generator[np.ndarray, None, None],
        fps: int = 30,
        size: Optional[Tuple[int, int]] = None
    ):
        if not size:
            first_frame = next(frames)
            size = (first_frame.shape[1], first_frame.shape[0])
            frames = (first_frame, *frames)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(file_path, fourcc, fps, size)
        
        for frame in frames:
            out.write(frame)
        
        out.release()
```

## Best Practices

1. **File Handling**:
   - Use context managers (with statements)
   - Handle file encoding properly
   - Implement proper error handling
   - Close files after use

2. **Performance**:
   - Use generators for large files
   - Implement buffering
   - Use appropriate chunk sizes
   - Consider memory usage

3. **Security**:
   - Validate file paths
   - Check file permissions
   - Sanitize file names
   - Handle sensitive data

## Common Patterns

1. **File Chunking**:
```python
def read_in_chunks(file_path: str, chunk_size: int = 8192):
    with open(file_path, 'rb') as file:
        while chunk := file.read(chunk_size):
            yield chunk
```

2. **File Backup**:
```python
import shutil
from datetime import datetime

def backup_file(file_path: str):
    backup_path = f"{file_path}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
    shutil.copy2(file_path, backup_path)
```

3. **File Validation**:
```python
import os
from typing import List

def validate_file(file_path: str, allowed_extensions: List[str]) -> bool:
    if not os.path.exists(file_path):
        return False
    
    extension = os.path.splitext(file_path)[1].lower()
    return extension in allowed_extensions
```

## Further Reading

- [Python File I/O](https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files)
- [Pillow Documentation](https://pillow.readthedocs.io/)
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Python JSON](https://docs.python.org/3/library/json.html) 