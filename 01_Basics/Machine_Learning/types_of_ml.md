# Types of Machine Learning

This guide covers the fundamental types of machine learning, their applications, and implementation examples in Python.

## Supervised Learning

Supervised learning involves training a model on labeled data, where the correct output is known.

### Classification
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)
```

### Regression
```python
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

# Load dataset
boston = load_boston()
X, y = boston.data, boston.target

# Train model
reg = LinearRegression()
reg.fit(X, y)

# Make predictions
predictions = reg.predict(X)
```

## Unsupervised Learning

Unsupervised learning involves training a model on unlabeled data, where the correct output is unknown.

### Clustering
```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60)

# Train model
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# Get cluster assignments
labels = kmeans.labels_
```

### Dimensionality Reduction
```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X = iris.data

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```

## Reinforcement Learning

Reinforcement learning involves training an agent to make decisions by rewarding desired behaviors.

### Q-Learning Example
```python
import numpy as np

class QLearning:
    def __init__(self, states, actions, learning_rate=0.1, discount_factor=0.95):
        self.q_table = np.zeros((states, actions))
        self.lr = learning_rate
        self.gamma = discount_factor

    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

    def get_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state])
```

## Semi-Supervised Learning

Semi-supervised learning uses both labeled and unlabeled data for training.

```python
from sklearn.semi_supervised import LabelSpreading
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15)

# Create semi-supervised dataset
n_labeled = 50
y_semi = np.copy(y)
y_semi[n_labeled:] = -1  # Unlabeled samples

# Train model
label_spreading = LabelSpreading(kernel='knn', alpha=0.8)
label_spreading.fit(X, y_semi)

# Get predictions
predictions = label_spreading.predict(X)
```

## Deep Learning

Deep learning uses neural networks with multiple layers to learn complex patterns.

### Neural Network Example
```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Create model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

## Transfer Learning

Transfer learning involves using knowledge gained from one task to improve learning in another task.

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

# Load pre-trained model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
base_model.trainable = False

# Add custom layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

## Best Practices

1. **Data Preparation**:
   - Clean and preprocess data
   - Handle missing values
   - Normalize/standardize features
   - Split data appropriately

2. **Model Selection**:
   - Choose appropriate algorithm
   - Consider data size and type
   - Evaluate computational requirements
   - Consider interpretability needs

3. **Evaluation**:
   - Use appropriate metrics
   - Implement cross-validation
   - Monitor for overfitting
   - Validate on test set

## Common Applications

1. **Classification**:
   - Image classification
   - Spam detection
   - Sentiment analysis
   - Disease diagnosis

2. **Regression**:
   - Price prediction
   - Weather forecasting
   - Sales forecasting
   - Risk assessment

3. **Clustering**:
   - Customer segmentation
   - Document clustering
   - Image segmentation
   - Anomaly detection

## Further Reading

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- [Deep Learning Book](https://www.deeplearningbook.org/) 