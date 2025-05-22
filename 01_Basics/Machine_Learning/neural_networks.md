# Neural Networks

This guide covers the fundamentals of neural networks, their architecture, and implementation using Python and popular deep learning frameworks.

## Basic Concepts

### Neuron Structure
```python
import numpy as np

class Neuron:
    def __init__(self, n_inputs):
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.bias
        return self.activation(self.output)
    
    def activation(self, x):
        # ReLU activation
        return max(0, x)
```

### Simple Neural Network
```python
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.layer1 = Neuron(input_size)
        self.layer2 = Neuron(hidden_size)
        self.layer3 = Neuron(output_size)
    
    def forward(self, x):
        h1 = self.layer1.forward(x)
        h2 = self.layer2.forward(h1)
        return self.layer3.forward(h2)
```

## Using TensorFlow/Keras

### Basic Neural Network
```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Create model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
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

### Convolutional Neural Network (CNN)
```python
# Create CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

### Recurrent Neural Network (RNN)
```python
# Create RNN model
model = models.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(None, 28)),
    layers.LSTM(32),
    layers.Dense(10, activation='softmax')
])
```

## Network Architectures

### Feedforward Neural Network
```python
class FeedforwardNN:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(layers.Dense(
                layer_sizes[i + 1],
                activation='relu' if i < len(layer_sizes) - 2 else 'softmax',
                input_shape=(layer_sizes[i],)
            ))
    
    def build(self):
        model = models.Sequential(self.layers)
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model
```

### Autoencoder
```python
# Create autoencoder
encoder = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu')
])

decoder = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(32,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(784, activation='sigmoid')
])

autoencoder = models.Sequential([encoder, decoder])
```

## Training and Optimization

### Custom Training Loop
```python
@tf.function
def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

### Learning Rate Scheduling
```python
initial_learning_rate = 0.1
decay_steps = 1000
decay_rate = 0.9

learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps,
    decay_rate
)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
```

## Regularization Techniques

### Dropout
```python
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])
```

### Batch Normalization
```python
model = models.Sequential([
    layers.Dense(64, input_shape=(784,)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(32),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(10, activation='softmax')
])
```

## Model Evaluation

### Cross-Validation
```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True)
for train_idx, val_idx in kfold.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=10)
```

### Model Evaluation
```python
# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Make predictions
predictions = model.predict(X_test)
```

## Best Practices

1. **Data Preparation**:
   - Normalize input data
   - Use data augmentation
   - Handle class imbalance
   - Split data appropriately

2. **Model Architecture**:
   - Start with simple models
   - Gradually increase complexity
   - Use appropriate activation functions
   - Implement proper initialization

3. **Training**:
   - Use appropriate batch size
   - Implement early stopping
   - Monitor training progress
   - Use learning rate scheduling

## Common Applications

1. **Image Classification**:
   - Object recognition
   - Face detection
   - Medical imaging
   - Quality control

2. **Natural Language Processing**:
   - Text classification
   - Sentiment analysis
   - Machine translation
   - Question answering

3. **Time Series Analysis**:
   - Stock prediction
   - Weather forecasting
   - Anomaly detection
   - Speech recognition

## Further Reading

- [TensorFlow Documentation](https://www.tensorflow.org/guide)
- [Keras Documentation](https://keras.io/guides/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) 