# Reinforcement Learning

This guide covers the fundamentals of reinforcement learning, including key algorithms and their implementation in Python.

## Basic Concepts

### Environment and Agent
```python
import gym
import numpy as np

class SimpleEnvironment:
    def __init__(self):
        self.state = 0
        self.action_space = [0, 1]  # Left, Right
        self.observation_space = range(5)  # States 0-4
    
    def reset(self):
        self.state = 0
        return self.state
    
    def step(self, action):
        if action == 0:  # Left
            self.state = max(0, self.state - 1)
        else:  # Right
            self.state = min(4, self.state + 1)
        
        reward = 1 if self.state == 4 else 0
        done = self.state == 4
        return self.state, reward, done, {}
```

## Q-Learning

### Basic Q-Learning Implementation
```python
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

### Training Loop
```python
def train_agent(env, agent, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
```

## Deep Q-Learning (DQN)

### DQN Implementation
```python
import tensorflow as tf
from tensorflow.keras import layers, models
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
    
    def _build_model(self):
        model = models.Sequential([
            layers.Dense(24, input_dim=self.state_size, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

## Policy Gradient Methods

### REINFORCE Algorithm
```python
class REINFORCE:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
    
    def _build_model(self):
        model = models.Sequential([
            layers.Dense(24, input_dim=self.state_size, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy',
                     optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model
    
    def get_action(self, state):
        policy = self.model.predict(state)[0]
        return np.random.choice(self.action_size, p=policy)
    
    def update(self, states, actions, rewards):
        discounted_rewards = self._discount_rewards(rewards)
        states = np.vstack(states)
        actions = np.array(actions)
        
        # Convert actions to one-hot encoding
        actions_one_hot = np.zeros((len(actions), self.action_size))
        actions_one_hot[np.arange(len(actions)), actions] = 1
        
        # Update policy
        self.model.fit(states, actions_one_hot, sample_weight=discounted_rewards, verbose=0)
    
    def _discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards
```

## Actor-Critic Methods

### Actor-Critic Implementation
```python
class ActorCritic:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 0.001
        
        # Actor Network
        self.actor = models.Sequential([
            layers.Dense(24, input_dim=state_size, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(action_size, activation='softmax')
        ])
        
        # Critic Network
        self.critic = models.Sequential([
            layers.Dense(24, input_dim=state_size, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        self.actor.compile(loss='categorical_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        self.critic.compile(loss='mse',
                           optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
    
    def get_action(self, state):
        policy = self.actor.predict(state)[0]
        return np.random.choice(self.action_size, p=policy)
    
    def update(self, state, action, reward, next_state, done):
        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]
        
        # Calculate advantage
        advantage = reward + self.gamma * next_value * (1 - done) - value
        
        # Update actor
        action_one_hot = np.zeros((1, self.action_size))
        action_one_hot[0][action] = 1
        self.actor.fit(state, action_one_hot, sample_weight=advantage, verbose=0)
        
        # Update critic
        target = reward + self.gamma * next_value * (1 - done)
        self.critic.fit(state, target, verbose=0)
```

## Best Practices

1. **Environment Design**:
   - Clear state representation
   - Meaningful reward function
   - Appropriate action space
   - Proper termination conditions

2. **Algorithm Selection**:
   - Q-learning for discrete actions
   - DQN for complex state spaces
   - Policy gradients for continuous actions
   - Actor-Critic for stability

3. **Training**:
   - Appropriate exploration strategy
   - Proper learning rate
   - Sufficient training episodes
   - Regular evaluation

## Common Applications

1. **Game Playing**:
   - Atari games
   - Chess
   - Go
   - Poker

2. **Robotics**:
   - Navigation
   - Manipulation
   - Locomotion
   - Control

3. **Autonomous Systems**:
   - Self-driving cars
   - Drone control
   - Resource management
   - Trading systems

## Further Reading

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- [OpenAI Gym Documentation](https://gym.openai.com/)
- [TensorFlow Agents](https://www.tensorflow.org/agents)
- [Stable Baselines](https://stable-baselines.readthedocs.io/) 