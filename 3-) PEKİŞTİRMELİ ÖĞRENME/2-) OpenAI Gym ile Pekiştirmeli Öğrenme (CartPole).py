import gym
import numpy as np

# Ortamı başlat
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Q-Tablosu
q_table = np.zeros((state_size, action_size))

# Parametreler
learning_rate = 0.1
discount_factor = 0.95
epsilon = 1.0  
epsilon_decay = 0.995

# Eğitimi başlat
for episode in range(1000):
    state, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(q_table[state])
        next_state, reward, done, _, _ = env.step(action)
        q_table[state, action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state]) - q_table[state, action])
        state = next_state
    epsilon *= epsilon_decay  # Keşfetme oranını azalt

print("Eğitim tamamlandı!")
