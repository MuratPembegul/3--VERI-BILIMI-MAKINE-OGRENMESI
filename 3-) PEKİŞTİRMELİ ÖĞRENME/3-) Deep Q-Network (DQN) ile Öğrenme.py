import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random

# Ortamı başlat
env = gym.make("CartPole-v1")

# Model oluştur
model = Sequential([
    Dense(24, activation='relu', input_shape=(env.observation_space.shape[0],)),
    Dense(24, activation='relu'),
    Dense(env.action_space.n, activation='linear')
])
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

# Bellek (deney havuzu)
memory = []
gamma = 0.95  # Gelecek ödüllerin önemi
epsilon = 1.0  
epsilon_min = 0.01  
epsilon_decay = 0.995  
batch_size = 32  

# Ajanı eğit
for episode in range(1000):
    state, _ = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    
    for time in range(500):
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(state))
        
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
        memory.append((state, action, reward, next_state, done))
        state = next_state
        
        if done:
            print(f"Episode {episode}: Score {time}")
            break
    
    if len(memory) > batch_size:
        minibatch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + gamma * np.max(model.predict(next_state))
            target_f = model.predict(state)
            target_f[0][action] = target
            model.fit(state, target_f, epochs=1, verbose=0)
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay  # Keşfetme oranını azalt

print("Eğitim tamamlandı! 🚀")
