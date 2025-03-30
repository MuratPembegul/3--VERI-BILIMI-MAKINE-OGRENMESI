import numpy as np

# Ortamı tanımla (4x4 ızgara)
env_size = 4
q_table = np.zeros((env_size, env_size))

# Parametreler
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1  # Keşfetme olasılığı

# Rastgele bir ortamda ajanı eğit
for episode in range(1000):
    state = np.random.randint(0, env_size)  # Başlangıç durumu
    
    for _ in range(100):
        action = np.random.choice([0, 1]) if np.random.rand() < epsilon else np.argmax(q_table[state])
        reward = 1 if state == env_size - 1 else -1  # Son duruma ulaşınca ödül
        new_state = (state + action) % env_size
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + discount_factor * np.max(q_table[new_state]) - q_table[state, action]
        )
        state = new_state

print("Öğrenilen Q-Tablosu:")
print(q_table)
