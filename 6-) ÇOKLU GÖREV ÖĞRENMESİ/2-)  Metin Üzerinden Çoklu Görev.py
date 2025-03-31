from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model

# Giriş katmanı (Maksimum 100 kelimelik bir metin)
input_text = Input(shape=(100,))

# Embedding + LSTM
x = Embedding(input_dim=5000, output_dim=128, input_length=100)(input_text)
x = LSTM(64)(x)

# Duygu analizi için çıktı
sentiment_output = Dense(3, activation='softmax', name="sentiment_output")(x)  # (Negatif, Nötr, Pozitif)

# Konu tahmini için çıktı
topic_output = Dense(5, activation='softmax', name="topic_output")(x)  # 5 farklı konu

# Modeli oluştur
model = Model(inputs=input_text, outputs=[sentiment_output, topic_output])

# Modeli derle
model.compile(optimizer='adam',
              loss={"sentiment_output": "sparse_categorical_crossentropy", "topic_output": "sparse_categorical_crossentropy"},
              metrics=["accuracy"])

model.summary()
