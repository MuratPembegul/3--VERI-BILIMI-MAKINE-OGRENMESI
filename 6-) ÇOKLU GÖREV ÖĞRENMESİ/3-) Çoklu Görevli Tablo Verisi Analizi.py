from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

# Giriş katmanı (örneğin 10 özellik)
input_data = Input(shape=(10,))

# Paylaşılan katmanlar
x = Dense(32, activation="relu")(input_data)
x = Dense(64, activation="relu")(x)

# Kredi puanı tahmini (regresyon)
credit_score_output = Dense(1, activation="linear", name="credit_score_output")(x)

# Dolandırıcılık tahmini (sınıflandırma)
fraud_output = Dense(1, activation="sigmoid", name="fraud_output")(x)

# Modeli oluştur
model = Model(inputs=input_data, outputs=[credit_score_output, fraud_output])

# Modeli derle
model.compile(optimizer="adam",
              loss={"credit_score_output": "mse", "fraud_output": "binary_crossentropy"},
              metrics={"credit_score_output": "mae", "fraud_output": "accuracy"})

model.summary()
