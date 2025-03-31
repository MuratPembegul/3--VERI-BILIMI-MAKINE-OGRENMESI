import tensorflow as tf
from tensorflow.keras import layers, Model

# Giriş katmanı (örneğin 64x64x3 boyutunda bir görüntü)
inputs = layers.Input(shape=(64, 64, 3))

# Paylaşılan CNN katmanları
x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Flatten()(x)

# Nesne tanıma için sınıflandırma başlığı
class_output = layers.Dense(10, activation='softmax', name="class_output")(x)

# Nesnenin konum tahmini için regresyon başlığı
bbox_output = layers.Dense(4, activation='linear', name="bbox_output")(x)

# Modeli oluştur
model = Model(inputs=inputs, outputs=[class_output, bbox_output])

# Modeli derle
model.compile(optimizer='adam',
              loss={"class_output": "sparse_categorical_crossentropy", "bbox_output": "mse"},
              metrics={"class_output": "accuracy", "bbox_output": "mae"})

model.summary()
