import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist

# Veri setini yükle (El yazısı rakamlar)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Etiketlerin bir kısmını kaybet (Yarı Denetimli Öğrenme için)
y_train_missing = np.copy(y_train)
y_train_missing[np.random.choice(len(y_train), size=int(0.9 * len(y_train)), replace=False)] = -1  # %90 etiketsiz

# Modeli oluştur
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Modeli derle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğit (etiketli verilerle)
labeled_mask = y_train_missing != -1
model.fit(x_train[labeled_mask], y_train[labeled_mask], epochs=5, validation_data=(x_test, y_test))

# Modeli etiketsiz verilerle tekrar eğit (pseudo-labeling)
pseudo_labels = np.argmax(model.predict(x_train[~labeled_mask]), axis=1)
model.fit(x_train[~labeled_mask], pseudo_labels, epochs=5, validation_data=(x_test, y_test))

# Test doğruluğunu hesapla
accuracy = model.evaluate(x_test, y_test)[1]
print(f"Son Doğruluk: {accuracy:.2f}")
