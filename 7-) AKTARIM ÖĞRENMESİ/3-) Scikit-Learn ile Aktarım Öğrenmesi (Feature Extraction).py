from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.applications import ResNet50
import numpy as np

# ResNet50 modelini yükleyelim (çıktı özelliklerini almak için)
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Varsayımsal bir resim kümesi (örnek)
X_train = np.random.rand(100, 224, 224, 3)  # 100 resim
X_train_features = base_model.predict(X_train)  # Özellikleri çıkaralım

# RandomForest ile sınıflandırma
clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_train = np.random.randint(0, 10, size=(100,))  # 10 sınıf
clf.fit(X_train_features, y_train)
