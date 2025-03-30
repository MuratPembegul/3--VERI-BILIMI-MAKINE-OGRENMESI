import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier

# Veri setini oluştur
X, y = datasets.make_classification(n_samples=500, n_features=10, random_state=42)

# Veriyi ikiye böl (az kısmı etiketli)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
y_train_missing = np.copy(y_train)
y_train_missing[np.random.choice(len(y_train), size=int(0.7 * len(y_train)), replace=False)] = -1  # -1 etiketsiz demek

# Self-Training modeli oluştur
base_model = SVC(probability=True)
self_training_model = SelfTrainingClassifier(base_model)

# Modeli eğit
self_training_model.fit(X_train, y_train_missing)

# Test doğruluğunu hesapla
accuracy = self_training_model.score(X_test, y_test)
print(f"Model Doğruluğu: {accuracy:.2f}")
