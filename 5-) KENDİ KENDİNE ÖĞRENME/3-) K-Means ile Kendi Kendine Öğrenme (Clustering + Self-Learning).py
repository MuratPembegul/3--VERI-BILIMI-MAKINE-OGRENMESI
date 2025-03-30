import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC

# Veri setini oluştur
X, _ = make_blobs(n_samples=500, centers=3, random_state=42)

# Verinin küçük bir kısmına rastgele etiket atayalım
y_labels = np.full(len(X), -1)  # Başlangıçta tümü etiketsiz
random_indices = np.random.choice(len(X), size=20, replace=False)
y_labels[random_indices] = np.random.choice([0, 1, 2], size=20)  # Rastgele 3 sınıf atadık

# K-Means modeliyle etiketsiz verileri sınıflandır
kmeans = KMeans(n_clusters=3, random_state=42)
pseudo_labels = kmeans.fit_predict(X)

# Sadece etiketsiz olanları değiştir
y_labels[y_labels == -1] = pseudo_labels[y_labels == -1]

# Self-Training ile tekrar eğit
base_model = SVC(probability=True)
self_training_model = SelfTrainingClassifier(base_model)
self_training_model.fit(X, y_labels)

print("Model eğitildi ve kendi kendine öğrendi!")
