import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Rastgele veri oluştur
np.random.seed(42)
X = np.random.rand(100, 2)

# K-Means modelini oluştur ve eğit
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Küme merkezlerini al
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Grafik çiz
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label="Merkezler")
plt.legend()
plt.title("K-Means Kümeleme")
plt.show()
