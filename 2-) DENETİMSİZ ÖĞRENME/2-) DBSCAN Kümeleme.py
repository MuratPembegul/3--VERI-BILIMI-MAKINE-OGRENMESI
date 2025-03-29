from sklearn.cluster import DBSCAN

# DBSCAN modelini oluştur
dbscan = DBSCAN(eps=0.1, min_samples=5)
labels = dbscan.fit_predict(X)

# Grafik çiz
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma', marker='o')
plt.title("DBSCAN Kümeleme")
plt.show()
