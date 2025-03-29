from sklearn.decomposition import PCA

# PCA uygula (2 boyuta indir)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Grafik çiz
plt.scatter(X_pca[:, 0], X_pca[:, 1], marker='o')
plt.title("PCA ile Boyut İndirgeme")
plt.show()
