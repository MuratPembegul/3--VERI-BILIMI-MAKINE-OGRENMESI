from sklearn.manifold import TSNE

# t-SNE uygula
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Grafik çiz
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], marker='o')
plt.title("t-SNE ile Boyut İndirgeme")
plt.show()
