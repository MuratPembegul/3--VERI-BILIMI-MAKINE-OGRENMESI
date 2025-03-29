import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Dendrogram çiz
plt.figure(figsize=(10, 5))
sch.dendrogram(sch.linkage(X, method='ward'))
plt.title("Hiyerarşik Kümeleme Dendrogramı")
plt.show()

# Hiyerarşik kümeleme modeli
hc = AgglomerativeClustering(n_clusters=3)
labels = hc.fit_predict(X)

# Küme grafiği
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='coolwarm', marker='o')
plt.title("Hiyerarşik Kümeleme Sonucu")
plt.show()
