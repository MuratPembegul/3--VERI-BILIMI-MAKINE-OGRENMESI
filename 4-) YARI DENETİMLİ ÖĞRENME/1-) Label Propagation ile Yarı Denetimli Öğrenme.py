import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation

# Veri setini oluştur (İki sınıflı)
X, y = datasets.make_moons(n_samples=200, noise=0.1)

# Etiketlerin %90'ını gizle (etiketsiz veri gibi olsun)
y_missing = np.copy(y)
y_missing[np.random.choice(len(y), size=int(0.9 * len(y)), replace=False)] = -1  # -1 etiketsiz demek

# Modeli eğit
model = LabelPropagation()
model.fit(X, y_missing)

# Sonuçları görselleştir
plt.scatter(X[:, 0], X[:, 1], c=model.predict(X), cmap='coolwarm')
plt.title("Label Propagation ile Yarı Denetimli Öğrenme")
plt.show()
