# Gerekli kütüphanelerin yüklenmesi
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# sklearn.datasets: İçinde hazır veri setleri barındırır. Biz burada "Iris" veri setini kullanacağız.
# sklearn.model_selection: Veriyi eğitim ve test setlerine ayırmak için kullanılır.
# sklearn.naive_bayes: Naive Bayes sınıflandırıcısını içerir.
# sklearn.metrics: Modelin başarımını ölçmek için metrikler sağlar (örneğin doğruluk skoru).

# Veri setini yükleyelim: Iris veri seti, çiçek türlerini sınıflandırmak 
# için kullanılan klasik bir veri setidir.
iris = load_iris()
# iris = load_iris(): Iris veri setini yükler.

# Özellikler ve etiketler
X = iris.data  # Çiçek özellikleri (örneğin yaprak uzunluğu, genişliği)
# X: Çiçek özelliklerini tutar (yaprak uzunluğu, genişliği gibi).

y = iris.target  # Çiçek türleri (örneğin Setosa, Versicolor, Virginica)
# y: Çiçek türlerini tutar (Setosa, Versicolor, Virginica).

# Veriyi eğitim ve test setlerine ayırma (%80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# train_test_split fonksiyonu, veriyi eğitim ve test setlerine ayırır. 
# Test seti %20 olarak ayarlanmıştır.
# X_train, X_test: Eğitim ve test setlerindeki özellikler.
# y_train, y_test: Eğitim ve test setlerindeki etiketler (çiçek türleri).

# Gaussian Naive Bayes modelini oluşturma
model = GaussianNB()
# model = GaussianNB(): Gaussian Naive Bayes modelini oluşturur.

# Modeli eğitim verisi ile eğitme (fit işlemi)
model.fit(X_train, y_train)
# model.fit(X_train, y_train): Eğitim verisini kullanarak modeli eğitir.

# Test verisi ile tahmin yapma
y_pred = model.predict(X_test)
# y_pred = model.predict(X_test): Test verisi üzerinde modelin tahminlerini yapar.

# Modelin doğruluk skorunu hesaplama
accuracy = accuracy_score(y_test, y_pred)
# accuracy_score: Modelin doğruluğunu hesaplar ve doğru tahminlerin oranını verir.

# Sonuçları yazdırma
print(f"Doğruluk oranı: {accuracy:.2f}")
# Doğruluk oranı: Modelin % kaç doğru tahmin yaptığını gösterir.

# Sonuç doğruluk oranı: 1.00 = Yani: %100
