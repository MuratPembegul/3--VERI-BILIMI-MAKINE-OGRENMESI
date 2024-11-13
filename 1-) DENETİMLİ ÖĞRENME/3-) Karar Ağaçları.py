# Gerekli kütüphaneleri içe aktaralım
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test setine ayırmak için
from sklearn.tree import DecisionTreeClassifier  # Karar ağacı sınıflandırıcı modeli
from sklearn.metrics import accuracy_score  # Model doğruluğunu ölçmek için
import numpy as np  # Veri işlemede kullanacağımız kütüphane
# train_test_split: Veriyi eğitim ve test setlerine ayırmamıza yardımcı olur.
# DecisionTreeClassifier: Karar ağacı sınıflandırıcı modelini oluşturur.
# accuracy_score: Modelin doğruluğunu kontrol ederiz.
# numpy: NumPy kütüphanesi veri işlemelerinde işimize yarar.

# X, müşterilerin web sitesinde geçirdiği zaman (örnek veri)
X = np.array([[5], [10], [15], [20], [25]])
# X: Müşterilerin web sitesinde geçirdiği süreleri temsil eden veriler.


# Y, müşterinin ürünü satın alıp almadığını gösteriyor (1: satın aldı, 0: satın almadı)
Y = np.array([0, 0, 1, 1, 1])
# Y: Müşterinin ürünü satın alıp almadığını gösteriyor. 
# 0 ürünü almadığını, 1 ise aldığını ifade eder.

# Veriyi eğitim ve test setlerine ayıralım (%80 eğitim, %20 test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# train_test_split: Veriyi %80 eğitim ve %20 test olarak ayırırız. 
# Test verisi modeli denemek için kullanılır.

# Karar ağacı sınıflandırıcı modelini oluşturalım
model = DecisionTreeClassifier()
# DecisionTreeClassifier(): Karar ağacı sınıflandırıcı modelini başlatır.

# Modeli eğitim verisiyle eğitelim
model.fit(X_train, Y_train)
# model.fit(): Modeli eğitim verisiyle (X_train, Y_train) eğitiriz.

# Test verisi üzerinde tahmin yapalım
predictions = model.predict(X_test)
# model.predict(): Test verisini kullanarak modelin tahminlerini elde ederiz 
# ve sonuçları ekrana yazdırırız.

# Tahmin sonuçlarını ekrana yazdıralım
print(f"Tahminler: {predictions}")


# Modelin doğruluğunu kontrol edelim
accuracy = accuracy_score(Y_test, predictions)
# accuracy_score(): Gerçek sonuçlar (Y_test) ile modelin tahmin ettiği sonuçları 
# karşılaştırarak doğruluk oranını hesaplar.
print(f"Doğruluk: {accuracy}")
