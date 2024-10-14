# Gerekli kütüphaneleri içe aktaralım
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test setine ayırmak için
from sklearn.linear_model import LogisticRegression  # Lojistik regresyon modeli
from sklearn.metrics import accuracy_score  # Model doğruluğunu ölçmek için
import numpy as np  # Veri işlemede kullanacağımız kütüphane
# train_test_split: Veriyi eğitim ve test setlerine ayırırız.
# LogisticRegression: Lojistik regresyon modelini kurarız.
# accuracy_score: Modelin doğruluğunu ölçmek için kullanırız.
# numpy: NumPy kütüphanesiyle verileri işlemeye hazır hale getiririz.


# X, müşterilerin web sitesinde geçirdiği zaman (örnek veri)
# Burada 5, 10, 15, 20 ve 25 dakika sitede geçirilen zamanları temsil ediyor
X = np.array([[5], [10], [15], [20], [25]])

# Y, müşterinin ürünü satın alıp almadığını gösteriyor (1: satın aldı, 0: satın almadı)
Y = np.array([0, 0, 1, 1, 1])

# X: Müşterilerin web sitesinde geçirdiği zaman. 
# Örneğin, bir müşteri sitede 5 dakika, diğeri 10 dakika kalmış.
# Y: Müşterinin ürünü satın alıp almadığını gösteriyor. 0 ürünü almadığını, 1 ise aldığını gösterir.



# Veriyi eğitim ve test setlerine ayıralım (%80 eğitim, %20 test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# train_test_split: Veriyi %80 eğitim ve %20 test olarak ayırır. 
# Bu, modelin doğruluğunu test etmek için yapılır. 

# Lojistik regresyon modelini oluşturalım
model = LogisticRegression()

# LogisticRegression(): Lojistik regresyon modelini başlatır.

# Modeli eğitim verisiyle eğitelim
model.fit(X_train, Y_train)

# model.fit(): Modeli, eğitim verisi (X_train, Y_train) üzerinde eğitir.

# Test verisi üzerinde tahmin yapalım
predictions = model.predict(X_test)

# model.predict(): Test verisiyle (X_test) model tahmin yapar ve bu tahminler ekrana yazdırılır.

# Tahmin sonuçlarını ekrana yazdıralım
print(f"Tahminler: {predictions}")



# Modelin doğruluğunu kontrol edelim
accuracy = accuracy_score(Y_test, predictions)
print(f"Doğruluk: {accuracy}")

# accuracy_score(): Gerçek sonuçlar 
# (Y_test) ile modelin tahmin ettiği sonuçları karşılaştırır ve doğruluğu hesaplar. 