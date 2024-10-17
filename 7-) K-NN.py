import numpy as np # Numpy kütüphanesini yüklüyoruz.
import matplotlib.pyplot as plt # Matplotlib kütüphanesini yüklüyoruz.
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
# Bu adımda, numpy (sayısal işlemler için), 
# matplotlib (grafikler için), 
# sklearn (makine öğrenmesi algoritmaları için) gibi gerekli kütüphaneleri yüklüyoruz. 
# load_iris fonksiyonu ile örnek bir veri seti yükleyeceğiz. 
# StandardScaler verileri ölçeklendirmek için, 
# KNeighborsClassifier ise K-NN algoritmasını çalıştırmak için kullanılacak.

# 1. Veri Kümesinin Yüklenmesi
data = load_iris()
# load_iris() ile Iris veri setini yüklüyoruz. 
# Bu veri seti 150 örnekten oluşan, 
# her bir örneğin dört özellik taşıdığı 
# ve çiçek türlerini sınıflandırmak için kullanıldığı bir veri setidir.

X = data.data
# X değişkeni, özelliklerimizi (verinin boyutlarını) içerir.
y = data.target
# y değişkeni ise sınıf etiketlerini (hangi tür çiçek olduğunu) içerir.

# 2. Eğitim ve Test Verisinin Ayrılması
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# train_test_split fonksiyonunu kullanarak 
# veri setini eğitim (%70) ve test (%30) verisi olarak ayırıyoruz. 
# Bu sayede modelimizi test edebileceğiz.

# 3. Verilerin Ölçeklendirilmesi
scaler = StandardScaler()
# Veriler arasında farklı ölçeklerde değerler olabilir, 
# bu yüzden StandardScaler ile verileri ölçeklendiriyoruz. 
# Bu adım, K-NN gibi mesafeye dayalı algoritmalarda performansı artırmak için önemlidir. 
X_train = scaler.fit_transform(X_train)
# scaler.fit_transform(X_train): X_train verilerini ölçeklendirir, 
# yani her özelliği ortalaması 0, standart sapması 1 olacak şekilde dönüştürür.

X_test = scaler.transform(X_test)
# scaler.transform(X_test): Test verisini de aynı şekilde ölçeklendirir, 
# ancak burada fit işlemi tekrar yapılmaz, 
# sadece eğitim verisi üzerinden elde edilen parametrelerle dönüştürme yapılır. 
# Bu sayede model, test verisine aşırı uyum sağlamaz (overfitting).

# 4. K-NN Modelinin Eğitilmesi
knn = KNeighborsClassifier(n_neighbors=3)
# KNeighborsClassifier(n_neighbors=3) ile K-NN algoritmasını 3 en yakın komşu (K=3) ile kullanacak 
# bir model oluşturuyoruz. 
# K değeri, komşu sayısını belirler ve modelin doğruluğunu etkileyen önemli bir parametredir.

knn.fit(X_train, y_train)
# fit() fonksiyonu ile modelimizi eğitim verileriyle (X_train ve y_train) eğitiyoruz.

# 5. Tahmin Yapılması
y_pred = knn.predict(X_test)
# predict() fonksiyonu ile modelimizi test verisine uyguluyoruz. 
# Bu fonksiyon, test verisine göre sınıf tahminleri yapar. 
# y_pred değişkeni, tahmin edilen sınıf etiketlerini tutar.

# 6. Sonuçların Değerlendirilmesi
accuracy = accuracy_score(y_test, y_pred)
# accuracy_score() ile modelin doğruluk oranını hesaplıyoruz. 
# Bu oran, test verisinin ne kadarını doğru sınıflandırdığımızı gösterir.

print(f"Model Doğruluğu: {accuracy * 100:.2f}%")
# Model Doğruluğu: 97.78%

cm = confusion_matrix(y_test, y_pred)
# confusion_matrix() fonksiyonu ile karışıklık matrisi (confusion matrix) oluşturuyoruz. 
# Bu matris, doğru ve yanlış sınıflandırmaların detaylarını gösterir. 
# Matrisin köşegenlerinde doğru sınıflandırmalar yer alır.
# y_test, test verisinin gerçek etiketlerini içerir.
# y_pred, model tarafından test verisi üzerinde yapılan tahminlerin etiketlerini içerir.
# accuracy_score() fonksiyonu, gerçek etiketlerle (y_test) tahmin edilen etiketlerin 
# (y_pred) karşılaştırmasını yapar ve 
# ne kadarının doğru sınıflandırıldığını hesaplar.

print("Karışıklık Matrisi:")
# Karışıklık Matrisi:
#[[16 0  0]
#[ 0 15  1]
#[ 0  0 13]]
# [16, 0, 0]: Gerçek sınıfı 0 olan 16 veri noktası var 
# ve model bunların hepsini doğru sınıflandırmış (tahmin edilen sınıf da 0).
#[0, 15, 1]: Gerçek sınıfı 1 olan 16 veri noktası var, model 15 tanesini doğru sınıflandırmış 
# (tahmin edilen sınıf 1), ancak 1 tanesini yanlışlıkla sınıf 2 olarak tahmin etmiş.
#[0, 0, 13]: Gerçek sınıfı 2 olan 13 veri noktası var ve 
# model bunların hepsini doğru sınıflandırmış (tahmin edilen sınıf da 2).

print(cm)

#       Tahminlenen Sınıf 0	           Tahminlenen Sınıf 1	                 Tahminlenen Sınıf 2
#Gerçek Sınıf 0	Doğru Sınıflandırma	   Yanlış Sınıflandırma	                 Yanlış Sınıflandırma
#Gerçek Sınıf 1	Yanlış Sınıflandırma	Doğru Sınıflandırma	                 Yanlış Sınıflandırma
#Gerçek Sınıf 2	Yanlış Sınıflandırma	Yanlış Sınıflandırma                  Doğru Sınıflandırma
