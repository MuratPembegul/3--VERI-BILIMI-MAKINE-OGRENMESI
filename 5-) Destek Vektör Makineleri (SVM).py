# Gerekli kütüphaneleri yüklüyoruz
from sklearn import datasets  # Veri setleri oluşturmak için
from sklearn.model_selection import train_test_split  # Eğitim ve test seti ayırma
from sklearn.svm import SVC  # SVM sınıflandırıcısı
from sklearn.metrics import accuracy_score  # Doğruluk hesaplama
# datasets: Scikit-learn kütüphanesindeki hazır veri setlerini kullanmak için bu kütüphaneyi yüklüyoruz.
# train_test_split: Veriyi eğitim ve test setlerine ayırmak için kullanıyoruz.
# SVC: Support Vector Classifier (SVM sınıflandırıcısı) anlamına gelir. 
# SVM modelini oluşturmak için bu sınıfı kullanıyoruz.
# accuracy_score: Modelin doğruluğunu hesaplamak için kullanılır.

# Örnek bir veri seti oluşturuyoruz
iris = datasets.load_iris()
# iris veri seti: Iris çiçeği türlerinin sınıflandırılması için sıkça kullanılan bir veri setidir. 
# Bu veri seti her bir çiçek türü için çeşitli özellikler içerir (yaprak uzunluğu, genişliği vb.).
X = iris.data  # Özellikler (Çiçeğin boyutları gibi veriler)
# X: Çiçeklerin özelliklerini temsil eden veri noktalarıdır (giriş verileri).
y = iris.target  # Hedef sınıflar (Çiçek türleri)
# y: Bu özelliklere karşılık gelen hedef sınıflardır
# (hangi çiçek türü olduğu: Setosa, Versicolor veya Virginica).

# Veriyi eğitim ve test seti olarak ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# train_test_split: Veriyi iki parçaya böleriz: Eğitim seti (%80) ve test seti (%20). 
# Eğitim seti modelimizi eğitmek için, 
# test seti ise modelimizin doğruluğunu değerlendirmek için kullanılır.
# random_state=42: Aynı sonuçları elde edebilmek için bu sabit bir değerdir 
# (farklı çalıştırmalarda aynı sonuçları almayı sağlar).

# SVM modelini oluşturuyoruz (doğrusal kernel kullanarak)
model = SVC(kernel='linear')
# SVC(kernel='linear'): Destek vektör makineleri modelini oluşturuyoruz. 
# Burada "linear" kernel kullanarak veriler arasındaki ayırma çizgisini doğrusal 
# (düz) bir şekilde yapmaya çalışıyoruz. 
# Kernel, sınıflandırma işleminin temelini oluşturan fonksiyondur.

# Modeli eğitiyoruz
model.fit(X_train, y_train)
# model.fit(): SVM modelini eğitim verileri ile eğitiyoruz. 
# Model, eğitim verilerini kullanarak veri noktaları arasındaki en iyi ayırma sınırını bulmaya çalışır.

# Tahmin yapıyoruz
y_pred = model.predict(X_test)
# model.predict(): Modeli test etmek için test verilerini kullanıyoruz. 
# Model, bu verilere dayanarak hangi çiçek türü olduğunu tahmin etmeye çalışır.

# Modelin doğruluğunu hesaplıyoruz
accuracy = accuracy_score(y_test, y_pred)
# accuracy_score(): Modelin ne kadar doğru çalıştığını hesaplar. 
# Bu, tahmin edilen sınıflarla gerçek sınıfların ne kadar örtüştüğünü gösterir.

print(f"Model Doğruluk Skoru: {accuracy:.2f}")
# f"Model Doğruluk Skoru: {accuracy:.2f}": 
# Sonuç olarak, modelimizin doğruluk skorunu ekrana yazdırırız. 
# Sonuç genellikle 0.00 ile 1.00 arasında bir değer olur (1.00 = %100 doğruluk).
