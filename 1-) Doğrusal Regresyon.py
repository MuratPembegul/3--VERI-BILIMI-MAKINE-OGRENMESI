# Önce scikit-learn kütüphanesini yüklüyoruz.
from sklearn.linear_model import LinearRegression

# 1. Basit bir veri seti oluşturalım 
# (örneğin bir odadaki masa sayısı ve o odadaki sandalye sayısı ilişkisi)
X = [[1], [2], [3], [4], [5]]  # Bağımsız değişkenler (örneğin masa sayıları)
y = [2, 4, 6, 8, 10]  # Bağımlı değişkenler (örneğin sandalye sayıları)

# 2. Modeli oluştur
model = LinearRegression()

# 3. Modeli eğit (verileri kullanarak modeli oluşturma)
model.fit(X, y)

# 4. Tahmin yap (Örneğin, 6 masalı bir odada kaç sandalye olur?)
tahmin = model.predict([[6]])

# 5. Sonucu ekrana bastıralım
print(tahmin)  # Bu satır tahmin sonucunu ekrana yazdırır

# X: Bu liste, bağımsız değişkenleri (input) içeriyor. 
# Burada, 1 masalı, 2 masalı, 3 masalı vs. odaları örnek verdik.
# y: Bu liste, bağımlı değişkenleri (output) temsil ediyor. 
# 2 sandalye, 4 sandalye, 6 sandalye gibi değerler var.
# model.fit(X, y): Modelimizi bu verilere göre eğitiyoruz. 
# Yani X ve y arasındaki ilişkiyi öğreniyor.
# model.predict([[6]]): 
# Modelimize 6 masalı bir oda bilgisi veriyoruz 
# ve 6 masalı odada kaç sandalye olacağını tahmin ediyoruz.
# print(tahmin): Tahmin edilen sonucu ekrana yazdırıyoruz.
# Sonuç : 12