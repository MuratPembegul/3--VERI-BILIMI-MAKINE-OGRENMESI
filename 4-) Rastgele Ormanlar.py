# Gerekli kütüphaneleri yüklüyoruz
from sklearn.datasets import make_classification  # Sınıflandırma için örnek veri seti oluşturur
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test olarak böler
from sklearn.ensemble import RandomForestClassifier  # Rastgele Orman sınıflandırıcısı
from sklearn.metrics import accuracy_score  # Doğruluk skorunu hesaplar

# make_classification: Örnek bir sınıflandırma veri seti oluşturur. 1000 örnek ve 10 özellikten oluşan bir veri seti oluşturuyoruz.
# train_test_split: Veriyi eğitim ve test seti olarak böleriz. 
# Modeli eğitmek ve doğruluğunu test etmek için kullanılır.
# RandomForestClassifier: Rastgele Orman algoritması sınıflandırma problemleri için kullanılır. 
# Karar ağaçlarının bir ensemble'ıdır (topluluğu).
# accuracy_score: Modelin tahminlerinin doğruluğunu ölçmek için kullanılır.

# 1. Adım: Veri seti oluşturma
# make_classification fonksiyonu ile 2 sınıflı örnek bir veri seti oluşturuyoruz
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_classes=2, random_state=42)
# make_classification: 2 sınıfa ait örnek veri seti oluşturuyoruz. 
# Her bir veri noktasının 10 farklı özelliği var. 
# Bu veri seti, algoritmamızın ne kadar iyi çalıştığını test etmek için kullanılır.

# 2. Adım: Veriyi eğitim ve test olarak bölme
# Veri setini %80 eğitim, %20 test olarak bölüyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# train_test_split: Verinin %80'ini modelimizi eğitmek için, %20'sini ise testi için ayırıyoruz. 
# random_state=42 parametresi, her çalıştırmada aynı sonuçları almak için kullanılır.

# 3. Adım: Rastgele Orman modelini oluşturma
# 100 adet karar ağacından oluşan bir Rastgele Orman sınıflandırıcısı oluşturuyoruz
model = RandomForestClassifier(n_estimators=100, random_state=42)
# RandomForestClassifier: 100 adet karar ağacından oluşan bir model oluşturuyoruz. 
# Rastgele ormanlar, birden fazla karar ağacını birleştirerek daha iyi bir genel tahmin yapar.

# 4. Adım: Modeli eğitme
# Eğitim verileri ile modeli eğitiyoruz
model.fit(X_train, y_train)
# fit: Eğitim verileri (X_train ve y_train) ile modelimizi eğitiyoruz. 
# Bu aşamada model, verilerdeki kalıpları öğrenir.

# 5. Adım: Tahmin yapma
# Test verileri ile tahmin yapıyoruz
y_pred = model.predict(X_test)
# predict: Test verileri (X_test) üzerinde tahminler yapıyoruz. 
# Model test verilerine bakarak hangi sınıfa ait olduklarını tahmin eder.

# 6. Adım: Modelin doğruluğunu ölçme
# Modelin doğruluk skorunu hesaplıyoruz
accuracy = accuracy_score(y_test, y_pred)
# accuracy_score: Modelin ne kadar doğru tahmin yaptığını ölçüyoruz. 
# Doğruluk, modelin kaç doğru tahmin yaptığına göre hesaplanır. 
# Bu doğruluk skoru, 
# 0 ile 1 arasında bir değer olur ve daha yüksek bir skor, daha iyi bir modeli gösterir.

# Sonucu ekrana yazdırma
print(f"Model Doğruluk Skoru: {accuracy:.2f}")

# Sonuç: Model Doğruluk Skoru: 0.90

