using System; // System kütüphanesini kullanıyoruz.

class MatematikVeKosul // Sınıfın adı: MatematikVeKosul
{
    static void Main() // Programın başladığı yer, Main metodu.
    {
        // Değişkenler
        int sayi1 = 10; // Birinci tam sayı değişkeni
        int sayi2 = 20; // İkinci tam sayı değişkeni

        // Toplama, çıkarma, çarpma ve bölme işlemleri
        int toplam = sayi1 + sayi2; // sayi1 ve sayi2'yi topluyoruz.
        int fark = sayi1 - sayi2; // sayi1 ve sayi2'nin farkı alınıyor.
        int carpim = sayi1 * sayi2; // sayi1 ile sayi2 çarpılıyor.
        int bolum = sayi2 / sayi1; // sayi2 sayi1'e bölünüyor.

        // İşlemlerin sonuçlarını ekrana yazdırma
        Console.WriteLine("Toplam: " + toplam);
        Console.WriteLine("Fark: " + fark);
        Console.WriteLine("Çarpım: " + carpim);
        Console.WriteLine("Bölüm: " + bolum);

        // Koşul ifadesi (if-else)
        if (toplam > 30) // Eğer toplam 30'dan büyükse
        {
            Console.WriteLine("Toplam 30'dan büyük.");
        }
        else // Aksi halde (toplam 30'a eşit veya küçükse)
        {
            Console.WriteLine("Toplam 30'dan küçük veya eşit.");
        }
    }
}
// int sayi1 = 10; ve int sayi2 = 20;

//Açıklama: İki tane tam sayı değişkeni tanımlıyoruz. sayi1 değeri 10, sayi2 değeri ise 20 olarak ayarlandı.

//int toplam = sayi1 + sayi2;

//Açıklama: Burada sayi1 ve sayi2 değişkenlerinin toplamını hesaplayıp toplam adlı yeni bir değişkene atıyoruz.

//int fark = sayi1 - sayi2;

//Açıklama: Bu satırda iki sayının farkını alıyoruz (sayi1 - sayi2).

//int carpim = sayi1 * sayi2;

//Açıklama: İki sayının çarpımını hesaplıyoruz.

//int bolum = sayi2 / sayi1;

//Açıklama: İkinci sayıyı birinci sayıya bölüyoruz ve sonucu bolum adlı değişkene atıyoruz.

//Console.WriteLine(...)

    //Açıklama: Her matematiksel işlemin sonucunu ekrana yazdırıyoruz.

//if (toplam > 30)

// Açıklama: Koşul ifadesi(if). Eğer toplam değişkeni 30'dan büyükse, ekrana "Toplam 30'dan büyük." mesajını yazdırıyoruz.
// Eğer bu şart sağlanmazsa (else), "Toplam 30'dan küçük veya eşit." mesajı ekrana yazılır.
