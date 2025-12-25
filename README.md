🌸 CNN ile Çiçek Sınıflandırma Projesi (Flower Classification)
Bu proje, Derin Öğrenme (Deep Learning) teknikleri kullanılarak 5 çiçek türünü sınıflandırmak amacıyla geliştirilmiştir.
TensorFlow ve Keras kütüphaneleri kullanılarak oluşturulan Convolutional Neural Network (CNN) modeli, tf_flowers veri seti üzerinde eğitilmiştir.

🚀 Proje Hakkında
Bu çalışmada, görüntü işleme ve yapay zeka teknikleri bir araya getirilmiştir.
Modelin ezberlemesini (overfitting) önlemek ve başarısını artırmak için Veri Zenginleştirme (Data Augmentation) ve Ön İşleme (Preprocessing) teknikleri etkin bir şekilde kullanılmıştır.

📂 Veri Seti
Projede TensorFlow Datasets kütüphanesinden tf_flowers veri seti kullanılmıştır.
Toplam Resim Sayısı: ~3670Sınıflar (5 Adet): Papatya (Daisy), Karahindiba (Dandelion), Gül (Rose), Ayçiçeği (Sunflower), Lale (Tulip).
Eğitim/Test Ayrımı: %80 Eğitim, %20 Test (Validasyon).

🛠 Kullanılan Teknolojiler ve Yöntemler
Python 3.x
TensorFlow & Keras (Model Mimarisi)
Matplotlib (Veri Görselleştirme)
TensorFlow Datasets (Veri Yönetimi)

Model Mimarisi ve Teknikler
CNN (Convolutional Neural Network): Görüntülerden özellik çıkarmak için Conv2D ve MaxPooling2D katmanları.
Data Augmentation: Rastgele döndürme, parlaklık/kontrast ayarı ve kırpma işlemleri ile veri çeşitliliği artırıldı.
Callbacks:
    EarlyStopping: Model gelişimi durursa eğitimi erken bitirme.
    ModelCheckpoint: En iyi ağırlıkları kaydetme (best_model.h5).
    ReduceLROnPlateau: Öğrenme oranını dinamik olarak ayarlama.
    
📊 Sonuçlar (Results)
Model 15 epoch boyunca eğitilmiş ve aşağıdaki başarı oranlarına ulaşılmıştır:
    Eğitim Doğruluğu (Training Accuracy)  ~  %85
    Validasyon Doğruluğu (Validation Accuracy)  ~  %79
        
💻 Kurulum ve Çalıştırma
Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:
    Projeyi Klonlayın:
         git clone https://github.com/kullanici_adiniz/repo_isminiz.git
         cd repo_isminiz
    Sanal Ortamı Oluşturun (Opsiyonel ama önerilir):
         python -m venv venv
         # Windows için:
         .\venv\Scripts\activate
         # Mac/Linux için:
         source venv/bin/activate
    Gerekli Kütüphaneleri Yükleyin:
         pip install -r requirements.txt
    Modeli Eğitin:
         python cnn.py

📝 Not
Bu proje eğitim amaçlı geliştirilmiştir.Model performansı daha fazla epoch sayısı veya Transfer Learning (örn: MobileNet, ResNet) yöntemleri ile artırılabilir.
