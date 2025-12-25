"""
flowers dataset:
     rgb: 224x224

CNN ile siniflandirma modeli ve problemi çözme
"""

#import libraries
from tensorflow_datasets import load
from tensorflow.data import AUTOTUNE
from tensorflow.keras import Sequential
from tensorflow.keras.layers import(
    Conv2D, #2D convulutional layer
    MaxPooling2D, #max pooling layer
    Flatten, #çok boyutlu veriyi tek boyutlu kale getirme
    Dense, #tam bağlantılı katman
    Dropout #rastgele nöronları kapatma ve overfitting önleme
)
from tensorflow.keras.optimizers import Adam # optimizer
from tensorflow.keras.callbacks import (
    EarlyStopping, #erken durdurma
    ReduceLROnPlateau, #öğrenme oranını azaltma
    ModelCheckpoint #model kaydetme
)
import tensorflow as tf
import matplotlib.pyplot as plt #görselleştirme


#veri seti yükleme
(ds_train, ds_val), ds_info= load("tf_flowers", #veri seti ismi
               split=["train[:80%]", #veri setinin %80i eğitim
                      "train[80%:]"], #veri setinin %20si test için
                as_supervised=True, #veri setinin görsel, etiket çiftinin olması
               with_info=True #veri seti hakkında bilgi alma
                                 )


print(ds_info.features) #veri seti hakkında bilgi yazdırma
print("Number of ckasses:", ds_info.features['label'].num_classes)



#örnek verileri görselleştirme
#eğitim setinden rastgele 3 resim ve etiket alma
fig = plt.figure(figsize = (10,5))
for i, (image, label) in enumerate(ds_train.take(3)):
    ax = fig.add_subplot(1 ,3, i+1) #1 satır 3 sütun i+1. resim
    ax.imshow(image.numpy().astype("uint8")) #resim görselleştirme
    ax.set_title(f"etiket: {label.numpy()}") #etiket başlık olarak yazdırma
    ax.axis("off") #eksenleri kapatma

plt.tight_layout()
plt.show()


IMG_SIZE=(180, 180)

#data augmentation + preprocessing(ön işleme) 
def preprocess_train(image, label):
    """
    resize, random flip, brightness, contrast, crop, normalize
    """
    image=tf.image.resize(image, IMG_SIZE)#boyutlandırma
    image=tf.image.random_flip_left_right(image)#yatay olarak rastgele çevirme
    image=tf.image.random_brightness(image, max_delta=0.1)#rastgele parlaklık
    image= tf.image.random_contrast(image, lower=0.9, upper=1.2)#rastgele kontrast
    image=tf.image.random_crop(image, size=(160,160 , 3))#rastgele kırp
    image=tf.image.resize(image, IMG_SIZE)#tekrar boyutlandırma
    image=tf.cast(image, tf.float32)/255.0#normalize etme
    return image, label


def preprocess_val(image, label):
    """
    resize, normalize
    """
    image=tf.image.resize(image, IMG_SIZE)#boyutlandırma
    image=tf.cast(image, tf.float32)/255.0#normalize etme
    return image, label

ds_train = (
    ds_train
    .map(preprocess_train, num_parallel_calls=AUTOTUNE)#ön işleme ve augmentasyon
    .shuffle(1000)#karıştırma
    .batch(32)#bach boyutu
    .prefetch(AUTOTUNE)#veri setini önceden hazırlama 
)

ds_val = (
    ds_val
    .map(preprocess_val, num_parallel_calls=AUTOTUNE)#ön işleme ve augmentasyon
    .shuffle(1000)#karıştırma
    .batch(32)#bach boyutu
)


#cnn modeli oluşturma
model = Sequential([
    #feature extraction katmanları
    Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)), #32 filtre, 3x3 kernel, relu aktivasyon fonksiyonu, 3 kanal (rgb)
    MaxPooling2D((2, 2)), #bir şey yazmazsak 2x2 varsayar

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    #sınıflandırma katmanları
    Flatten(), #çok boyutlu veriyi tek boyutlu vektöre çevirme
    Dense(512, activation='relu'),
    Dropout(0.5), #overfittingi engellemek için rastgele nöronları kapatma
    Dense(ds_info.features['label'].num_classes, activation='softmax') #çıkış katmanı, softmax aktivasyon fonksiyonu
])


#callbacks tanımlanması
callbacks =[
#eğer validasyon kaybı 3 epoch boyunca iyileşmezse eğitimi durdur ve en iyi ağırlıkları yükle
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True), # erken durdurma

#vall loss 2 epoch boyunca iyileşmezse learning rate 0.2 carpani ile azaltılır
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, min_lr=1e-9), # öğrenme oranını azaltma

#her epoch sonunda eğer model daha iyi ise kaydolur  
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)# en iyi ağırlığı kaydetme
]


#derleme
model.compile(
    optimizer =Adam(learning_rate=0.001), # adam optimizer, öğrenme oranı 0.001 olarak ayarla
    loss = 'sparse_categorical_crossentropy', #kayıp fonksiyonu, etiketler tamsayı olduğu içn sparse kullun
    metrics = ['accuracy'] #başarı metriği
)


#traning
history = model.fit(
    ds_train, # eğitim veri seti
    validation_data= ds_val, #validasyon veri seti
    epochs = 15, #epoch sayısı
    callbacks= callbacks, #callback'ler
    verbose = 1 #eğitim ilerlemesini göster
)


#model evaluation
plt.figure(figsize=(12,5))

#doğruluk grafiği
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Validasyon Doğruluğu')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()


#loss grafiği
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Validasyon Kaybı')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()