import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Veri Önişleme (Data Preprocessing)
# Ağ çalıştırılmadan önce Preprocessing kodu çalıştırılmalı!
# Veri önişleme kısmı Jupyter Notebook üzerinde yapıldı ve raporda da cıktılar, görseller detaylarıyla gösterildi.
# Daha sonra yapılan işlemler ve denemeler bu script üzerinde çalıştırıldı.

df = pd.read_csv("housing.csv") #veriseti içeri aktarıldı.




df1 = df[df.ocean_proximity != "ISLAND"] #çok az bulunan ISLAND sınıfı çıkarıldı. (Raporda gösterildi.)
df2 =  pd.get_dummies(df1.ocean_proximity, prefix='ocean_proximity')
df1 = df1.drop(["ocean_proximity"],axis=1)
df_encoded = pd.concat([df1,df2],axis=1)  # Kategorik değişkenler sayısallaştırıldı. (One-Hot Encoding)

df_encoded["Bedrooom_PerHouseHold"] = df_encoded["total_bedrooms"] / df_encoded["households"]
df_encoded.drop(columns=["total_bedrooms"],inplace=True) # hane başı ortalama yatak odası sütunu oluşturuldu. Toplam yatak odası sütunu kaldırıldı.

df_encoded["Room_PerHouseHold"] = df_encoded["total_rooms"] / df_encoded["households"]
df_encoded.drop(columns=["total_rooms"],inplace=True) # hane başı ortalama oda sütunu oluşturuldu. Toplam oda sütunu kaldırıldı.

df_encoded["People_PerHouseHold"] = df_encoded["population"] / df_encoded["households"]
df_encoded.drop(columns=["population","households"],inplace=True) # hane başı ortalama insan sütunu oluşturuldu. Toplam popülasyon ve hane sayısı sütunu kaldırıldı.
# Bu 6 satır ile yapılan sütun değişiklikleri ile kısmen de olsa daha iyi sonuç alındığı gözlenmiştir.


df_encoded.plot(kind="scatter", x="longitude", y="latitude", alpha=1, s=df_encoded["People_PerHouseHold"]/100, label="People_PerHouseHold", figsize=(10,7),
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,) # California haritasına göre ev fiyatlarının artışı ısı haritası ile gösterildi.
plt.legend()
df_encoded.plot(kind="scatter", x="People_PerHouseHold", y="median_house_value",alpha=0.1) #hane başı insan sayısı ile ev fiyatı arasındaki ilişkiyi gösteren grafik.



df_encoded.median_house_value = df_encoded.median_house_value / (5*10**5) # tahmin edilecek değer 0 ile 1 arasına ölçeklendi.
df_encoded.dropna(axis=0,inplace=True) #Boş değer içeren veriler kaldırıldı. (Raporda detaylı olarak gösterildi.)

df_encoded.to_pickle("df.pkl") # Ağın çalıştırıldığı kodlarda çağrılabilmesi için veriseti kaydedildi.



corrmat = df_encoded.corr()
f, ax = plt.subplots(figsize=(12, 9))
f.suptitle("Correlation Matrix")
sns.heatmap(corrmat, square=True, annot=True) # Sütunların Korelasyon matrisi

plt.show()