import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from MLP_Class_FromScratch import MultiP

listind=np.linspace(start=0,stop=20427,num=3000).astype(int) #Verisetinin tamamını kullanmadığımız durumda alt satırdaki ilk iki nokta yerine listind yazılır.
print(listind)
df = pd.read_pickle("df.pkl").iloc[:,:]
y = df.median_house_value.to_numpy()
X = df.drop(["median_house_value"],axis=1).to_numpy()

scaler1 = MinMaxScaler()
scaler2 = StandardScaler()
scaler3 = RobustScaler()



iter = 1 # Ağ'ın çalışma sayısı burada belirlenir.
mse, r2 = [],[] # Ağın her çalışmasında elde edilen değerlendirme metrikleri bu boş listelerin içine aktarılacak.
for i in range(iter):
        mlp = MultiP(inputDim=X.shape[1], firstLayer=32, secondLayer=16, outputDim=1)

        X = scaler2.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        mlp.train(X_train,y_train,epochs=300,lr=0.1,dropout1=0.0,dropout2=0.0)  # Dropout oranları hem eğitimde hem testte verilmelidir ve birbiriyle uyuşmalıdır.

        y_pred = np.zeros(len(y_test))
        for k in range(len(X_test)):
                output = mlp.predict(X_test[k],dropout1=0.0,dropout2=0.0)
                y_pred[k] = output # her veri için tahmin edilen değer y_pred dizisine aktarılıyor.
        r2.append(r2_score(y_test, y_pred))
        mse.append(mean_squared_error(y_test, y_pred))

print(f"\n\nr2 score: {sum(r2) / len(r2)}")
print(f"mean squared error: {sum(mse) / len(mse)}")

plt.plot(range(100),y_test[:100]*(5*10**5),c='b', label = 'Gerçek Değerler')
plt.plot(range(100),y_pred[:100]*(5*10**5),c='r', label= 'Tahmin Edilen Değerler')
plt.legend()
plt.show()