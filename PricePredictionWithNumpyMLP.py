import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from MLP_Class_FromScratch import MultiP

listind=np.random.choice(20000,size=2000,replace=False)
df = pd.read_pickle("df.pkl").iloc[listind,:]
y = df.median_house_value.to_numpy()
X = df.drop(["median_house_value"],axis=1).to_numpy()

scaler1 = MinMaxScaler()
scaler2 = StandardScaler()
X = scaler2.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

mlp = MultiP(inputDim=X_train.shape[1],firstLayer=64,secondLayer=64,outputDim=1)

mlp.train(X_train,y_train,epochs=100,lr=0.01,dropout1=0.2,dropout2=0.2)  #eğitim yapıldı.

y_pred = np.zeros(len(y_test))
for k in range(len(X_test)):
        output = mlp.predict(X_test[k],dropout1=0.2,dropout2=0.2)  # tahmin edilen çıkış
        y_pred[k] = output


print(f"r2 score: {r2_score(y_test*(5*10**5),y_pred*(5*10**5))}")

plt.plot(range(200),y_test[:200]*(5*10**5),c='b', label = 'Gerçek Değerler')
plt.plot(range(200),y_pred[:200]*(5*10**5),c='r', label= 'Tahmin Edilen Değerler')
plt.legend()
plt.show()