import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from MLP_Class_FromScratch import MultiP

listind=np.random.choice(20000,size=1000,replace=False)
df = pd.read_pickle("df.pkl").iloc[listind,:]
y = df.median_house_value.to_numpy()
X = df.drop(["median_house_value"],axis=1).to_numpy()

scaler1 = MinMaxScaler()
scaler2 = StandardScaler()




iter = 5
mse, r2 = [],[]
for i in range(iter):
        mlp = MultiP(inputDim=X.shape[1], firstLayer=64, secondLayer=64, outputDim=1)

        X = scaler2.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        mlp.train(X_train,y_train,epochs=100,lr=0.1,dropout1=0.2,dropout2=0.2)  #eğitim yapıldı.

        y_pred = np.zeros(len(y_test))
        for k in range(len(X_test)):
                output = mlp.predict(X_test[k],dropout1=0.2,dropout2=0.2)  # tahmin edilen çıkış
                y_pred[k] = output
        r2.append(r2_score(y_test, y_pred))
        mse.append(mean_squared_error(y_test, y_pred))

# print(f"\n\nr2 score: {r2_score(y_test,y_pred)}")
# print(f"mean squared error: {mean_squared_error(y_test,y_pred)}")
print(f"\n\nr2 score: {sum(r2) / len(r2)}")
print(f"mean squared error: {sum(mse) / len(mse)}")

plt.plot(range(100),y_test[:100]*(5*10**5),c='b', label = 'Gerçek Değerler')
plt.plot(range(100),y_pred[:100]*(5*10**5),c='r', label= 'Tahmin Edilen Değerler')
plt.legend()
plt.show()