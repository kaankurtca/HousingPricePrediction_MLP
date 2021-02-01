import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler,MinMaxScaler, RobustScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Input
from keras.optimizers import SGD
from keras.metrics import MeanSquaredError
from keras.optimizers import RMSprop, SGD
import keras

listind=np.random.choice(20000,size=4000,replace=False)
df = pd.read_pickle("df.pkl").iloc[:,:]
y = df.median_house_value
X = df.drop(["median_house_value"],axis=1)

scaler1 = MinMaxScaler()
scaler2 = StandardScaler()
scaler3 = RobustScaler()
X = scaler2.fit_transform(X)

# ee = EllipticEnvelope(contamination=0.01)
# yhat = ee.fit_predict(X)
# # select all rows that are not outliers
# mask = yhat != -1
# X, y = X[mask, :], y[mask]


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)



# # identify outliers in the training dataset
# iso = IsolationForest(contamination=0.1)
# yhat = iso.fit_predict(X_train)
# # select all rows that are not outliers
# mask = yhat != -1
# X_train, y_train = X_train[mask, :], y_train[mask]
# s=len(X_train)
# print(f-s)
# print(X_train.shape)




def buildModel():
    model = Sequential()
    model.add(Dense(128, input_dim=X.shape[1], activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))  # İki gizli katmanlı sinir ağı oluşturuldu.

    model.compile(optimizer=keras.optimizers.RMSprop(), loss='mse', metrics=['mse'])

    return model
model = buildModel()

print(model.summary())
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=100)
history = model.fit(X_train,y_train,batch_size=1024, epochs=500, validation_split=0.2, callbacks=[early_stop])

y_pred = model.predict(X_test)

print(f"r2 score: {r2_score(y_test*(5*10**5),y_pred*(5*10**5))}")
#print(f"{y_test[:10]}\n{y_pred[:10]}")

plt.plot(range(250),y_test[:250]*(5*10**5),c='b', label='Gerçek değerler')
plt.plot(range(250),y_pred[:250]*(5*10**5),c='r', label= 'Tahmin')
plt.legend()




hist = pd.DataFrame(history.history)
hist["epoch"]=history.epoch



plt.figure()
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.plot(hist['epoch'],hist['mse'],label='Train Loss')
plt.plot(hist['epoch'],hist['val_mse'],label='Validation Loss')
plt.legend()
plt.show()