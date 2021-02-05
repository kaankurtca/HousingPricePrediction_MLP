import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input
from keras.optimizers import SGD
from keras.metrics import MeanSquaredError
from keras.optimizers import RMSprop, SGD
import keras

listind = np.random.choice(20000, size=4000, replace=False)
df = pd.read_pickle("df.pkl").iloc[:, :]
y = df.median_house_value
X = df.drop(["median_house_value"], axis=1)

scaler1 = MinMaxScaler()
scaler2 = StandardScaler()
scaler3 = RobustScaler()


# X = scaler2.fit_transform(X)


def buildModel():
    model = Sequential()
    model.add(Dense(128, input_dim=X.shape[1], activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='relu'))  # İki gizli katmanlı sinir ağı oluşturuldu.

    model.compile(optimizer=keras.optimizers.RMSprop(), loss='mse', metrics=['mse'])

    return model



iter = 3
mse, r2 = [],[]
for i in range(iter):
    model = buildModel()
    X = scaler2.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # X_train = scaler2.fit_transform(X_train)
    # X_test = scaler2.transform(X_test)

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
    history = model.fit(X_train, y_train, batch_size=1024, epochs=500, validation_split=0.2, callbacks=[early_stop])

    y_pred = model.predict(X_test)
    r2.append(r2_score(y_test, y_pred))
    mse.append(mean_squared_error(y_test, y_pred))
    # print(f"\n\nr2 score: {r2_score(y_test,y_pred)}")
    # print(f"mean squared error: {mean_squared_error(y_test,y_pred)}")

print(f"\n\nr2 score: {sum(r2) / len(r2)}")
print(f"mean squared error: {sum(mse) / len(mse)}")

plt.plot(range(50), y_test[:50] * (5 * 10 ** 5), c='b', label='Gerçek değerler')
plt.plot(range(50), y_pred[:50] * (5 * 10 ** 5), c='r', label='Tahmin')
plt.legend()

hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.plot(hist['epoch'], hist['mse'], label='Train Loss')
plt.plot(hist['epoch'], hist['val_mse'], label='Validation Loss')
plt.legend()

filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))

plt.show()
