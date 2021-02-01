import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df = pd.read_csv("housing.csv")

df1 = df[df.ocean_proximity != "ISLAND"]; df1=df
df2 =  pd.get_dummies(df1.ocean_proximity, prefix='ocean_proximity')
df1 = df1.drop(["ocean_proximity"],axis=1); df_encoded = df1
df_encoded = pd.concat([df1,df2],axis=1)

df_encoded["Bedrooom_PerHouseHold"] = df_encoded["total_bedrooms"] / df_encoded["households"]
df_encoded.drop(columns=["total_bedrooms"],inplace=True)

df_encoded["Room_PerHouseHold"] = df_encoded["total_rooms"] / df_encoded["households"]
df_encoded.drop(columns=["total_rooms"],inplace=True)

df_encoded["People_PerHouseHold"] = df_encoded["population"] / df_encoded["households"]
df_encoded.drop(columns=["population","households"],inplace=True)

df_encoded.median_house_value = df_encoded.median_house_value / (5*10**5)
df_encoded.dropna(axis=0,inplace=True)

df_encoded.to_pickle("df.pkl")



# dataset = df_encoded.to_numpy()
#
# np.save("C:/Users/osman/PycharmProjects/YSA_TermProject/dataset.npy",dataset)