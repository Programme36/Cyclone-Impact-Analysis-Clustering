import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression


origenal_datafream = pd.read_csv("cyclones.csv")
df = origenal_datafream.copy()

df2 = pd.DataFrame(df,columns=["pressure","speed"])
scale = StandardScaler()
df2 = scale.fit_transform(df2)

df["pressure"] = df2[:,0]
df["speed"] = df2[:,1]

kmeans = KMeans(n_clusters=5 ,random_state=46 ,n_init=45)
X = df[["pressure","speed"]]
df["group"] = kmeans.fit_predict(X)

predicted = df.groupby("group")["group"].count()
true = df.groupby("category_name")["category_name"].count()


print(true)
print(predicted)



