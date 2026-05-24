import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

data = pd.read_csv("knn_data.csv")
x = data[["Temperature"]]
y = data[["Fuel_consumption"]]

model = KNeighborsRegressor(n_neighbors=3)
model.fit(x,y)

print("the fuel consumption:",model.predict([[58]]))