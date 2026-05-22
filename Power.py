import pandas as pd
data = pd.read_csv("power_data.csv")
from sklearn.linear_model import LinearRegression

x= data[["wind_speed","blade_angle","rotor_speed"]]
y= data[["power_output"]]

model= LinearRegression()
model.fit(x,y)

print("coefficient:",model.coef_)
print("intercept:",model.intercept_)
print("the Required power output :",model.predict([[9,11,100]]))
