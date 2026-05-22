import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("power_data.csv")
x = data[["wind_speed","blade_angle","rotor_speed"]]
y =data[["power_output"]]

model=LinearRegression()
model.fit(x,y)

windspeed  = int(input("wind_speed"))
bladeangle = int(input("blade_angle"))
rotorspeed = int(input("rotor_speed"))

print("Coefficient:",model.coef_)
print("Intercept:",model.intercept_)
print("Required power output: ",model.predict([[windspeed,bladeangle,rotorspeed]]))