import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("machine_failure.csv")
label_encoder = LabelEncoder()
data ["Type_encoded"]= label_encoder.fit_transform(data["Type"])

x= data [["Airtemperature","Processtemperature","Rotationalspeed","Torque","Toolwear","TWF","HDF","PWF","OSF","RNF","Type_encoded"]]
y= data [["Machinefailure"]]

model=LogisticRegression()
model.fit(x,y)
print("coeff:",model.coef_)
print("intercept:",model.intercept_)
prediction = model.predict([[301,311,1450,52,28,1,0,0,0,0,2]])
print("Predicted Machine Failure: ", prediction)

if prediction == 0:
    print("NO FAILURE")
else:
    print("FAILURE")