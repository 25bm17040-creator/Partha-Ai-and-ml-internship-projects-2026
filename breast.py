import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("breast_cancer.csv")
label_encoder=LabelEncoder()
data["diagnosis_labelled"] = label_encoder.fit_transform(data["diagnosis"])

X = data [["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concavepoints_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concavepoints_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concavepoints_worst","symmetry_worst","fractal_dimension_worst"]]
Y = data[["diagnosis_labelled"]]

model= LogisticRegression()
model.fit(X,Y)
print("coeff:",model.coef_)
print("intercerpt:",model.intercept_)
prediction= model.predict([[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.1587,0.3003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]])
print("predicted cancer type:",prediction)
if prediction == 0:
    print("BENGIN")
else:
    print("MALIGANT")