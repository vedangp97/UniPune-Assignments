import numpy as np
import pandas as pd
from math import sqrt

data = pd.read_csv("data.csv")
x = data.iloc[:,:-1].values    #values
y = data.iloc[:,1].values

from sklearn.model_selection import train_test_split #moddel
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size = 1/3,random_state = 1)#test_size,random_state

from sklearn.linear_model import LinearRegression as lr #linear
regression = lr()
regression = regression.fit(X_train,Y_train)
Y_pred = regression.predict(X_test)

from sklearn.metrics import r2_score,mean_squared_error#r2_score

print("Accuracy: ",r2_score(Y_test,Y_pred))

print(regression.coef_,"*x+",regression.intercept_)

rmse = sqrt(mean_squared_error(Y_test,Y_pred))
print("Root mean Squared error: ",rmse)

inputval = input("Enter no of hours: ")
inputval = np.array(inputval,dtype = np.float64).reshape(-1,1)#dtype
print("Risk = ",regression.predict(inputval))

import matplotlib.pyplot as plt
plt.scatter(X_train,Y_train,color = "black")
plt.plot(X_test,Y_pred,color="blue")
plt.title("Train data")
plt.xlabel("No of hours")
plt.ylabel("Risk")
plt.show()