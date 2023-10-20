# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the required libraries and read the dataframe.
2.Assign hours to X and scores to Y.
3.Implement training set and test set of the dataframe.
4.Plot the required graph both for test data and training data.
5.Find the values of MSE , MAE and RMSE. 

## Program:
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: R Guruprasad
RegisterNumber:  212222240033
*/
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

df.head()
df.tail()
#segregating data to variables
Array value of x 
X=df.iloc[:,:-1].values
print(X)
Array value of y
Y=df.iloc[:,-1].values
print(Y)

#graph plot for training data

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.model_selection import train_test_split
#print(X_train,X_test,Y_train,Y_test)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Values of y prediction
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)

#graph plot for test data
Training set graph
plt.scatter(X_train,Y_train,color="green")
plt.plot(X_train,reg.predict(X_train),color="red")
plt.title("Training set(H vs S)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
Test set graph
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,reg.predict(X_test),color="silver")
plt.title("Test set(H vs S)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

Values of MSE,MAE and RMSE
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

a=np.array([[10]])
Y_pred1=reg.predict(a)
print(Y_pred1)
```

## Output:
![image](https://github.com/R-Guruprasad/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390308/7b8c5214-c5cf-4e4e-b916-6747faf2cf29)
![image](https://github.com/R-Guruprasad/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390308/e59bc550-bffd-4c80-b59a-cc0e8de2ddca)
![image](https://github.com/R-Guruprasad/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390308/afd824b7-ce57-431f-8bd4-fa95b56638d9)
![image](https://github.com/R-Guruprasad/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390308/c4a24cbd-bf99-4c28-92dd-26fade4204c2)
![image](https://github.com/R-Guruprasad/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390308/f42cb9c8-13b5-4fd3-862e-2f2cade90995)
![image](https://github.com/R-Guruprasad/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390308/7ddcabee-6431-4abb-8a80-51c5ac3eda3e)
![image](https://github.com/R-Guruprasad/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390308/64b476a3-cdc9-47a7-8de4-0a41dbec1dbd)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
