# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 14:35:10 2016

@author: hazie
"""

# SIMPLE LINEAR REGRESSION

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the Test set results
# create a vector that will containg the prediction of test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train,y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title('Salary vs. Experience (Training set)')
plt.xlabel('Experience, years')
plt.ylabel('Salary, USD')
plt.show()

# Visualising the Test set results
plt.scatter(X_test,y_test,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title('Salary vs. Experience (Test set)')
plt.xlabel('Experience, years')
plt.ylabel('Salary, USD')
plt.show()



