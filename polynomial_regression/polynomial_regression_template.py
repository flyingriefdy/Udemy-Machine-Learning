# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 07:20:31 2016

@author: hazie
"""

# Polynomial Regression
# Step 1: Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 2: Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Step 3: Fitting Regression Model to the dataset

# Step 4: Predicting a new result with Polynomial Regression
y_pred = regressor.predict(6.5)

# Step 5: Visualising the Polynomial Regression results
# To increase resolution, use X_grid
# X_grid = np.arange(min(X),max(X),0.1)
# X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X_grid,y,color = 'red')
plt.plot(X_grid,regressor.predict(X),color ='blue')
plt.title('Level vs. Salary (Regression Model)')
plt.xlabel('Level')
plt.ylabel('Salary, USD')
plt.show
