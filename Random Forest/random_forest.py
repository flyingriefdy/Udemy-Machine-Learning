# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:55:15 2016

@author: hazie
"""

# Random Forest Intuition
# Step 1: Pick at random K data points from the Training set
# Step 2: Build the decision tree associated to these K data points
# Step 3: Choose the number NTrees of trees you want to build and repeat STEP 1 & 2
# Step 4: For a new data point, make each one of your Ntree treees predict the value 
# of Y to for the data point in question, and assign the new data point the average

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
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)'''

# Step 3: Fitting Random Forest Model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X,y)

# Step 4: Predicting a new result with Random Forest Regression
y_pred = regressor.predict(6.5)

# Step 5: Visualising the Random Forest results
# To increase resolution, use X_grid
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color ='blue')
plt.title('Level vs. Salary (Random Forest Regression Model)')
plt.xlabel('Level')
plt.ylabel('Salary, USD')
plt.show