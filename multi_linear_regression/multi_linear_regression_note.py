# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 17:19:04 2016

@author: hazie
"""

# CHAPTER MULTI LINEAR REGRESSION
# Building a model
# 2. Backward elimination
# Step 1 - select a signifiance level to stay in the model (e.g., 0.05)
# Step 2 - Fit the full model with all possible predictors
# Step 3 - Consider the predictor with the highest P-value. If P>SL, go Step 4
# Step 4 - Remove the predictor
# Step 5 - Fit the model without this variable. Return to Step 3 until P<SL

# 3. Forward selection
# Step 1 - select a signifiance level to stay in the model
# Step 2 - fit all simple regression models y~xn. Select the one with lowest P-value
# Step 3 - Keep this variable and fit all possible models with one extra predictor 
#          to the one(s) you already have
# Step 4 - Consider the predictor with the lowest P-value. If P<SL, go to step 3, otherwise FIN
#
# 4. Birectional elimination
# Step 1 - Select significance level
# Step 2 - Perform the next step of Forward Selection 
#           new variables must have P<SLenter to enter
# Step 3 - Perform ALL steps of Backward elimination
#           old variables must have P<SLstay to stay
# Step 4 - No new variables can enter and no old variables can exit
#
# 5. All possible models
# Step 1 - Select criterion of goodness fit e.g. Akaike criterion
# Step 2 - Construct all possible regression models: 2^N-1 total combinations
# Step 3 - Select the one with best criterion
#
#
#
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# What if one of the predictor are more statically significant variables?
# Now we shall find the team of strong predictor and the weaker predictor
# The effects can be positive or negative i.e., one unit increase in 
# predictor to increase/decrease in dependent variable unit

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
# statsmodel does not take account of b0 in y = b0 + b1X1 + ... bnXn
# add column corresponding to b0*X0 where X0 assumed to be 1
# astype(int) to prevent datatype error
# axis = 1 for column axis = 0 for row
X = np.append(arr = np.ones((50,1)).astype(int),values = X,axis = 1)
# X_opt will only contain statistical significant predictor
'''X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Check P value vs SL
regressor_OLS.summary()
# Remove 2
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# Remove 1
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# Remove 4
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()












