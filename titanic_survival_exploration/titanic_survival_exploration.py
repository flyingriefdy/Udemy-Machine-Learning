# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 13:53:37 2016

@author: hazie
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import visuals as vs



# importing the dataset
dataset = pd.read_csv('titanic_data.csv')

# sorting the dataset
outcomes = dataset['Survived']
data = dataset.drop('Survived',axis =1)

# survival stats
vs.survival_stats(data,outcomes,'Age')

# accuracy method
def accuracy_score(truth, pred):
    if len(truth) == len(pred):
        return "Predictions have an accuracy of {:.2f}%".format((truth == pred).mean()*100)
    else:
        return "Number of predictions does not match number of outcomes"
        
#predictions = pd.Series(np.ones(5, dtype = int))
#print (accuracy_score(outcomes[:5], predictions))

def predictions_0(data):
    predictions = []
    for _, passenger in data.iterrows():
        survive = 0
        predictions.append(survive)
    
    return pd.Series(predictions)
    




def predictions_1(data):
    
    predictions = []
    for _, passenger in data.iterrows():
        survive = 0
        if passenger['Sex'] == 'female' or passenger['Age'] < 10 :
            survive = 1
        
        predictions.append(survive)
        
    return pd.Series(predictions)
    
    
def predictions_2(data):
    
    predictions = []
    for _, passenger in data.iterrows():
        survive = 0
        if passenger['Sex'] == 'female' or passenger['Age'] < 10 :
            survive = 1
        
        predictions.append(survive)
        
    return pd.Series(predictions)    
    
# Make the predictions
predictions = predictions_2(data)
print (accuracy_score(outcomes,predictions))
    
    
        
        
        
        
        
        
        
        
        