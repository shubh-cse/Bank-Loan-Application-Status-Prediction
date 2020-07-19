# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 12:28:02 2020

@author: Shubh Gupta
"""

#importing libraries
import numpy as np
import pandas as pd
import pickle

#loading the dataset
df = pd.read_csv('bank-loan.csv')
df.shape

#missing values for the dataset
missing_value=pd.DataFrame(df.isnull().sum()).rename(columns={0:'count'})
missing_value

#creating new dataframe with 700 observations
df_new = pd.DataFrame(df[:700])
df_new.shape

#model building
from sklearn.model_selection import train_test_split
X = df_new.drop(columns='default')
y = df_new['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#applying random forest on the imbalanced dataset
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20, class_weight='balanced')
classifier.fit(X_train, y_train)

#creating pickle file for the classifier
filename = 'loan-predictor.pkl'
pickle.dump(classifier, open(filename, 'wb'))