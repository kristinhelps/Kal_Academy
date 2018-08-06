#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 21:50:48 2018

@author: KristinHelps
"""

# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
housing = pd.read_csv('KC_House_Data.csv')
X = housing.iloc[:, 3:].values #eliminate id(0), date(2)
y = housing.iloc[:, 2:3].values #home price @ column 3/index 2

#Describe Dataset
housing.describe()

#Checking to make sure no values are missing from dataset
housing.info()

#Plotting Histograms
#housing.hist(bins=50, figsize = (20,15))

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Looking for correlations
corr_matrix = housing.corr()
corr_matrix["price"].sort_values(ascending=False)
#from pandas.plotting import scatter_matrix
#attributes = ["price", "sqft_living", "bedrooms",
#              "yr_built"]
#scatter_matrix(housing[attributes], figsize=(12, 8))

#Making a copy so I"m not adding a new category to the original dataset
housing_copy = housing.copy ()

#Does the ratio of bedrooms to bathroom affect the price
housing_copy["beds per bath"] = housing_copy["bedrooms"]/housing["bathrooms"]
corr_matrix = housing_copy.corr()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train_scaled = sc_y.fit_transform(y_train)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train_scaled, y_train_scaled )

#Evaluating the Training Set
# Predicting the Test set results
y_pred_scaled = regressor.predict(X_test_scaled)
y_pred = sc_y.inverse_transform(y_pred_scaled)

from sklearn.metrics import mean_squared_error
lin_mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(lin_mse)

#RMSE = 190,813.197...this means that linear regression is not very accurate for predicting house values for this dataset

import statsmodels.formula.api as sm
X_opt = np.append(arr = np.ones((len(X),1)).astype(int), values=X, axis = 1)

# Backward elmination

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

SL = 0.05
X_Modeled = backwardElimination(X_opt, SL)

#Backwords elmination only removed the column for 'floors' so at the SL = 0.05, all of the features have some influence over the price

#Re-running how good/bad is Linear Regression after backwards elimination

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X_Modeled, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
sc_y = StandardScaler()
y_train_scaled = sc_y.fit_transform(y_train)

# Fitting Multiple Linear Regression to the Training set

regressor = LinearRegression()
regressor.fit(X_train_scaled, y_train_scaled )

#Evaluating the Training Set
# Predicting the Test set results
y_pred_scaled = regressor.predict(X_test_scaled)
y_pred = sc_y.inverse_transform(y_pred_scaled)

lin_mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(lin_mse)

#Now RMSE is 215306.363...which is worse after backwards elmination 

#Polynomial Linear Regression 

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X_train_scaled)

regressor = LinearRegression()
regressor.fit(X_poly, y_train_scaled )

# Predicting a new result with Polynomial Regression
X_test_poly = poly_reg.transform(X_test_scaled)
y_pred_poly_scaled = regressor.predict(X_test_poly)
y_pred_poly = sc_y.inverse_transform(y_pred_poly_scaled)

poly_mse = mean_squared_error(y_test, y_pred_poly)
poly_rmse = np.sqrt(poly_mse)

#poly_rmse is ~7e15 on the test set but ~115k on the training set,
#so polynomial regression dramatically overfits the data

#SVR 

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train_scaled, y_train_scaled)

# Predicting a new result with SVR
y_pred_SVR_scaled = regressor.predict(X_test_scaled)
y_pred_SVR = sc_y.inverse_transform(y_pred_SVR_scaled)

SVR_mse = mean_squared_error(y_test, y_pred_SVR)
SVR_rmse = np.sqrt(SVR_mse)

#SVR_rsme is 202209.545...which is slightly better than Linear Regression

#Decision Tree Regression

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train_scaled, y_train_scaled)

# Predicting a new result with Decision Tree
y_pred_DecisionTree_scaled = regressor.predict(X_test_scaled)
y_pred_DecisionTree = sc_y.inverse_transform(y_pred_DecisionTree_scaled)

DecisionTree_mse = mean_squared_error(y_test, y_pred_DecisionTree)
DecisionTree_rmse = np.sqrt(DecisionTree_mse)

#DecisionTree_rmse is 250774.912...which is worse than SVR and Linear Regression

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train_scaled, y_train_scaled)

# Predicting a new result with Random Forest
y_pred_RandomForest_scaled = regressor.predict(X_test_scaled)
y_pred_RandomForest = sc_y.inverse_transform(y_pred_RandomForest_scaled)

RandomForest_mse = mean_squared_error(y_test, y_pred_RandomForest)
RandomForest_rmse = np.sqrt(RandomForest_mse)

#RandomForest_rmse is 195640.817 which is the most accurate yet...

# Fitting Logistic Regression to the Training set
#Don't know why this wouldn't run
"""from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(random_state = 0)
regressor.fit(X_train_scaled, y_train)

# Predicting a new result with Logistic Regression
y_pred_LogisticRegression = regressor.predict(X_test_scaled)

LogisticRegression_mse = mean_squared_error(y_test, y_pred_LogisticRegression)
LogisticRegression_rmse = np.sqrt(LogisticRegression_mse)"""


















"""plt.scatter(X_test[:,2], y_test, color = 'red')
plt.scatter(X_test[:,2], y_pred, color = 'blue')
plt.title("Testing Linear Regression Model")
plt.xlabel ("Multiple Variables")
plt.ylabel("Price")
plt.show()"""


# Encoding categorical data, #don't need this for this dataset
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3:7] = labelencoder.fit_transform(X[:, 3:7])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
"""
# Avoiding the Dummy Variable Trap
#X = X[:, 1:]




#to be able to calculate P-values
#import statsmodels.formula.api as sm

#need to add a new column with all 1's for b0(arr, values, axis) 21K rows, 1 column
#X = np.append(arr=np.ones((21,613, 1)).astype(int), values=X, axis=1)

#Backward elmination, : all rows, [which columns]
#X_opt = X[:, []]
#sm.OLS