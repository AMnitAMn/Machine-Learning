# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 18:01:53 2019

@author: Arpit
"""

# Imporying the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Imorting Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

"""#Missing Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy='mean')
imputer = imputer.fit(X.iloc[:,1:3])
X.iloc[:,1:3] = imputer.transform(X.iloc[:, 1:3])

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()
X.iloc[:,0] = labelencoder_X.fit_transform(X.iloc[:,0]) 
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y) """

"""#Splitting the dataset into Training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, yTrain, yTest = train_test_split(X, y, test_size = 0.2 , random_state = 0)"""

"""
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""
#Fitting the regression model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

#Predicting a new result with Polynomial Regression
y_pred = regressor.predict([[6.5]])

m = min(X)

#Plotting the Regression Result (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X , y , color = 'red')
plt.plot(X_grid , regressor.predict(X_grid) , color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()