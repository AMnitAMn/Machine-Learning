# Data preprocessing


# Imporying the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Imorting Dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,3]

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

#Splitting the dataset into Training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, yTrain, yTest = train_test_split(X, y, test_size = 0.2 , random_state = 0)

"""
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""
