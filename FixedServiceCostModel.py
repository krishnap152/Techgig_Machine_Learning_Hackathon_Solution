# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:03:09 2018

@author: Krishna.Tiwari
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:54:26 2018

@author: Krishna.Tiwari
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 17:33:24 2018

@author: Krishna.Tiwari
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


# Importing the dataset
dataset = pd.read_csv('C:/Users/krishna.tiwari/Desktop/techgig/newVariant.csv')
X = dataset.drop("FixedServiceCost",axis=1)
X = X.drop("ProductModelNo",axis=1)
X = X.drop("Year",axis=1)
y = dataset["FixedServiceCost"]
y = y.astype('float32')
#Y = pd.DataFrame(y)
 
#dataset = dataset.drop("ProductModelNo",axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


######### Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_predlr = regressor.predict(X_test)

regressor.score(X_test, y_test)
# The coefficients
print('Coefficients: \n', regressor.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_predlr))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f', r2_score(y_test, y_predlr))

########## KNR
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X,y)
pred = knn.predict(X_test)

mse = (((pred-y_test)**2).sum())/len(pred)

print("Mean Square Error: ",mse)

print("Mean squared error: %.2f"
      % mean_squared_error(y_test, pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f', r2_score(y_test, pred))

########### XGBOOST
model = XGBRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_predd = regressor.predict(X_test)

print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_predd))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f', r2_score(y_test, y_predd))

# Visualising the Training set results
plt.scatter(X_train['RepairExpenditureDuringWarranty'], y_train, color = 'red')
plt.plot(X_train['RepairExpenditureDuringWarranty'], regressor.predict(X_train), color = 'blue')
plt.title('RepairExpenditureDuringWarranty vs Fixed Service Charge (Training set)')
plt.xlabel('Repair Expenditure During Warranty')
plt.ylabel('Fixed Service Charge')
plt.show()

# Visualising the Test set results
plt.scatter(X_test['RepairExpenditureDuringWarranty'], y_test, color = 'red')
plt.plot(X_train['RepairExpenditureDuringWarranty'], regressor.predict(X_train), color = 'blue')
plt.title('RepairExpenditureDuringWarranty vs Fixed Service Charge (Test set)')
plt.xlabel('Repair Expenditure During Warranty')
plt.ylabel('Fixed Service Charge')
plt.show()



