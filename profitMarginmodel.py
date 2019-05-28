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
import seaborn as sns


# Importing the dataset
dataset = pd.read_csv('C:/Users/krishna.tiwari/Desktop/techgig/newVariant.csv')
f, ax = plt.subplots(figsize=(15, 15))
#dataset.hist(ax = ax)
#plt.show()

from pandas.plotting import scatter_matrix
scatter_matrix(dataset,ax=ax)
plt.show()


dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False,ax=ax)
plt.show()
names = dataset.columns
plt.matshow(dataset.corr())
corr = dataset.corr()
f, ax = plt.subplots(figsize=(15, 15))
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


X = dataset.drop("ProfitMargin",axis=1)
X = X.drop("ProductModelNo",axis=1)
X = X.drop("Year",axis=1)
y = dataset["ProfitMargin"]
y = y.astype('float32')


#dataset = dataset.drop("ProductModelNo",axis=1)
#visualize the relationship between the features and the response using scatterplots
sns.pairplot(dataset, x_vars=['ManufacturingCost','SellingCost','MarketShare','ClaimsMadeByCustInWarranty','ServiceCenter','RepairExpenditureDuringWarranty'], y_vars='ProfitMargin', size=7, aspect=0.7)

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

print("Mean Square Error",mse)
print('Variance score: %.2f', r2_score(y_test, pred))

########### XGBOOST
model = XGBRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = (((y_pred-y_test)**2).sum())/len(y_pred)

print("Mean Square Error",mse)
print('Variance score: %.2f', r2_score(y_test, y_pred))

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_predd = regressor.predict(X_test)


# Visualising the Training set results
plt.scatter(X_train['ManufacturingCost'], y_train, color = 'red')
plt.plot(X_train['ManufacturingCost'], regressor.predict(X_train), color = 'blue')
plt.title('Manufacturing Cost vs Profit Margin (Training set)')
plt.xlabel('Manufacturing Cost')
plt.ylabel('Profit Margin')
plt.show()

# Visualising the Test set results
plt.scatter(X_test['ManufacturingCost'], y_test, color = 'red')
plt.plot(X_train['ManufacturingCost'], regressor.predict(X_train), color = 'blue')
plt.title('Manufacturing Cost vs Profit Margin (Test set)')
plt.xlabel('Manufacturing Cost')
plt.ylabel('Profit Margin')
plt.show()



