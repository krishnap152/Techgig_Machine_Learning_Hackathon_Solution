# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:54:19 2018

@author: Krishna.Tiwari
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from numpy import loadtxt
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeRegressor
dataset = pd.read_csv('C:/Users/krishna.tiwari/Desktop/techgig/Training_data.csv')


X = dataset.drop("ProfitMargin",axis=1)
X = X.drop("ProductModelNo",axis=1)
X = X.drop("Year",axis=1)
y = dataset["ProfitMargin"]


############# PCA ###########################
pca = PCA(n_components=5)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s",fit.explained_variance_ratio_)
print(fit.components_)

############# XGBOOST ########################
model = XGBRegressor()
model.fit(X, y)
# feature importance
print(model.feature_importances_)
plot_importance(model)
plt.show()


############ Xtra Tree Classifier ####################3

model = DecisionTreeRegressor()
model.fit(X, y)
print(model.feature_importances_)