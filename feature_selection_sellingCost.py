# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:27:26 2018

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


X = dataset.drop("SellingCost",axis=1)
X = X.drop("ProductModelNo",axis=1)
X = X.drop("Year",axis=1)
y = dataset["SellingCost"]


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