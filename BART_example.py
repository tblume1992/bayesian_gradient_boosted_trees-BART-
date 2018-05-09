# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 13:42:06 2018

@author: t-blu
"""
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
import pymc3 as pm
boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * .9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]
weights = np.full((200), .1)
weights[0] = 1
xi = X_train
yi = y_train


# x,y --> use where no need to change original y
#Creating a bunch of initial values that we need
ei = 0 # initialization of error
predf = 0 # initial prediction 0
count = 1
adap = 1 #This constant can be used to help with the convergence of the dynamic shrinkage (if needed).  It simply multiplies the calculated learning rate.
testme = 0
rmse_list_adap = [] #Creating the rmse data set
sumei = 0
learning_rates = [] #Creating the learning rates data set
lr = .01#Setting the Learning rate, should be set to 1 when doing the dynamic shrinkage
mcmc_learning_rates = pd.DataFrame([])
for i in range(100): # loop will make n trees (n_estimators).
    

    tree = DecisionTreeRegressor(max_depth = 2,min_samples_leaf = 1) 
    tree.fit(xi,yi)
    predi =  tree.predict(xi)
    predicted =  tree.predict(X_test)
    if count > 2:
        data = dict(x=ei, y=predf)
        with pm.Model() as model:

             pm.glm.GLM.from_formula('y ~ x', data)
             trace = pm.sample(1000,tune = 1000, cores=2)
        mcmc_learning_rates = pd.concat([mcmc_learning_rates,pm.trace_to_dataframe(trace).iloc[:,1].tail(500)], ignore_index = True, axis = 1)
        lr = np.mean(pm.trace_to_dataframe(trace).iloc[:,1].tail(500))
        
    #The first model is just the basic tree, subsequent models will be based on the residuals
    if count < 2:
        predf = np.mean(yi)
    else:
        predf = predf + adap*lr*predi  # final prediction will be previous prediction value + new prediction of residual
    if count < 2:
        testme = np.mean(y_test)
    else:
        testme = testme + lr*predicted
    ei = y_train - predf  # needed originl y here as residual always from original y    
    yi = ei # update yi as residual to reloop

    sumei = np.sum(ei**2) # Calculate the sum of squared errors
    count = count + 1
    learning_rates.append(lr)
    rmse = (np.mean((y_test - testme)**2)) #Calculate RMSE
    rmse_list_adap.append(rmse)



#List of the RMSE's for each iteration
rmse_list_adap = pd.DataFrame(rmse_list_adap)
#List of the learning rates for each iteration
learning_rates = pd.DataFrame(learning_rates)


xi = X_train
yi = y_train

from sklearn.ensemble import GradientBoostingRegressor
n_est = 250
rnd_reg = GradientBoostingRegressor(max_depth=2, n_estimators=n_est, subsample = .9, learning_rate=.01)
rnd_reg.fit(xi,yi)
predictions = rnd_reg.predict(X_test)
test_score = np.zeros((n_est,), dtype=np.float64)
for i, y_pred in enumerate(rnd_reg.staged_predict(X_test)):
    test_score[i] = (np.mean((y_test - y_pred)**2))

    

plt.plot(test_score, label = 'Sklearn GBT')
plt.plot(rmse_list_adap, label = 'Bayesian Gradient Boosting')
plt.legend()
plt.show()
values = learning_rates.index.values
appendme = pd.DataFrame([.01,.01])
max_mcmc = (pd.DataFrame(np.max(mcmc_learning_rates, axis = 0)))
max_mcmc = appendme.append(max_mcmc, ignore_index = True)

appendme = pd.DataFrame([.01,.01])
min_mcmc = (pd.DataFrame(np.min(mcmc_learning_rates, axis = 0)))
min_mcmc = appendme.append(min_mcmc, ignore_index = True)

plt.plot(learning_rates, label = 'Expected Learning Rate')
plt.fill_between(np.squeeze(pd.DataFrame(values)),np.squeeze(max_mcmc),np.squeeze(min_mcmc), alpha = .2, color = 'grey')
