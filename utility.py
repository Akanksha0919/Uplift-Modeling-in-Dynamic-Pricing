import os
import pandas as pd
import numpy as np
import re
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression # We will use sklearn here as it is required in the next algorithms
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from imblearn.combine import SMOTEENN
#grid search for hyperparameter tuning
from sklearn.ensemble import AdaBoostClassifier
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler
standardscaler = StandardScaler()
import matplotlib.pyplot as plt

logit = LogisticRegression(penalty='none', fit_intercept=True)
abc = AdaBoostClassifier(learning_rate = 0.05, n_estimators=100, random_state=0)

def evaluate_logit_ada_scaled_data(train_scale,y_train,test,y_test,train):
    fig, ax = plt.subplots(figsize=(15,5))

    # Create list of models for the loop
    models = [logit,abc] 

    # Loop to train and evaluate a model
    for i,model in enumerate(models): 
      ax = plt.subplot(1, 2, i+1)
      print('Model trained: {}'.format(type(model).__name__))
      model.fit(train_scale, y_train) 

      # Make prediction using the test set
      y_pred= model.predict_proba(test)[:,1]
      temp = model.predict(test)

          #acc = metrics.accuracy_score(Y_test, y_pred)

      fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
      auc = metrics.auc(fpr, tpr)

      min_freq = y_pred.sum()/len(y_pred) * 100

      print('The model predicted the minority class {:.2f}'.format(min_freq), '% of the time')
      print('The AUC is {:.2f}'.format(auc))
      print('The sensitivity for the model is {:.2f}'.format(tpr[1]))
      print('F1 score:', f1_score(y_test, temp))
      print("G-mean:", geometric_mean_score(np.ravel(y_test),temp))
      # Plot ROC curve
      metrics.RocCurveDisplay.from_estimator(model, train, y_train, ax=ax,name = 'Training set {}'.format(type(model).__name__)) 
      metrics.RocCurveDisplay.from_estimator(model, test, y_test, ax=ax,name = 'Test set {}'.format(type(model).__name__)) 

def evaluate_logit_ada_original_data(train,y_train,test,y_test):
    fig, ax = plt.subplots(figsize=(15,5))

    # Create list of models for the loop
    models = [logit,abc] 

    # Loop to train and evaluate a model
    for i,model in enumerate(models): 
      ax = plt.subplot(1, 2, i+1)
      print('Model trained: {}'.format(type(model).__name__))
      if format(type(model).__name__) == 'AdaBoostClassifier':
          print('in abc')
          ada = AdaBoostClassifier(learning_rate = 0.05, n_estimators=100, random_state=0)
          ada.fit(train, y_train)
          y_pred= ada.predict_proba(test)[:,1]
          temp = ada.predict(test)

          fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
          auc = metrics.auc(fpr, tpr)

          min_freq = y_pred.sum()/len(y_pred) * 100

          print('The model predicted the minority class {:.2f}'.format(min_freq), '% of the time')
          print('The AUC is {:.2f}'.format(auc))
          print('The sensitivity for the model is {:.2f}'.format(tpr[1]))
          print('F1 score:', f1_score(y_test, temp))
          print("G-mean:", geometric_mean_score(np.ravel(y_test),temp))

          metrics.RocCurveDisplay.from_estimator(ada, train, y_train, ax=ax,name = 'Training set {}'.format(type(model).__name__)) 
          metrics.RocCurveDisplay.from_estimator(ada, test, y_test, ax=ax,name = 'Test set {}'.format(type(model).__name__)) 
      else:
          print('in logit')
          model.fit(train, y_train)
          # Plot ROC curve
          y_pred= model.predict_proba(test)[:,1]
          temp = model.predict(test)

          #acc = metrics.accuracy_score(Y_test, y_pred)

          fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
          auc = metrics.auc(fpr, tpr)

          min_freq = y_pred.sum()/len(y_pred) * 100

          print('The model predicted the minority class {:.2f}'.format(min_freq), '% of the time')
          print('The AUC is {:.2f}'.format(auc))
          print('The sensitivity for the model is {:.2f}'.format(tpr[1]))
          print('F1 score:', f1_score(y_test, temp))
          print("G-mean:", geometric_mean_score(np.ravel(y_test),temp))

          metrics.RocCurveDisplay.from_estimator(model, train, y_train, ax=ax,name = 'Training set {}'.format(type(model).__name__)) 
          metrics.RocCurveDisplay.from_estimator(model, test, y_test, ax=ax,name = 'Test set {}'.format(type(model).__name__))

def evaluate_logit_ada_resample_data(train,y_train,test,y_test,sampling_strategy):
    fig, ax = plt.subplots(figsize=(15,5))

    # Create list of models for the loop
    models = [logit,abc] 
    resample = SMOTEENN(random_state=88, sampling_strategy=sampling_strategy)

    # Loop to train and evaluate a model
    for i,model in enumerate(models): 
      ax = plt.subplot(1, 2, i+1)
      print('Model trained: {}'.format(type(model).__name__))
      X_resampled, y_resampled = resample.fit_resample(train, np.ravel(y_train))
      if format(type(model).__name__) == 'AdaBoostClassifier':
          print('in abc')
          ada = AdaBoostClassifier(learning_rate = 0.05, n_estimators=100, random_state=0)
          ada.fit(X_resampled, y_resampled)
          y_pred= ada.predict_proba(test)[:,1]
          temp = ada.predict(test)

          #acc = metrics.accuracy_score(Y_test, y_pred)

          fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
          auc = metrics.auc(fpr, tpr)

          min_freq = y_pred.sum()/len(y_pred) * 100

          print('The model predicted the minority class {:.2f}'.format(min_freq), '% of the time')
          print('The AUC is {:.2f}'.format(auc))
          print('The sensitivity for the model is {:.2f}'.format(tpr[1]))
          print('F1 score:', f1_score(y_test, temp))
          print("G-mean:", geometric_mean_score(np.ravel(y_test),temp))

          metrics.RocCurveDisplay.from_estimator(ada, train, y_train, ax=ax,name = 'Training set {}'.format(type(model).__name__)) 
          metrics.RocCurveDisplay.from_estimator(ada, test, y_test, ax=ax,name = 'Test set {}'.format(type(model).__name__)) 
      else:
          print('in logit')
          model.fit(X_resampled, y_resampled)

          # Make prediction using the test set
          y_pred= model.predict_proba(test)[:,1]
          temp = model.predict(test)

            #acc = metrics.accuracy_score(Y_test, y_pred)

          fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
          auc = metrics.auc(fpr, tpr)

          min_freq = y_pred.sum()/len(y_pred) * 100

          print('The model predicted the minority class {:.2f}'.format(min_freq), '% of the time')
          print('The AUC is {:.2f}'.format(auc))
          print('The sensitivity for the model is {:.2f}'.format(tpr[1]))
          print('F1 score:', f1_score(y_test, temp))
          print("G-mean:", geometric_mean_score(np.ravel(y_test),temp))

          # Plot ROC curve
          metrics.RocCurveDisplay.from_estimator(model, train, y_train, ax=ax,name = 'Training set {}'.format(type(model).__name__)) 
          metrics.RocCurveDisplay.from_estimator(model, test, y_test, ax=ax,name = 'Test set {}'.format(type(model).__name__)) 