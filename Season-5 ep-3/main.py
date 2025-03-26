#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 13:33:05 2025

@author: vishaladithya
"""

# dependencies

import os
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
from sklearn.metrics import classification_report

# loading the data
TRAIN_CSV_FILEPATH = os.path.join("Data","train.csv")
TEST_CSV_FILEPATH = os.path.join("Data","test.csv")
SAMPLE_SUB_FILEPATH = os.path.join("Data","sample_submission.csv")
SUB_FILEPATH1 = os.path.join("Submissions","so-1.csv")
SUB_FILEPATH2 = os.path.join("Submissions","so-2.csv")
SUB_FILEPATH3 = os.path.join("Submissions","so-3.csv")
SUB_FILEPATH4 = os.path.join("Submissions","so-4.csv")



train_df = pd.read_csv(TRAIN_CSV_FILEPATH)
train_df.isna().sum()

def Preprocessing(data):
    data.set_index(data["id"],inplace = True)
    data.drop(columns = ["id","day"],inplace = True)
    return data

train_df = Preprocessing(train_df)

X = train_df.drop(columns = ["rainfall"]).values
y = train_df["rainfall"].values

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.1,random_state=44,shuffle=True)

#Training the model


est = xgb.XGBClassifier(eval_metric = "auc",
                        verbosity = 2,
                        n_jobs = 1,
                        booster = "gblinear")

param_grid = {"learning_rate":[0.1,0.01,0.2],
              "n_estimators":[2000,1000,3000],
              "max_depth":[6,8,10],
              "colsample_bytree":[0.5,0.7,0.9]}

rsv = GridSearchCV(est, param_grid,verbose=2,cv=10,scoring="roc_auc")
rsv.fit(X,y)

best_est = rsv.best_estimator_

y_hat = best_est.predict(X_val)

print(classification_report(y_val, y_hat))

test_df = pd.read_csv(TEST_CSV_FILEPATH)
preprocessed_test_df = Preprocessing(test_df)

X_test = preprocessed_test_df.values
y_pred = best_est.predict(X_test)
sub = pd.read_csv(SAMPLE_SUB_FILEPATH)
sub["rainfall"] = y_pred
sub.to_csv(SUB_FILEPATH4,index = False)

rsv.best_params_
