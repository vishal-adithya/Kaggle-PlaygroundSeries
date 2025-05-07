#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 18:43:57 2025

@author: vishaladithyaa
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

TRAIN_DF_FILEPATH = os.path.join("Data","train.csv")
TEST_DF_FILEPATH = os.path.join("Data","test.csv")

train_df = pd.read_csv(TRAIN_DF_FILEPATH)

def Preprocessing(df):
    df = df.set_index(df["id"])
    df.drop(columns = ["id"],inplace = True)
    df["Sex"] = df["Sex"].map(lambda x:0 if x=="male" else 1)
    df["BMI"] = round(df["Weight"]/((df["Height"]/100)**2),2)
    return df
    
preprocessed_df = Preprocessing(train_df)
preprocessed_df.head()

sample = preprocessed_df.sample(10000)

sns.lineplot(data = sample,x = "Heart_Rate",y = "Calories")
plt.title("Calories X Heart Rate")
plt.show()

sns.lineplot(data = sample,x = "Duration",y = "Calories")
plt.title("Calories X Duration")
plt.show()

sns.lineplot(data = sample,x = "BMI",y = "Calories")
plt.title("Calories X BMI")
plt.show()

sns.barplot(data = sample,x="Sex",y = "Calories")
plt.title("Calories X Sex")
plt.show()




print(preprocessed_df[['Calories',"Duration","Heart_Rate","BMI","Sex","Age","Height","Weight"]].corr())
