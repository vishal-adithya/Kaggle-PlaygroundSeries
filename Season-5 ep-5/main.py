#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 18:43:57 2025

@author: vishaladithyaa
"""
import pandas as pd
import numpy as np
import os

TRAIN_DF_FILEPATH = os.path.join("Data","train.csv")
TEST_DF_FILEPATH = os.path.join("Data","test.csv")

train_df = pd.read_csv(TRAIN_DF_FILEPATH)

def Preprocessing(df):
    df = df.set_index(df["id"])
    df["Sex"] = df["Sex"].map(lambda x:0 if x=="male" else 1)
    return df
    
preprocessed_df = Preprocessing(train_df)
