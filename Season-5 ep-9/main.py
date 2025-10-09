import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns

TRAIN_DF_FILEPATH = os.path.join("Data","train.csv")
TEST_DF_FILEPATH = os.path.join("Data","test.csv")

train_df = pd.read_csv(TRAIN_DF_FILEPATH)
train_df.set_index("id",inplace=True)
train_df.head()

