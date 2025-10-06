import pandas as pd
import numpy as np
import os

TRAIN_DF_FILEPATH = os.path.join("Data","train.csv")


train_df = pd.read_csv(TRAIN_DF_FILEPATH)
train_df.set_index("id",inplace=True)
train_df.head()