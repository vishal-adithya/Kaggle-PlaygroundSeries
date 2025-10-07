import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb

TRAIN_DF_FILEPATH = os.path.join("Data","train.csv")


train_df = pd.read_csv(TRAIN_DF_FILEPATH)
train_df.set_index("id",inplace=True)
train_df.head()

print(train_df["road_type"].unique())
print(train_df["lighting"].unique())
print(train_df["weather"].unique())
print(train_df["time_of_day"].unique())

train_df.isnull().sum()

def Preprocessing(df):
    df = df.copy()
    dummie_1 = pd.get_dummies(df["road_type"],dtype = "float")
    dummie_2 = pd.get_dummies(df["lighting"],dtype = "float") 
    dummie_3 = pd.get_dummies(df["weather"],dtype = "float")
    dummie_4 = pd.get_dummies(df["time_of_day"],dtype = "float")
    concat_df = pd.concat([dummie_1,dummie_2,dummie_3,dummie_4,df],axis = 1)
    concat_df.drop(columns = ["road_type","lighting","weather","time_of_day"],inplace = True)

    concat_df["road_signs_present"] = concat_df["road_signs_present"].map(lambda x: 0.0 if x == False else 1.0)
    concat_df["public_road"] = concat_df["public_road"].map(lambda x: 0.0 if x == False else 1.0)
    concat_df["holiday"] = concat_df["holiday"].map(lambda x: 0.0 if x == False else 1.0)
    concat_df["school_season"] = concat_df["school_season"].map(lambda x: 0.0 if x == False else 1.0)

    concat_df["num_lanes"] = concat_df["num_lanes"].astype("float")
    concat_df["speed_limit"] = concat_df["speed_limit"].astype("float")
    concat_df["num_reported_accidents"] = concat_df["num_reported_accidents"].astype("float")
    return concat_df

preprocessed_train_df = Preprocessing(train_df)

preprocessed_train_df.corr()

sns.histplot(preprocessed_train_df["curvature"],kde = True)
plt.show()

sns.histplot(preprocessed_train_df["speed_limit"],kde = True)
plt.show()

sns.histplot(preprocessed_train_df["num_reported_accidents"],kde = True)
plt.show()

preprocessed_train_df.head()

X = preprocessed_train_df.drop(columns = ["accident_risk"])
y = preprocessed_train_df["accident_risk"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=False,random_state=47)
dtrain = xgb.DMatrix(data = X_train,label = y_train)

reg = xgb.XGBRegressor(random_state= 47,booster = "gblinear")
param_grid = {
    "n_estimators":[100,300,500],
    "max_depth":[4,6,8],
    "learning_rate":[0.1,0.01,0.2],
    "colsample_bytree":[0.7,0.85,1.0],
    "subsample":[0.7,0.85,1.0]
}

