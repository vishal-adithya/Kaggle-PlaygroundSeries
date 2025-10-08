import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,RandomizedSearchCV
import xgboost as xgb

TRAIN_DF_FILEPATH = os.path.join("Data","train.csv")
MODEL_FILEPATH = os.path.join("experiments","kps-s5-e10-reg-gbtree-rmse.json")

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

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,shuffle=False,random_state=47)
dtrain = xgb.DMatrix(data = X_train,label = y_train)
dval = xgb.DMatrix(data = X_val,label = y_val)

reg = xgb.XGBRegressor(random_state= 47,booster = "gbtree",early_stopping_rounds = 20)
param_grid = {
    "n_estimators":[100,300,500],
    "max_depth":[4,6,8],
    "learning_rate":[0.1,0.01,0.2],
    "colsample_bytree":[0.7,0.85,1.0],
    "subsample":[0.7,0.85,1.0]
}


rsv = RandomizedSearchCV(reg,
                            param_distributions=param_grid,
                            scoring = "neg_root_mean_squared_error",
                            verbose=4,
                            n_jobs=-1,
                            n_iter=5,
                            random_state=47,
                            cv = 10
                            )
rsv.fit(X_train,y_train,eval_set = [(X_train,y_train),(X_val,y_val)])
best_params = rsv.best_params_
print(rsv.best_score_)

reg = xgb.XGBRegressor(**best_params,random_state= 47,booster = "gbtree",early_stopping_rounds = 20)
reg.fit(X_train,y_train,eval_set = [(X_train,y_train),(X_val,y_val)])



def Plot_LearningCurves(model):
    hist = model.evals_result()
    
    t_rmse = hist["validation_0"]["rmse"]
    v_rmse = hist["validation_1"]["rmse"]
    
    epochs = np.arange(1,len(t_rmse)+1)
    plt.figure(figsize = (8,5))
    
    plt.plot(epochs,t_rmse,label = "Train RMSE")
    plt.plot(epochs,v_rmse,label = "Validation RMSE")
    
    plt.xlabel("No of Boosting Rounds")
    plt.ylabel("RMSE")
    plt.title("GbTree : Learning Curves")
    
    plt.legend()
    plt.grid(True)
    
    plt.show()

Plot_LearningCurves(reg)
reg.save_model(MODEL_FILEPATH)

model = xgb.XGBRegressor()
model.load_model(MODEL_FILEPATH)