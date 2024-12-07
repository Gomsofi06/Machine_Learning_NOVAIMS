import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils import *

train_df = pd.read_csv('preprocessed_data/train_data.csv', index_col="Claim Identifier")

X = train_df.drop(["Claim Injury Type Encoded"], axis = 1)
y = train_df["Claim Injury Type Encoded"]

numerical_features = [
    'Age at Injury','Average Weekly Wage','Birth Year','IME-4 Count',
    'Number of Dependents','Days_to_First_Hearing','Days_to_C2','Days_to_C3',
    'Accident_Season_Sin','Accident_Season_Cos',
    'C-2 Date_Year','C-2 Date_Month','C-2 Date_Day','C-2 Date_DayOfWeek',
    'C-3 Date_Year','C-3 Date_Month','C-3 Date_Day', 'C-3 Date_DayOfWeek',
    'First Hearing Date_Year','First Hearing Date_Month','First Hearing Date_Day','First Hearing Date_DayOfWeek'
]

n_folds = 5
for random_state in [42,69]:
    fold=1
    K_fold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for train_index, val_index in K_fold.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        NA_imputer(X_train,y_train)
        
        create_new_features(X_train,y_train)

        st = StandardScaler()
        X_train[numerical_features] = st.fit_transform(X_train[numerical_features])
        X_val[numerical_features] = st.transform(X_val[numerical_features])

        X_train.to_csv(f"./Kfolds_csv/X_train_{fold}_{random_state}.csv")
        X_val.to_csv(f"./Kfolds_csv/X_val_{fold}_{random_state}.csv")

        # folds 1 to 5
        # random_state 42 and 69

