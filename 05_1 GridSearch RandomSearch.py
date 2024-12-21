# Import libraries
import pandas as pd
import numpy as np

#make the split here
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import time

from utils import *
from utils_feature_selection import check_performace
from utils_dicts import *

import warnings
warnings.filterwarnings('ignore')

random_state=68+1
start = start_time = time.time()

# Load the data
train_df = pd.read_csv("./preprocessed_data/train_data.csv", index_col="Claim Identifier")

param_grids = {
    "CatBoostClassifier": {
        "iterations": [300, 500, 800],
        "depth": [3, 6, 9],
        "learning_rate": [0.03, 0.05, 0.1],
        "l2_leaf_reg": [3, 6, 9],
    },
    "XGBClassifier": {
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.03, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.5, 1],
    },
    "DecisionTreeClassifier": {
        "max_depth": [None, 10, 20, 30],
        "criterion": ["gini", "entropy", "log_loss"],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
}

feature_selection = essential_features

X = train_df.drop(["Claim Injury Type Encoded"], axis = 1)
y = train_df["Claim Injury Type Encoded"]

# Define models
models = {
    "CatBoostClassifier": CatBoostClassifier(verbose=0, random_state=random_state,custom_metric='F1'),
    "XGBClassifier": XGBClassifier(objective="multi:softmax", num_class=8, eval_metric="merror", random_state=random_state),
    #"DecisionTreeClassifier": DecisionTreeClassifier(random_state=random_state),
}

# Split the data
random_state = 68+1
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.2, stratify = y, shuffle = True, random_state=random_state)

# Preprocess the data
remove_outliers(X_train)
X_train, X_val = apply_frequency_encoding(X_train, X_val)
NA_imputer(X_train,X_val)
create_new_features(X_train,X_val)

#Standardize the data
scaler_train = StandardScaler()
X_train[numerical_features] = scaler_train.fit_transform(X_train[numerical_features])
X_val[numerical_features] = scaler_train.transform(X_val[numerical_features])

drop_list = []
if feature_selection != None:
    for col in X_train.columns:
        if col not in feature_selection:
            drop_list.append(col)

X_train = X_train.drop(drop_list, axis=1)
X_val = X_val.drop(drop_list, axis=1)

K_fold = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
best_models = {}

for model_name, model in models.items():
    print(f"\nTuning {model_name}...")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grids[model_name],
        n_iter=10,
        scoring="f1_macro",
        cv=3,  
        random_state=42,
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train)
    best_models[model_name] = (random_search.best_estimator_, random_search.best_params_,random_search.best_score_)

    print(f"Best parameters for {model_name}: {random_search.best_params_}")
    print(f"Best cross-validated f1 socre for {model_name}: {random_search.best_score_:.4f}")

end_time = time.time()
hours_passed = (end_time - start_time) / 3600
print(f"It took {hours_passed:.2f} hours")

for model_name, (best_model, best_params, best_score) in best_models.items():
    save_scores(f"{model_name} RandomSearch, {feature_selection}", best_params, best_score, hours_passed)


