# General Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time

# Sklearn packages
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Models
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Ray
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import classification_report, f1_score

from utils import *
from utils_dicts import *

from functools import partial

import warnings
warnings.filterwarnings("ignore")

# Set the resources
available_resources = {"cpu": 6, "gpu": 0} # change to match your resources

start = start_time = time.time()
ray.init()
print(ray.available_resources())
os.environ["RAY_FUNCTION_SIZE_ERROR_THRESHOLD"] = "300000000"  # In bytes

# Load the data
train_df = pd.read_csv("./preprocessed_data/train_data.csv", index_col="Claim Identifier")

X = train_df.drop(["Claim Injury Type Encoded"], axis = 1)
y = train_df["Claim Injury Type Encoded"]

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

# Save the data in a way ray understands
X_train_ray = ray.put(X_train)
y_train_ray = ray.put(y_train)
X_val_ray = ray.put(X_val)
y_val_ray = ray.put(y_val)


search_space = {
    # Model Dependent
    "n_estimators": tune.grid_search([100, 200, 300]),         
    "learning_rate": tune.grid_search([0.01, 0.05, 0.1]),   # [0.01, 0.05, 0.1, 1.5]  
    "max_depth": tune.grid_search([5, 7]),                              
    "subsample": tune.grid_search([0.6, 0.9]),            
    "colsample_bytree": tune.grid_search([0.6, 0.9]),
    "reg_lambda": tune.grid_search([10, 100]),     # [1,10 ,100]
    "gamma": tune.grid_search([0.1, 0.3]),      #[0, 0.1, 0.3]            
    "grow_policy": tune.grid_search(["depthwise", "lossguide"]),
    
    # Always Use
    "use_SMOTE or use_RandomUnderSampler": tune.grid_search([False, "SMOTE", "RandomUnderSampler"]),
    "random_state":random_state
}

# Create Model
def XGBBoosted_GridSearch(config):
    X_train_gridsearch = ray.get(X_train_ray)
    y_train_gridsearch = ray.get(y_train_ray)
    X_val_gridsearch = ray.get(X_val_ray)
    y_val_gridsearch = ray.get(y_val_ray)

    # SMOTE or RandomUnderSampling
    if "use_SMOTE or use_RandomUnderSampler" == "SMOTE":
        smote = SMOTE()
        X_train_gridsearch, y_train_gridsearch = smote.fit_resample(X_train_gridsearch, y_train_gridsearch)
    elif "use_SMOTE or use_RandomUnderSampler" == "RandomUnderSampler":
        rus = RandomUnderSampler()
        X_train_gridsearch, y_train_gridsearch = rus.fit_resample(X_train_gridsearch, y_train_gridsearch)

    X_train_gridsearch = X_train_gridsearch.drop("Average Weekly Wage", axis = 1)
    X_val_gridsearch = X_val_gridsearch.drop("Average Weekly Wage", axis = 1)
    
    model = XGBClassifier(
                        n_estimators=config["n_estimators"],        
                        learning_rate=config["learning_rate"],      
                        max_depth=config["max_depth"],                          
                        subsample=config["subsample"],              
                        colsample_bytree=config["colsample_bytree"],
                        reg_lambda=config["reg_lambda"],
                        gamma=config["gamma"],                      
                        grow_policy=config["grow_policy"],          
                        objective="multi:softmax",                  
                        num_class=8,                                
                        eval_metric="merror",   
                        random_state = config["random_state"],                                      
                        verbosity=0                                 
                    )

    model.fit(X_train_gridsearch,y_train_gridsearch)
    
    # Predict on validation data
    y_pred = model.predict(X_val_gridsearch)

    # Compute F1 score
    f1 = f1_score(y_val_gridsearch, y_pred, average="macro")

    # Report results to Ray Tune
    session.report({"f1_score": f1})


# Run Grid Search
time_passed = (time.time() - start_time) / 60
print("Starting Grid Search after:", time_passed, "minutes")
analysis = tune.run(
    XGBBoosted_GridSearch,
    config=search_space, 
    scheduler=ASHAScheduler(metric="f1_score", mode="max", grace_period=5),
    resources_per_trial=available_resources,
    trial_dirname_creator=custom_trial_dirname,
    verbose=1
)

# Results
best_trial = analysis.get_best_trial(metric="f1_score", mode="max")
print("Best trial config: ", best_trial.config)
print("Best trial final F1 score: ", best_trial.last_result["f1_score"])

end_time = time.time()
hours_passed = (end_time - start_time) / 3600
print(f"It took {hours_passed:.2f} hours")

total_trials = len(analysis.trials)
print(f"Total number of trials: {total_trials}")

save_scores("XGBoost", best_trial.config, best_trial.last_result["f1_score"], hours_passed)