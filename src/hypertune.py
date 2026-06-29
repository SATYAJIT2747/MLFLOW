from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow

import dagshub
dagshub.init(repo_owner='satyagudu1146', repo_name='MLFLOW', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/satyagudu1146/MLFLOW.mlflow")


data = load_breast_cancer()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [10 , 50, 100],
    'max_depth': [None, 5, 10 , 15]}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy' , n_jobs=-1 , verbose=2)

mlflow.set_experiment("breast_cancer_hyperparameter_tuning_with_best_model_logging")

with mlflow.start_run() as parent:
    grid_search.fit(x_train, y_train)
    for i in range(len(grid_search.cv_results_['params'])):
       with mlflow.start_run(nested=True) as child:
            params = grid_search.cv_results_['params'][i]
            mean_test_score = grid_search.cv_results_['mean_test_score'][i]
            mlflow.log_params(params)
            mlflow.log_metric('mean_test_score', mean_test_score)
            
            
            # log training data
            train_data = pd.concat([x_train, y_train], axis=1)
            train_data = mlflow.data.from_pandas(train_data)
            mlflow.log_input(train_data, "train")
            
            # log test data
            test_data = pd.concat([x_test, y_test], axis=1)
            test_data = mlflow.data.from_pandas(test_data)
            mlflow.log_input(test_data, "test")
            
            #log src code
            mlflow.log_artifact(r"C:\Users\Satyajit\Desktop\MLOPS\MLFLOW\src\hypertune.py")
            
            # tags
            mlflow.set_tags({"author": "Satyajit" , "model": "RandomForest" , "dataset": "Breast Cancer Dataset"})
            
            
# Best parameters
mlflow.log_params(grid_search.best_params_)

# Best CV score
mlflow.log_metric("best_accuracy", grid_search.best_score_)

# Best model
mlflow.sklearn.log_model(
    grid_search.best_estimator_,
    "best_random_forest_model"
)