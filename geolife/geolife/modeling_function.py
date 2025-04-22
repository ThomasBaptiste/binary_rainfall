# modeling_function.py
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

#---------------------------MODELING---------------------------------

def model_hyperparameters_evaluation_timesplitcv(model, param_grid, tscv, X, y):
# return performances of the model for timesplit crossvalidation and best set of hyperparameters

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='roc_auc', n_jobs=1)

    # cv timesplit loop
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[test_index]

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_proba = grid_search.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, y_proba)
        print(f"Fold {i + 1}: Best params = {grid_search.best_params_}, AUC = {auc:.3f}")

    return best_model