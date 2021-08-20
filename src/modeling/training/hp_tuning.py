"""Hyperparameter tuning auxiliar methods."""

import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from xgboost import XGBRegressor


def get_results(model_name, gs):
    """Print results of grid search in tabular form.

    Arguments:
        model_name {string} -- model name 
        gs {GridSearch (sklearn)} -- grid search algorithm class for the model

    Returns:
        dataframe (pandas) -- tabular representation of results for MAE, RMSE and R2
    """

    rcols = ['Name', 'Model', 'BestParameters', 'Scorer', 'Index', 'BestScore', 'BestScoreStd', 'MeanScore',
             'MeanScoreStd', 'Best']
    res = pd.DataFrame(columns=rcols)

    results = gs.cv_results_
    model = gs.best_estimator_

    scoring = {'MAE': 'neg_mean_absolute_error',
               'R2': 'r2',
               'RMSE': 'neg_mean_squared_error'}

    for scorer in sorted(scoring):
        best_index = np.nonzero(
            results['rank_test_%s' % scoring[scorer]] == 1)[0][0]
        if scorer == 'RMSE':
            best = np.sqrt(-results['mean_test_%s' %
                                    scoring[scorer]][best_index])
            best_std = np.sqrt(results['std_test_%s' %
                                       scoring[scorer]][best_index])
            scormean = np.sqrt(-results['mean_test_%s' %
                                        scoring[scorer]].mean())
            stdmean = np.sqrt(results['std_test_%s' % scoring[scorer]].mean())

        elif scorer == 'MAE':
            best = (-results['mean_test_%s' % scoring[scorer]][best_index])
            best_std = results['std_test_%s' % scoring[scorer]][best_index]
            scormean = (-results['mean_test_%s' % scoring[scorer]].mean())
            stdmean = results['std_test_%s' % scoring[scorer]].mean()
        else:
            best = results['mean_test_%s' % scoring[scorer]][best_index]*100
            best_std = results['std_test_%s' % scoring[scorer]][best_index]*100
            scormean = results['mean_test_%s' % scoring[scorer]].mean()*100
            stdmean = results['std_test_%s' % scoring[scorer]].mean()*100

        r1 = pd.DataFrame([(model_name, model, gs.best_params_, scorer, best_index, best, best_std, scormean,
                            stdmean, gs.best_score_)],
                          columns=rcols)
        res = res.append(r1)

        bestscore = np.sqrt(-gs.best_score_)

    print("Best Score: {:.6f}".format(bestscore))
    print('---------------------------------------')
    print('Best Parameters:')
    print(gs.best_params_)

    return res


def residuals_plots(model, X, y, save=False):
    """Generate residual plot.

    Arguments:
        model {Pipeline (sklearn)} -- trained model
        X -- testing set predictors
        y -- testing set target
        save -- whether or not to save plot
    """
    y_pred = model.predict(X)

    sns.regplot(y=(y - y_pred), x=y_pred)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    if save:
        plt.savefig(f'{model.__class__.__name__}.png', bbox_inches='tight')


def hyperparameter_testing(model, param_grid, X, y, cv=5):
    """Grid search model for best hyperparameters.

    Arguments:
        model {Pipeline (sklearn)} -- model to train
        param_grid -- set of hyperparameters to test
        X -- predictors
        y -- target
        cv -- k value for k-fold Cross-Validation

    Returns:
        res {dataframe (pandas)} -- hyperparameter tuning best results
        gs {GridSearch (sklearn)} -- fitted grid search
    """
    gs = GridSearchCV(estimator=model, param_grid=param_grid, refit='neg_mean_squared_error', scoring=list(
        ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']), cv=cv, verbose=10, n_jobs=-1)
    gs.fit(X, y)

    res = get_results(model.__class__.__name__, gs)
    print(residuals_plots(model, X, y))
    print(res.loc[:, 'Scorer': 'MeanScoreStd'])

    return res, gs


#### HYPERPARAMETER TUNING ####

def train_tune_ridge(X, y, num_features):
    """Hyperparameter tuning for Ridge.

    Arguments:
        X -- predictors
        y -- target
        num_features -- number of features to use in feature selection algorithms

    Returns:
        res {dataframe (pandas)} -- hyperparameter tuning best results
        gs {GridSearch (sklearn)} -- fitted grid search
    """
    ridge_model = Pipeline([
        ('cat_encoder', ce.LeaveOneOutEncoder(verbose=2)),
        ('scl', RobustScaler()),
        ('reduce_dim', 'passthrough'),
        ('model', Ridge(random_state=101))])

    param_grid = [
        {
            'scl': [StandardScaler(), RobustScaler()],
            'reduce_dim': [PCA(random_state=101)],
            'reduce_dim__n_components': num_features,
            'reduce_dim__whiten': [True, False],
            'model__alpha': [0.01, 0.05, 0.1, 0.5, 1],
            'model__max_iter': [1, 2]
        },
        {
            'scl': [MinMaxScaler()],
            'reduce_dim': [SelectKBest(chi2)],
            'reduce_dim__k': num_features,
            'model__alpha': [0.2, 0.5, 1],  # 0.01, 0.05, 0.1, 2
            'model__max_iter': [1, 2]
        },
    ]
    return hyperparameter_testing(
        ridge_model, param_grid, X, y)


def train_tune_lasso(X, y, num_features):
    """Hyperparameter tuning for Lasso.

    Arguments:
        X -- predictors
        y -- target
        num_features -- number of features to use in feature selection algorithms

    Returns:
        res {dataframe (pandas)} -- hyperparameter tuning best results
        gs {GridSearch (sklearn)} -- fitted grid search
    """
    lasso_model = Pipeline([
        ('cat_encoder', ce.LeaveOneOutEncoder(verbose=2)),
        ('scl', RobustScaler()),
        ('reduce_dim', 'passthrough'),
        ('model', Lasso(random_state=101))])

    param_grid = [
        {
            'scl': [StandardScaler(), RobustScaler()],
            'reduce_dim': [PCA(random_state=101)],
            'reduce_dim__n_components': num_features,
            'reduce_dim__whiten': [True, False],
            'model__alpha': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4],
            'model__max_iter': [1, 2],  # , 10, 100],
            'model__selection': ['random', 'cyclic']
        },
        {
            'scl': [MinMaxScaler()],
            'reduce_dim': [SelectKBest(chi2)],
            'reduce_dim__k': num_features,
            'model__alpha': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4],
            'model__max_iter': [1, 2],  # , 10, 100],
            'model__selection': ['random', 'cyclic']
        }
    ]
    return hyperparameter_testing(
        lasso_model, param_grid, X, y)


def train_tune_elastic(X, y, num_features):
    """Hyperparameter tuning for ElasticNet.

    Arguments:
        X -- predictors
        y -- target
        num_features -- number of features to use in feature selection algorithms

    Returns:
        res {dataframe (pandas)} -- hyperparameter tuning best results
        gs {GridSearch (sklearn)} -- fitted grid search
    """
    elastic_model = Pipeline([
        ('cat_encoder', ce.LeaveOneOutEncoder(verbose=2)),
        ('scl', RobustScaler()),
        ('reduce_dim', 'passthrough'),
        ('model', ElasticNet(random_state=101))])

    param_grid = [
        {
            'scl': [StandardScaler(), RobustScaler()],
            'reduce_dim': [PCA(random_state=101)],
            'reduce_dim__n_components': num_features,
            'reduce_dim__whiten': [True, False],
            'model__max_iter': [1, 2],
            'model__alpha': [1e-6, 1e-5, 1e-4],
            'model__l1_ratio': [0.8, 0.9, 0.99]
        },
        {
            'scl': [MinMaxScaler()],
            'reduce_dim': [SelectKBest(chi2)],
            'reduce_dim__k': num_features,
            'model__max_iter': [1, 2],
            'model__alpha': [1e-6, 1e-5, 1e-4],
            'model__l1_ratio': [0.8, 0.9, 0.99]
        }
    ]
    return hyperparameter_testing(
        elastic_model, param_grid, X, y)


def train_tune_xgb(X, y, num_features):
    """Hyperparameter tuning for XGBoost.

    Arguments:
        X -- predictors
        y -- target
        num_features -- number of features to use in feature selection algorithms

    Returns:
        res {dataframe (pandas)} -- hyperparameter tuning best results
        gs {GridSearch (sklearn)} -- fitted grid search
    """
    xgb_model = Pipeline([
        ('cat_encoder', ce.LeaveOneOutEncoder(verbose=2)),
        ('scl', RobustScaler()),
        ('reduce_dim', 'passthrough'),
        ('model', XGBRegressor(random_state=101, n_jobs=-1, verbosity=2))])

    param_grid = {
        'scl': [RobustScaler()],
        'reduce_dim': [PCA(random_state=101)],
        'reduce_dim__n_components': np.delete(num_features, [len(num_features)-1]),
        'reduce_dim__whiten': [True, False],
        'model__n_estimators': [500],  # [100, 250],
        'model__learning_rate': [0.1],  # 0.01, 0.05, 0.3
        'model__reg_lambda': [1e-1],  # 1e-04, 1e-03, 1e-02, 5e-1
        'model__reg_alpha': [1e-3],  # 1e-02, 0.1, 1
        'model__max_depth': [4],
        'model__subsample': [0.8],
        'model__colsample_bytree': [0.7]
    }
    """
    {
        'scl': [MinMaxScaler()],
        'reduce_dim': [SelectKBest(chi2)],
        'reduce_dim__k': num_features,
        'model__n_estimators': [500],  # [100, 250],
        'model__learning_rate': [0.1, 0.3],  # 0.01, 0.05
        'model__reg_lambda': [1e-02, 1e-1, 5e-1],  # 1e-04, 1e-03
        'model__reg_alpha': [1e-3, 1e-02],  # 0.1, 1
        'model__max_depth': [4],
        'model__subsample': [0.8],
        'model__colsample_bytree': [0.7]
    }
    """
    return hyperparameter_testing(
        xgb_model, param_grid, X, y)


def train_tune_gb(X, y, num_features):
    """Hyperparameter tuning for GradientBoosting.

    Arguments:
        X -- predictors
        y -- target
        num_features -- number of features to use in feature selection algorithms

    Returns:
        res {dataframe (pandas)} -- hyperparameter tuning best results
        gs {GridSearch (sklearn)} -- fitted grid search
    """
    gb_model = Pipeline([
        ('cat_encoder', ce.LeaveOneOutEncoder(verbose=2)),
        ('scl', RobustScaler()),
        ('reduce_dim', 'passthrough'),
        ('model', GradientBoostingRegressor(
            random_state=101))])

    param_grid = {
        'scl': [RobustScaler()],
        'reduce_dim': [PCA(random_state=101)],
        'reduce_dim__n_components': np.delete(num_features, [len(num_features)-1]),
        'reduce_dim__whiten': [True, False],
        'model__n_estimators': [500],
        'model__learning_rate': [1e-2, 5e-2],  # 0.01, 0.1
        'model__max_depth': [4],
        'model__max_features': ['auto', 'sqrt']  # sqrt
    }
    """
    {
        'scl': [MinMaxScaler()],
        'reduce_dim': [SelectKBest(chi2)],
        'reduce_dim__k': num_features,
        'model__n_estimators': [500],
        'model__learning_rate': [1e-3, 1e-2, 5e-2, 0.1],  # 0.01, 0.1
        'model__max_depth': [4],
        'model__max_features': ['auto', 'sqrt']  # sqrt
    }
    """
    return hyperparameter_testing(
        gb_model, param_grid, X, y)


def train_tune_rf(X, y, num_features):
    """Hyperparameter tuning for RandomForest.

    Arguments:
        X -- predictors
        y -- target
        num_features -- number of features to use in feature selection algorithms

    Returns:
        res {dataframe (pandas)} -- hyperparameter tuning best results
        gs {GridSearch (sklearn)} -- fitted grid search
    """
    rf_model = Pipeline([
        ('cat_encoder', ce.LeaveOneOutEncoder(verbose=2)),
        ('scl', RobustScaler()),
        ('reduce_dim', 'passthrough'),
        ('model', RandomForestRegressor(n_jobs=-1, random_state=101, verbose=2))])

    param_grid = {
        'scl': [RobustScaler()],
        'reduce_dim': [PCA(random_state=101)],
        'reduce_dim__n_components': np.delete(num_features, [len(num_features)-1]),
        'reduce_dim__whiten': [True, False],
        'model__bootstrap': [True],  # False
        'model__max_depth': [4],  # 20
        'model__max_features': ['auto'],  # sqrt
        'model__n_estimators': [500]  # 400
    }
    return hyperparameter_testing(
        rf_model, param_grid, X, y)


def train_tune_selected_model(model_name, X, y):
    """Main method for hyperparameter tuning.

    Arguments:
        model_name -- name of algorithm to tune
        X -- predictors
        y -- target
    """
    dim = X.shape[1]
    N_FEATURES_OPTIONS = np.arange(dim-5, dim+1)

    if model_name == "ridge":
        return train_tune_ridge(X, y, N_FEATURES_OPTIONS)
    if model_name == "lasso":
        return train_tune_lasso(X, y, N_FEATURES_OPTIONS)
    if model_name == "elastic":
        return train_tune_elastic(X, y, N_FEATURES_OPTIONS)
    if model_name == "xgboost":
        return train_tune_xgb(X, y, N_FEATURES_OPTIONS)
    if model_name == "gradient_boosting":
        return train_tune_gb(X, y, N_FEATURES_OPTIONS)
    if model_name == "random_forest":
        return train_tune_rf(X, y, N_FEATURES_OPTIONS)
    print("Model name does not match any of the defined algorithms for tuning.")
