"""Methods for cross-validation of different algorithms."""

import category_encoders as ce
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from mlxtend.regressor import StackingCVRegressor
from src.modeling.training.model_builder import ModelBuilder
from xgboost import XGBRegressor


def setup_ridge(X, y):
    """Compile Ridge model with best hyperparameters.

    Arguments:
        X {numpy array} -- predictors
        y {numpy array} -- target

    Returns:
        Pipeline (sklearn) -- compiled model with categorical encoding,
        scaler and dimensionality reduction algorithm
    """
    return Pipeline([
        ('cat_encoder', ce.LeaveOneOutEncoder(verbose=2)),
        ('scl', MinMaxScaler()),
        ('reduce_dim', SelectKBest(chi2, k=32)),
        ('model', Ridge(alpha=0.5, random_state=101))])


def setup_lasso(X, y):
    """Compile Lasso model with best hyperparameters.

    Arguments:
        X {numpy array} -- predictors
        y {numpy array} -- target

    Returns:
        Pipeline (sklearn) -- compiled model with categorical encoding,
        scaler and dimensionality reduction algorithm
    """
    return Pipeline([
        ('cat_encoder', ce.LeaveOneOutEncoder(verbose=2)),
        ('scl', RobustScaler()),
        ('reduce_dim', PCA(29)),
        ('model', Lasso(alpha=1e-6, random_state=101))])


def setup_elastic(X, y):
    """Compile Elastic Net model with best hyperparameters.

    Arguments:
        X {numpy array} -- predictors
        y {numpy array} -- target

    Returns:
        Pipeline (sklearn) -- compiled model with categorical encoding,
        scaler and dimensionality reduction algorithm
    """
    return Pipeline([
        ('cat_encoder', ce.LeaveOneOutEncoder(verbose=2)),
        ('scl', RobustScaler()),
        ('reduce_dim', PCA(29)),
        ('model', ElasticNet(alpha=1e-4, l1_ratio=0.9,
                             random_state=101))])


def setup_xgb(X, y):
    """Compile XGBoost model with best hyperparameters.

    Arguments:
        X {numpy array} -- predictors
        y {numpy array} -- target

    Returns:
        Pipeline (sklearn) -- compiled model with categorical encoding,
        scaler and dimensionality reduction algorithm
    """
    return Pipeline([
        ('cat_encoder', ce.LeaveOneOutEncoder(verbose=2)),
        ('scl', RobustScaler()),
        ('reduce_dim', PCA(28, whiten=True)),
        ('model', XGBRegressor(
            n_estimators=500, reg_alpha=1e-3, reg_lambda=0.1,
            learning_rate=0.1, max_depth=4, subsample=0.8, colsample_bytree=0.7,
            random_state=101, n_jobs=-1, verbosity=2))])


def setup_gb(X, y):
    """Compile Gradient Boosting model with best hyperparameters.

    Arguments:
        X {numpy array} -- predictors
        y {numpy array} -- target

    Returns:
        Pipeline (sklearn) -- compiled model with categorical encoding,
        scaler and dimensionality reduction algorithm
    """
    return Pipeline([
        ('cat_encoder', ce.LeaveOneOutEncoder(verbose=2)),
        ('scl', RobustScaler()),
        ('reduce_dim', PCA(31, whiten=True)),
        ('model', GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05,
            max_depth=4, max_features='sqrt',
            random_state=101))])


def setup_rf(X, y):
    """Compile Random Forest model with best hyperparameters.

    Arguments:
        X {numpy array} -- predictors
        y {numpy array} -- target

    Returns:
        Pipeline (sklearn) -- compiled model with categorical encoding,
        scaler and dimensionality reduction algorithm
    """
    return Pipeline([
        ('cat_encoder', ce.LeaveOneOutEncoder(verbose=2)),
        ('scl', RobustScaler()),
        ('reduce_dim', PCA(27)),
        ('model', RandomForestRegressor(
            n_estimators=500, bootstrap=True, max_depth=4,
            n_jobs=-1, random_state=101, verbose=2))])


def setup_nn(X, y):
    """Compile Neural Network model.

    Arguments:
        X {numpy array} -- predictors
        y {numpy array} -- target

    Returns:
        Pipeline (sklearn) -- compiled model with categorical encoding
        and scaler
    """
    model_builder = ModelBuilder()
    model = model_builder.create_FFNN(X.shape[1])
    epochs = 500
    batch_size = 256
    min_delta = 0.01

    return Pipeline([
        ('cat_encoder', ce.LeaveOneOutEncoder(verbose=2)),
        ('scl', RobustScaler()),
        ('model', KerasRegressor(build_fn=model_builder.create_FFNN, input_dim=X.shape[1],
                                 epochs=epochs, batch_size=batch_size, verbose=2))
    ])


def setup_stacking(X, y):
    """Compile Stacking model with best hyperparameters. Comprised of
    Ridge, Lasso and Elastic Net with Gradient Boosting as meta-learner

    Arguments:
        X {numpy array} -- predictors
        y {numpy array} -- target

    Returns:
        Pipeline (sklearn) -- compiled model with categorical encoding,
        scaler and dimensionality reduction algorithm
    """
    ridge_model = Ridge(alpha=0.5, random_state=101)
    lasso_model = Lasso(alpha=1e-6, random_state=101)
    elastic_model = ElasticNet(alpha=1e-4, l1_ratio=0.9,
                               random_state=101)

    estimators = (
        ridge_model,
        lasso_model,
        elastic_model
    )

    return Pipeline([
        ('cat_encoder', ce.LeaveOneOutEncoder(verbose=2)),
        ('scl', RobustScaler()),
        ('reduce_dim', PCA(22)),
        ('model', StackingCVRegressor(
            regressors=estimators,
            meta_regressor=GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.05, max_depth=4,
                max_features='sqrt', random_state=101),
            verbose=2))
    ])


def cross_selected_model(model_name, X, y):
    """Get performance scores for model cross-validation.

    Arguments:
        model_name {string} -- algorithm name
        X {numpy array} -- predictors
        y {numpy array} -- target

    Returns:
        dict -- scores for model cross-validation of MSE, MAE and R2
    """
    if model_name == "ridge":
        model = setup_ridge(X, y)
    if model_name == "lasso":
        model = setup_lasso(X, y)
    if model_name == "elastic":
        model = setup_elastic(X, y)
    if model_name == "xgboost":
        model = setup_xgb(X, y)
    if model_name == "gradient_boosting":
        model = setup_gb(X, y)
    if model_name == "random_forest":
        model = setup_rf(X, y)
    if model_name == "ffnn":
        model = setup_nn(X, y)
    if model_name == "stacking":
        model = setup_stacking(X, y)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=101)
    return cross_validate(model, X, y, scoring=list(
        ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']), cv=kfold, verbose=3, n_jobs=1)


def fit_selected_model(model_name, X, y):
    """Train model.

    Arguments:
        model_name {string} -- algorithm name
        X {numpy array} -- predictors
        y {numpy array} -- target

    Returns:
        model -- model fitted in parameter data
    """
    if model_name == "ridge":
        model = setup_ridge(X, y)
    if model_name == "lasso":
        model = setup_lasso(X, y)
    if model_name == "elastic":
        model = setup_elastic(X, y)
    if model_name == "xgboost":
        model = setup_xgb(X, y)
    if model_name == "gradient_boosting":
        model = setup_gb(X, y)
    if model_name == "random_forest":
        model = setup_rf(X, y)
    if model_name == "ffnn":
        model = setup_nn(X, y)
    if model_name == "stacking":
        model = setup_stacking(X, y)

    model.fit(X, y)

    return model
