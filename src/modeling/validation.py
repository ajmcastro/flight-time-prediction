"""Methods, plots and evaluation metrics for
model performance assessment."""
# %%
import json
import math
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mlxtend.evaluate import feature_importance_permutation
from src import consts as const
from src import utils


# os.chdir('/home/afonso/Projects/flight_time_prediction')


# Importing libraries
sns.set(palette='Set2')

# Reading configurations
with open('src/settings.json') as settings:
    props = json.load(settings)

imputed = props['dataset']['imputed']
target = props['dataset']['target']
cat_encoding = props['dataset']['cat_encoding']
fleet_type = props['dataset']['fleet_type']

model_name = props['model']['model_name']

# Loading test sets
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

X_test = np.load(
    const.MODELING_DIR / 'data' / target / f'X_test_{target}{"_imputed" if imputed else ""}_{cat_encoding}_{fleet_type}.npy')
y_test = np.load(
    const.MODELING_DIR / 'data' / target / f'y_test_{target}{"_imputed" if imputed else ""}_{cat_encoding}_{fleet_type}.npy')

np.load = np_load_old

# Make sure y has no zero-valued targets for proper MAPE calculation
idx_zero = np.where(y_test != 0)
X_test = X_test[idx_zero]
y_test = y_test[idx_zero]

# %%
# Load models
model = joblib.load(const.MODEL_DIR / 'best_performers' /
                    f'{model_name}_{target}_{fleet_type}.sav')


# %%
y_pred = model.predict(X_test).flatten()

# %%
# Main regression metrics
test_mae = mean_absolute_error(y_test, y_pred)
test_rmse = math.sqrt(mean_squared_error(y_test, y_pred))
test_mape = utils.mape(y_test, y_pred)
test_r_squared = r2_score(y_test, y_pred) * 100

print(f'MAE: {test_mae:.3f}')
print(f'RMSE: {test_rmse:.3f}')
print(f'MAPE: {test_mape:.3f}')
print(f'R squared: {test_r_squared:.3f}')

# %%
# Plotting predicted vs actual results
utils.plot_true_vs_pred(
    y_test, y_pred, fig_path=const.IMAGES_DIR / f'{model_name}_{target}_{fleet_type}_true_vs_pred.png')

# %%
# Plotting MAPE discretisation
utils.plot_perc_mape(
    y_test, y_pred, fig_path=const.IMAGES_DIR / f'{model_name}_{target}_{fleet_type}_perc_mape.png')

# %%
# Statistical metrics
v_mae = np.vectorize(utils.mae)
mae_list = v_mae(y_test, y_pred)
print(f'Mean: {np.around(mae_list.mean(), 2)}')
print(f'Std: {np.around(mae_list.std(), 2)}')
print(f'Min: {np.around(mae_list.min(), 2)}')
print(f'Median: {np.around(np.median(mae_list), 2)}')
print(f'Max: {np.around(mae_list.max(), 2)}')

# %%
sns.set(palette="Set2")


def residual_plot(y_true, y_pred, save=False):
    sns.regplot(y=(y_true - y_pred), x=y_pred)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    if save:
        plt.savefig(const.IMAGES_DIR /
                    f'{model_name}_{target}_{fleet_type}_residuals.png', bbox_inches='tight', facecolor='w')


residual_plot(y_test, y_pred)

# %%


def cumsum_pca(pca, save=False):
    # Plotting the Cumulative Summation of the Explained Variance
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')
    if save:
        plt.savefig(const.IMAGES_DIR /
                    f'{model_name}_{target}_{fleet_type}_cumsum_pca.png', bbox_inches='tight', facecolor='w')


cumsum_pca(model['reduce_dim'])


# %%
# (For trees only) Getting non-zero feature importances
X, _ = utils.modeling_dataset(target)
feature_list = X.columns.tolist()

importances = list(model.feature_importances_)
feature_importances = [(feature, round(importance, 3))
                       for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(
    [x for x in feature_importances if x[1] > 0], key=lambda x: x[1], reverse=True)
[print('{:20}: {}'.format(*pair)) for pair in feature_importances]

# Plotting feature importance
utils.plot_feature_importance(
    feature_importances, fig_path=const.IMAGES_DIR / f'{model_name}_{target}_{fleet_type}_feature_importance.png')

# %%
# Assess outlier fleet in residuals plot
residuals = y_test - y_pred
outliers = np.where(np.absolute(residuals) > 100)
X_test[outliers][:, 3]

# %%
df = pd.read_csv(const.PROCESSED_DATA_DIR /
                 'cleaned_dropped_nan.csv', sep='\t')

X, y = utils.prepare_for_modelling(df, target, fleet_type=fleet_type)

cols = list(X.columns)
imp_vals, _ = feature_importance_permutation(
    predict_method=model.predict,
    X=X_test,
    y=y_test,
    metric='r2',
    num_rounds=1,
    seed=1)

# %%
feature_importances = [(feature, round(importance, 3))
                       for feature, importance in zip(cols, imp_vals / np.sum(imp_vals))]
feature_importances = sorted(
    [x for x in feature_importances if x[1] > 0], key=lambda x: x[1], reverse=True)
[print('{:20}: {}'.format(*pair)) for pair in feature_importances]

# %%
plt.figure()
plt.bar(range(X_test.shape[1]), imp_vals / np.sum(imp_vals))
plt.xticks(range(X_test.shape[1]))
plt.xlim([-1, X_test.shape[1]])
plt.ylim([0, 0.5])
plt.show()

# %%
v_mae = np.vectorize(utils.mae)
mae_list = v_mae(y_test, y_pred)
np.count_nonzero(mae_list < 5) / mae_list.size

# %%


def rmse(y_true, y_pred):
    return math.sqrt(np.mean(np.square(y_true - y_pred)))


# %%
v_rmse = np.vectorize(rmse)
rmse_list = v_rmse(y_test, y_pred)
np.count_nonzero(rmse_list < 2.6) / rmse_list.size
