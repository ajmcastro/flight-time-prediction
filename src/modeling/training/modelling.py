"""Main file for model building and hyperparameter tuning."""
# %%
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import dump

from src import consts as const
from src.modeling.training import hp_tuning
from src.modeling.training import cross_validation

# os.chdir('/home/afonso/Projects/flight_time_prediction')

# Reading configurations
with open('src/settings.json') as settings:
    props = json.load(settings)

imputed = props['dataset']['imputed']
target = props['dataset']['target']
cat_encoding = props['dataset']['cat_encoding']
fleet_type = props['dataset']['fleet_type']

model_name = props['model']['model_name']


# %%
# Loading train-val sets
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

X_train = np.load(
    const.MODELING_DIR / 'data' / target / f'X_train_{target}{"_imputed" if imputed else ""}_{cat_encoding}_{fleet_type}.npy')
y_train = np.load(
    const.MODELING_DIR / 'data' / target / f'y_train_{target}{"_imputed" if imputed else ""}_{cat_encoding}_{fleet_type}.npy')

np.load = np_load_old


# %%
# Train and tune
res, gs = hp_tuning.train_tune_selected_model(model_name, X_train, y_train)


# %%
# Cross-validation
results = cross_validation.cross_selected_model(model_name, X_train, y_train)

print('MAE mean: ' + str(results['test_neg_mean_absolute_error'].mean()) + '\n' +
      'MAE std: ' + str(results['test_neg_mean_absolute_error'].std()) + '\n' +
      'R2 mean: ' + str(results['test_r2'].mean() * 100) + '\n' +
      'R2 std: ' + str(results['test_r2'].std() * 100) + '\n' +
      'RMSE mean: ' + str(math.sqrt(results['test_neg_mean_squared_error'].mean())) + '\n' +
      'RMSE std: ' + str(math.sqrt(results['test_neg_mean_squared_error'].std())) + '\n')


# %%
# Train model
model = cross_validation.fit_selected_model(model_name, X_train, y_train)


# %%
# Save model
dump(model, const.MODEL_DIR / 'best_performers' /
     f'{model_name}_{target}_{fleet_type}.sav')


# %%
# Residual plot
sns.set(palette="Set2")

y_pred = model.predict(X_train).flatten()

plt.scatter(y_pred, y_train - y_pred)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
# plt.savefig('residuals.png', bbox_inches='tight')
