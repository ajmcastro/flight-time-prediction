# %%
import json
import math
import os

import category_encoders as ce
import joblib
import numpy as np
import pandas as pd
from mlxtend.regressor import StackingCVRegressor
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src import consts as const
from src import utils

os.chdir('/home/afonso/Projects/MScThesis')

# %%
# Reading configurations
with open('src/settings.json') as settings:
    props = json.load(settings)

imputed = props['dataset']['imputed']
target = props['dataset']['target']
cat_encoding = props['dataset']['cat_encoding']
fleet_type = props['dataset']['fleet_type']

RSEED = 50
model_name = props['model']['model_name']


df = pd.read_csv(const.PROCESSED_DATA_DIR /
                 'cleaned_dropped_nan.csv', sep='\t')

# Getting predictors-target dataset
X, y = utils.prepare_for_modelling(df, target, fleet_type=fleet_type)

# %%
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# %%
ridge_model = Ridge(alpha=0.5, random_state=101)
lasso_model = Lasso(alpha=1e-6, random_state=101)
elastic_model = ElasticNet(alpha=1e-4, l1_ratio=0.9,
                           random_state=101)

estimators = (
    ridge_model,
    lasso_model,
    elastic_model
)

model = PMMLPipeline([
    ('cat_encoder', ce.LeaveOneOutEncoder(verbose=2)),
    ('scl', RobustScaler()),
    ('reduce_dim', PCA(29)),
    ('model', StackingCVRegressor(
        regressors=estimators,
        meta_regressor=GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05, max_depth=4,
            max_features='sqrt', random_state=101),
        verbose=2))
])

model.fit(X_train, y_train)

# %%
idx_zero = np.where(y_test != 0)[0]
X_test = X_test.iloc[idx_zero]
y_test = y_test.iloc[idx_zero]

# %%
y_pred = model.predict(X_test).flatten()
test_mae = mean_absolute_error(y_test, y_pred)
test_rmse = math.sqrt(mean_squared_error(y_test, y_pred))
test_mape = utils.mape(y_test, y_pred)
test_r_squared = r2_score(y_test, y_pred)

print(f'MAE: {test_mae:.3f}')
print(f'RMSE: {test_rmse:.3f}')
print(f'MAPE: {test_mape:.3f}')
print(f'R squared: {test_r_squared:.3f}')

# %%
utils.plot_perc_mape(
    y_test, y_pred, fig_path=const.IMAGES_DIR / f'{model_name}_{target}_{fleet_type}_perc_mape.png')


# %%
joblib.dump(model, "model.pkl.z", compress = 9)
#sklearn2pmml(model, "model.pmml", with_repr=True)
