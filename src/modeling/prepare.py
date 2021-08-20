"""Creation and storage of datasets for each predictive task."""
# %%
# Importing libraries
import json
import os

import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax, skew
from sklearn.model_selection import train_test_split

from src import consts as const
from src import utils

# Reading configurations
with open('src/settings.json') as settings:
    props = json.load(settings)

imputed = props['dataset']['imputed']
target = props['dataset']['target']
cat_encoding = props['dataset']['cat_encoding']
fleet_type = props['dataset']['fleet_type']


df = pd.read_csv(const.PROCESSED_DATA_DIR /
                 'cleaned_dropped_nan.csv', sep='\t')

# Getting predictors-target dataset
X, y = utils.prepare_for_modelling(df, target, fleet_type=fleet_type)

### Uncomment to add a validation set to storage ###

# Splitting data into train-test sets
# X_train, y_train, X_val, y_val, X_test, y_test = utils.split_data(
#    X, y, imputed=imputed)

#####################################################

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# %%
### Uncomment to store encoded data ###

# Encode categorical variables
# X_train, X_test, encoder = utils.encode_categorical(
#    X_train, y_train, X_test=X_test, y_test=y_test, encoder_name=cat_encoding)

# Scaling data
# X_train, X_test, scaler = utils.scale_data(X_train, X_test)

#######################################

# Creating directories to store data
transformed_data_dir = const.MODELING_DIR / 'data'
if not os.path.exists(transformed_data_dir):
    os.makedirs(transformed_data_dir)
# target folder
target_dir = transformed_data_dir / target
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Save train-val-test sets in numpy format files
np.save(target_dir /
        f'X_train_{target}{"_imputed" if imputed else ""}_{cat_encoding}_{fleet_type}.npy', X_train)
np.save(target_dir /
        f'y_train_{target}{"_imputed" if imputed else ""}_{cat_encoding}_{fleet_type}.npy', y_train)
np.save(target_dir /
        f'X_test_{target}{"_imputed" if imputed else ""}_{cat_encoding}_{fleet_type}.npy', X_test)
np.save(target_dir /
        f'y_test_{target}{"_imputed" if imputed else ""}_{cat_encoding}_{fleet_type}.npy', y_test)

### Uncomment to store encoder and scaler ###

# np.save(target_dir /
#        f'encoder_{target}{"_imputed" if imputed else ""}_{cat_encoding}_{fleet_type}.npy', encoder)

# np.save(target_dir /
#        f'scaler_{target}{"_imputed" if imputed else ""}_{cat_encoding}_{fleet_type}.npy', scaler)

##############################################

# Uncomment to create and store dataset with
#  numerical features transformed with BoxCox ###

"""
# Transformed train
pos_num_cols = X.select_dtypes(include=['number']).columns
pos_num_cols = X[pos_num_cols].loc[:, X[pos_num_cols].gt(0).all()].columns
X_linear = X[X_train.index]

skew_features = X_linear[pos_num_cols].apply(
    lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
low_skew = skew_features[skew_features < -0.5]
skew_index = (high_skew + low_skew).index

for i in skew_index:
    X_linear[i] = boxcox1p(X_linear[i], boxcox_normmax(X_linear[i] + 1))

X_linear = utils.encode_categorical(
    X_linear, y_train, encoder_name=cat_encoding)

np.save(target_dir /
        f'X_linear_train_{target}{"_imputed" if imputed else ""}_{cat_encoding}_{fleet_type}.npy', X_linear)
"""
