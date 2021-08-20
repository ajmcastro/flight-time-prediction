# %%
import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import consts as const
from src.processing.imputation import helper

sns.set(palette="Set2")

# %%
# Read data
df_incomplete, df_no_nan, df = helper.imputation_dataset(
    const.PROCESSED_DATA_DIR / 'enhanced_with_nan.csv')

# %%
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df_incomplete, df_no_nan, test_size=0.3)

# %%
# Encode categorical using Target Encoding
num_cols = y_train.select_dtypes(include=np.number).columns
cat_cols = y_train.select_dtypes(include=['object']).columns

# %%
# Encode categorical using Label Encoding
train_encoder = ce.ordinal.OrdinalEncoder(verbose=2, handle_missing=-1)
train_encoder = train_encoder.fit(y_train[cat_cols])
X_train[cat_cols] = train_encoder.transform(X_train[cat_cols])
y_train[cat_cols] = train_encoder.transform(y_train[cat_cols])
X_train[X_train[cat_cols] == -1] = np.nan

test_encoder = ce.ordinal.OrdinalEncoder(verbose=2, handle_missing=-1)
test_encoder = test_encoder.fit(y_test[cat_cols])
X_test[cat_cols] = test_encoder.transform(X_test[cat_cols])
X_test[X_test[cat_cols] == -1] = np.nan

# %%
# Convert to arrays
X_train_arr = X_train.values.astype(float)
y_train_arr = y_train.values.astype(float)

X_test_arr = X_test.values.astype(float)

# %%
# Normalize data
obs_row_idx = np.where(np.isfinite(np.sum(X_train_arr, axis=1)))
X_train_complete_records = X_train_arr[obs_row_idx]
train_scaler = StandardScaler().fit(X_train_complete_records)
X_train_arr = train_scaler.transform(X_train_arr)
y_train_arr = train_scaler.transform(y_train_arr)
del X_train_complete_records

test_obs_row_idx = np.where(np.isfinite(np.sum(X_test_arr, axis=1)))
X_test_complete_records = X_test_arr[test_obs_row_idx]
test_scaler = StandardScaler().fit(X_test_complete_records)
X_test_arr = test_scaler.transform(X_test_arr)
del X_test_complete_records

# %%
imp = IterativeImputer(max_iter=10, random_state=0, verbose=2)
imp.fit(X_train_arr)

# %%
imputed_arr = imp.transform(X_test_arr)

# %%
# Revert normalization
imputed_arr = test_scaler.inverse_transform(imputed_arr)

# %%
# Revert categorical encoding
imputed_df = pd.DataFrame(imputed_arr, columns=y_train.columns)
imputed_df[num_cols] = imputed_df[num_cols].round(2)
imputed_df[cat_cols] = imputed_df[cat_cols].round()
imputed_df[cat_cols] = test_encoder.inverse_transform(imputed_df[cat_cols])

# %%
num_nan_mask = X_test[num_cols].apply(pd.isnull)
mse = helper.masked_mse(y_test[num_cols], imputed_df[num_cols], num_nan_mask)
print(mse)

# %%
cat_nan_mask = X_test[cat_cols].apply(pd.isnull)
accuracy = helper.masked_accuracy(
    y_test[cat_cols], imputed_df[cat_cols], cat_nan_mask)
print(accuracy)

# %%
mse = mean_squared_error(
    y_test[num_cols].values[num_nan_mask], imputed_df[num_cols].values[num_nan_mask])
mae = mean_absolute_error(
    y_test[num_cols].values[num_nan_mask], imputed_df[num_cols].values[num_nan_mask])
actual_mean = np.mean(y_test[num_cols].values[num_nan_mask])
print('MSE: {:.3f}\nMAE: {:.3f}\nMean: {:.3f}'.format(mse, mae, actual_mean))
