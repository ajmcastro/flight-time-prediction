# %%
import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import consts as const
from src import utils
from src.processing.imputation import helper
from src.processing.imputation.vae.vae_keras.vae import VariationalAutoencoder

sns.set(palette="Set2")

## %%
# Read data
df_incomplete, df_no_nan, df = helper.imputation_dataset(
    const.PROCESSED_DATA_DIR / 'enhanced_with_nan.csv')

## %%
# Split data
_, X_test, y_train, y_test = train_test_split(
    df_incomplete, df_no_nan, test_size=0.3)

## %%
# Encode categorical using Target Encoding
num_cols = y_train.select_dtypes(include=np.number).columns
cat_cols = y_train.select_dtypes(include=['object']).columns

## %%
# Encode categorical using Label Encoding
train_encoder = ce.ordinal.OrdinalEncoder(verbose=2)
train_encoder = train_encoder.fit(y_train[cat_cols])
y_train[cat_cols] = train_encoder.transform(y_train[cat_cols])

test_encoder = ce.ordinal.OrdinalEncoder(
    verbose=2, handle_missing=-1)
test_encoder = test_encoder.fit(y_test[cat_cols])
X_test[cat_cols] = test_encoder.transform(X_test[cat_cols])
X_test[X_test[cat_cols] == -1] = np.nan

## %%
# Convert to arrays
y_train_arr = y_train.values.astype(float)
X_test_arr = X_test.values.astype(float)

## %%
# Normalize data
train_scaler = StandardScaler().fit(y_train_arr)
y_train_arr = train_scaler.transform(y_train_arr)

test_obs_row_idx = np.where(np.isfinite(np.sum(X_test_arr, axis=1)))
X_test_complete_records = X_test_arr[test_obs_row_idx]
test_scaler = StandardScaler().fit(X_test_complete_records)
X_test_arr = test_scaler.transform(X_test_arr)
del X_test_complete_records


# %%
batch_size = 256

def fit_batchsize(data, batch_size=256):
    n_size = (len(data) // batch_size) * batch_size
    return data[: n_size]


y_train_arr = fit_batchsize(y_train_arr, batch_size)
X_test_arr = fit_batchsize(X_test_arr, batch_size)
y_test = fit_batchsize(y_test, batch_size)


# %%
vae = VariationalAutoencoder(
    batch_size=batch_size, original_dim=y_train.shape[1], latent_dim=16, intermediate_dim=256, epsilon_std=1.0)

# train the model
vae.train(y_train_arr)


# %%
X_test_encoded = vae.encode(X_test_arr)
imputed_arr = vae.generate(X_test_encoded)

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
num_nan_mask = fit_batchsize(num_nan_mask, batch_size)
mse = helper.masked_mse(y_test[num_cols], imputed_df[num_cols], num_nan_mask)
print(mse)

# %%
cat_nan_mask = X_test[cat_cols].apply(pd.isnull)
cat_nan_mask = fit_batchsize(cat_nan_mask, batch_size)
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

