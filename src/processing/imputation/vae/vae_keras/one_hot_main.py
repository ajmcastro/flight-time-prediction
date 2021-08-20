# %%
import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
dummy_df_inc, dummy_no_nan = helper.one_hot_encoding(df_incomplete, df_no_nan)


## %%
# Split data
X_train, y_train, X_val, y_val, X_test, y_test = utils.split_data(
    dummy_df_inc, dummy_no_nan)


## %%
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include=['object']).columns


## %%
# Convert to arrays
X_train_arr = X_train.values.astype(float)
y_train_arr = y_train.values.astype(float)

X_val_arr = X_val.values.astype(float)
y_val_arr = y_val.values.astype(float)

X_test_arr = X_test.values.astype(float)

## %%
# Normalize data
obs_row_idx = np.where(np.isfinite(np.sum(X_train_arr, axis=1)))
X_train_complete_records = X_train_arr[obs_row_idx]
train_scaler = StandardScaler().fit(X_train_complete_records)
X_train_arr = train_scaler.transform(X_train_arr)
y_train_arr = train_scaler.transform(y_train_arr)
del X_train_complete_records

X_val_arr = train_scaler.transform(X_val_arr)
y_val_arr = train_scaler.transform(y_val_arr)

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


y_train = fit_batchsize(y_train, batch_size)
y_val = fit_batchsize(y_val, batch_size)
X_test_arr = fit_batchsize(X_test_arr, batch_size)
y_test = fit_batchsize(y_test, batch_size)


# %%
vae = VariationalAutoencoder(
    batch_size=batch_size, original_dim=y_train.shape[1], latent_dim=16, intermediate_dim=256, epsilon_std=1.0)

# train the model
vae.train(y_train.values, y_val.values, 100)


# %%
X_test_encoded = vae.encode(X_test_arr)
imputed_arr = vae.generate(X_test_encoded)

## %%
# Revert normalization
imputed_arr = test_scaler.inverse_transform(imputed_arr)

## %%
# Revert categorical encoding


def build_categorical_dataset(data, dummy_cat_cols, cat_cols):
    mle_complete = utils.maximum_likelihood_categorical(
        data, dummy_cat_cols)

    mle_df = pd.DataFrame(data=mle_complete, columns=dummy_cat_cols)
    cat_df = utils.reverse_dummy(mle_df)
    return cat_df[cat_cols]


dummy_cat_cols = dummy_df_inc.drop(num_cols, axis=1).columns
imputed_cat_subset = imputed_arr[:, len(num_cols):]
cat_subset = build_categorical_dataset(
    imputed_cat_subset, dummy_cat_cols, cat_cols)

## %%
# Rejoin dataset
imputed_num_subset = imputed_arr[:, :len(num_cols)]
imputed_num_subset = pd.DataFrame(imputed_num_subset, columns=num_cols)
whole_imputed = pd.concat([imputed_num_subset, cat_subset], axis=1)


## %%
X_test = df_incomplete.loc[X_test.index]
X_test = fit_batchsize(X_test, batch_size)
y_test = df_no_nan.loc[y_test.index]
y_test.reset_index(drop=True, inplace=True)

## %%
# Numerical mask
num_nan_mask = X_test[num_cols].apply(pd.isnull)
num_nan_mask.reset_index(drop=True, inplace=True)

# %%
# Numerical error (MSE)
mse = helper.masked_mse(
    y_test[num_cols], whole_imputed[num_cols], num_nan_mask)
print(mse)


# %%
# Categorical error (accuracy)
cat_nan_mask = X_test[cat_cols].apply(pd.isnull)
accuracy = helper.masked_accuracy(
    y_test[cat_cols], whole_imputed[cat_cols], cat_nan_mask)
print(accuracy)


# %%
mse = mean_squared_error(
    y_test[num_cols].values[num_nan_mask], whole_imputed[num_cols].values[num_nan_mask])
mae = mean_absolute_error(
    y_test[num_cols].values[num_nan_mask], whole_imputed[num_cols].values[num_nan_mask])
actual_mean = np.mean(y_test[num_cols].values[num_nan_mask])
print('MSE: {:.3f}\nMAE: {:.3f}\nMean: {:.3f}'.format(mse, mae, actual_mean))


# %%
missing_cols = df.columns[df.isna().any()].tolist()
mse_dict = {}
accuracy_dict = {}
for col in missing_cols:
    col_nan_mask = X_test[col].apply(pd.isnull)
    if df[col].dtype == np.number:
        mse_dict[col] = helper.masked_mse(
            y_test[col], imputed_df[col], col_nan_mask)
    if df[col].dtype == np.object:
        accuracy_dict[col] = helper.masked_accuracy(
            y_test[col], imputed_df[col], col_nan_mask)

# %%
# Encode categorical data
complete_observations = df[~pd.isnull(df).any(axis=1)]
df_encoder = ce.ordinal.OrdinalEncoder(verbose=2, handle_missing=-1)
df_encoder = df_encoder.fit(complete_observations[cat_cols])
df[cat_cols] = df_encoder.transform(df[cat_cols])
df[df[cat_cols] == -1] = np.nan

# %%
# Normalize full data
df_arr = df.values.astype(float)
full_obs_row_idx = np.where(np.isfinite(np.sum(df_arr, axis=1)))
df_complete_records = df_arr[full_obs_row_idx]
full_scaler = StandardScaler().fit(df_complete_records)
df_arr = full_scaler.transform(df_arr)
del df_complete_records

# %%
full_imputed = vae.impute(data_with_nan=np.copy(df_arr), max_iter=1)

# %%
# Revert Normalization
full_imputed = full_scaler.inverse_transform(full_imputed)

# %%
# Revert categorical encoding
full_imputed_df = pd.DataFrame(full_imputed, columns=y_train.columns)
full_imputed_df[cat_cols] = full_imputed_df[cat_cols].round()
full_imputed_df[cat_cols] = df_encoder.inverse_transform(
    full_imputed_df[cat_cols])

# %%
full_df = pd.read_csv(const.PROCESSED_DATA_DIR /
                      'compound_delay_code.csv', sep='\t', parse_dates=utils.DATE_COLS)

# %%
decomposed_date_names = ['off_block_date',
                         'take_off_date', 'landing_date', 'on_block_date']
full_imputed_df = utils.remove_decomposed_dates(
    full_imputed_df, cols=decomposed_date_names)

# %%
full_imputed_df = pd.concat([full_df[['scheduled_block_time', 'scheduled_departure_date',
                                      'scheduled_arrival_date', 'delay_code'] + decomposed_date_names], full_imputed_df], axis=1)


# %%
# Save imputed dataframe
full_imputed_df.to_csv(const.PROCESSED_DATA_DIR / 'imputed_dataset.csv',
                       sep='\t', encoding='utf-8', index=False)


# %%
