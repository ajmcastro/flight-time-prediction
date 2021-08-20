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
from src.processing.imputation.vae.vae import VariationalAutoencoder

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
"""
encoder = ce.TargetEncoder(verbose=2, handle_missing='return_nan')

X_train_encoded = encoder.fit_transform(X_train[cat_cols], y_train['air_time'])
X_train[cat_cols] = X_train_encoded
y_train_encoded = encoder.fit_transform(y_train[cat_cols], y_train['air_time'])
y_train[cat_cols] = y_train_encoded

X_test_encoded = encoder.fit_transform(X_test[cat_cols], y_test['air_time'])
X_test[cat_cols] = X_test_encoded
y_test_encoded = encoder.fit_transform(y_test[cat_cols], y_test['air_time'])
y_test[cat_cols] = y_test_encoded

df_encoded = encoder.fit_transform(df[cat_cols], df['air_time'])
df[cat_cols] = df_encoded
"""

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
# training parameters:
epochs = 100
batch_size = 256
learning_rate = 0.001

# INITIALISE AND TRAIN VAE
# define dict for network structure:
network_architecture = \
    dict(n_hidden_recog_1=128,  # 1st layer encoder neurons
         n_hidden_recog_2=64,  # 2nd layer encoder neurons
         n_hidden_gener_1=64,  # 1st layer decoder neurons
         n_hidden_gener_2=128,  # 2nd layer decoder neurons
         n_input=y_train_arr.shape[1],  # data input size
         n_z=32)  # dimensionality of latent space

# initialise VAE:
vae = VariationalAutoencoder(network_architecture,
                             learning_rate=learning_rate,
                             batch_size=batch_size)

# %%
# train VAE on full data
#vae = vae.train(data=y_train_arr,
#                training_epochs=epochs)
vae = vae.load_session()

# %%
# plot training history
fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(vae.losshistory_epoch, vae.losshistory, ax=ax)
ax.set(xlabel='Epoch', ylabel='Evidence Lower Bound (ELBO)')
plt.savefig('elbo_vae.png', bbox_inches='tight', facecolor='w')

# %%
# impute missing values
imputed_arr = vae.impute(data_with_nan=np.copy(X_test_arr), max_iter=1)

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
# Removing few not-filled nulls
nan_idx = imputed_df[imputed_df.isnull().any(axis=1)].index
imputed_df.drop(nan_idx, axis=0, inplace=True)
y_test.drop(nan_idx, axis=0, inplace=True)

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
