# %%
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src import consts as const
from src import utils
from src.processing import attribute_builder as ab
from src.processing import plotting, refactor
from src.processing.imputation import helper
from src.processing.imputation.autoencoder.masked_ae import Autoencoder

sns.set(palette="Set2")

# %%
df = pd.read_csv(const.PROCESSED_DATA_DIR / 'cleaned_with_nan.csv', sep='\t')

# %%
plotting.missing_values(df, save=True)

# %%
# Drop unnecessary features
df.drop(['flight_number', 'tail_number', 'aircraft_model', 'fleet',
         'taxi_out', 'air_time', 'taxi_in', 'actual_block_time',
         'scheduled_block_time', 'scheduled_rotation_time', 'prev_delay_code'],
        axis=1, inplace=True)

# %%
df_no_nan = df.dropna()

# impute Nan values
nan_columns = df.columns[df.isna().any()].tolist()
incomplete_sample = df_no_nan.sample(frac=0.5, random_state=1)
incomplete_sample = helper.impute_nan(
    incomplete_sample, miss_rate=.4, cols=nan_columns)
df_incomplete = df_no_nan.copy()
df_incomplete.loc[incomplete_sample.index] = incomplete_sample


# %%
dummy_df_inc, dummy_no_nan = helper.one_hot_encoding(df_incomplete, df_no_nan)


# %%
# Split data
X_train, y_train, X_val, y_val, X_test, y_test = utils.split_data(
    dummy_df_inc, dummy_no_nan)

# %%
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include=['object']).columns


# %%
# Convert to arrays
X_train_arr = X_train.values.astype(float)
y_train_arr = y_train.values.astype(float)

X_val_arr = X_val.values.astype(float)
y_val_arr = y_val.values.astype(float)

X_test_arr = X_test.values.astype(float)

# %%
# Normalize data
obs_row_idx = np.where(np.isfinite(np.sum(X_train_arr, axis=1)))
X_train_complete_records = X_train_arr[obs_row_idx]
train_scaler = MinMaxScaler().fit(X_train_complete_records)
X_train_arr = train_scaler.transform(X_train_arr)
y_train_arr = train_scaler.transform(y_train_arr)
del X_train_complete_records

X_val_arr = train_scaler.transform(X_val_arr)
y_val_arr = train_scaler.transform(y_val_arr)

test_obs_row_idx = np.where(np.isfinite(np.sum(X_test_arr, axis=1)))
X_test_complete_records = X_test_arr[test_obs_row_idx]
test_scaler = MinMaxScaler().fit(X_test_complete_records)
X_test_arr = test_scaler.transform(X_test_arr)
del X_test_complete_records


# %%
autoencoder = Autoencoder(X_train_arr.shape[1])
autoencoder.train(X_train_arr, y_train_arr, X_val_arr,
                  y_val_arr, batch_size=512)

# autoencoder.load_model('autoencoder.h5')


# %%
imputed_arr = autoencoder.infer(np.copy(X_test_arr))


# %%
# Revert normalization
imputed_arr = test_scaler.inverse_transform(imputed_arr)


# %%
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


# %%
# Rejoin dataset
imputed_num_subset = imputed_arr[:, :len(num_cols)]
imputed_num_subset = pd.DataFrame(imputed_num_subset, columns=num_cols)
imputed_df = pd.concat([imputed_num_subset, cat_subset], axis=1)

# %%
X_test = df_incomplete.loc[X_test.index]
y_test = df_no_nan.loc[y_test.index]
imputed_df = imputed_df[y_test.columns]

# %%
nan_num_cols = list(set(num_cols).intersection(nan_columns))
nan_cat_cols = list(set(cat_cols).intersection(nan_columns))

# %%
# Numerical error (MSE)
# Numerical mask
num_nan_mask = X_test[nan_num_cols].apply(pd.isnull)
masked_y_test = y_test[nan_num_cols]
masked_pred = imputed_df[nan_num_cols]
mse = helper.masked_mse(
    masked_y_test, masked_pred, num_nan_mask)
mae = helper.masked_mae(
    masked_y_test, masked_pred, num_nan_mask)
baseline = mean_absolute_error(masked_y_test,
                               np.full(masked_y_test.values.shape, masked_y_test.mean()))


# %%
# Categorical error (accuracy)
cat_nan_mask = X_test[nan_cat_cols].apply(pd.isnull)
accuracy = helper.masked_accuracy(
    y_test[nan_cat_cols], imputed_df[nan_cat_cols], cat_nan_mask)


# %%
print('MSE: {:.3f}'.format(mse))
print('MAE: {:.3f}'.format(mae))
print('Baseline (mean): {:.3f}'.format(baseline))
print('Accuracy: {:.3f}'.format(accuracy))


# %%
mae_dict = {}
mse_dict = {}
mape_dict = {}
accuracy_dict = {}
for col in nan_columns:
    col_nan_mask = X_test[col].apply(pd.isnull)
    if df[col].dtype == np.number:
        mae_dict[col] = helper.masked_mae(
            y_test[col], imputed_df[col], col_nan_mask)
        mse_dict[col] = helper.masked_mse(
            y_test[col], imputed_df[col], col_nan_mask)
        mape_dict[col] = helper.masked_mape(
            y_test[col], imputed_df[col], col_nan_mask)
    if df[col].dtype == np.object:
        accuracy_dict[col] = helper.masked_accuracy(
            y_test[col], imputed_df[col], col_nan_mask)
print('MAE', end="\n\n")
[print('{}: {:.3f}'.format(key, mae_dict[key])) for key in mae_dict]
print('MSE', end="\n\n")
[print('{}: {:.3f}'.format(key, mse_dict[key])) for key in mse_dict]
print('MAPE', end="\n\n")
[print('{}: {:.3f}'.format(key, mape_dict[key])) for key in mape_dict]
print('ACCURACY', end="\n\n")
[print('{}: {:.3f}'.format(key, accuracy_dict[key])) for key in accuracy_dict]


# %%
# Baseline Numerical
num_y_test = y_test[nan_num_cols]
num_nan_mask = X_test[nan_num_cols].apply(pd.isnull)
mean_df = X_test[nan_num_cols].copy().fillna(X_test[nan_num_cols].mean())
print(
    f'Baseline MAE: {mean_absolute_error(num_y_test.values[num_nan_mask], mean_df.values[num_nan_mask])}')
print(
    f'Baseline MSE: {mean_squared_error(num_y_test.values[num_nan_mask], mean_df.values[num_nan_mask])}')


# %%
# Baseline Categorical
cat_y_test = y_test[nan_cat_cols]
cat_nan_mask = X_test[nan_cat_cols].apply(pd.isnull)
most_freq_df = X_test[nan_cat_cols].copy().apply(
    lambda x: x.fillna(x.value_counts().index[0]))
truth_values = [true == pred for true, pred in zip(
    cat_y_test.values[cat_nan_mask], most_freq_df.values[cat_nan_mask])]
truth_values = Counter(truth_values)
print(
    f'Baseline Accuracy: {truth_values[True] / (truth_values[True] + truth_values[False])}')


# %%
b_mae_dict = {}
b_mse_dict = {}
b_mape_dict = {}
b_accuracy_dict = {}
for col in nan_columns:
    col_nan_mask = X_test[col].apply(pd.isnull)
    if df[col].dtype == np.number:
        b_mae_dict[col] = helper.masked_mae(
            y_test[col], mean_df[col], col_nan_mask)
        b_mse_dict[col] = helper.masked_mse(
            y_test[col], mean_df[col], col_nan_mask)
        b_mape_dict[col] = helper.masked_mape(
            y_test[col], mean_df[col], col_nan_mask)
    if df[col].dtype == np.object:
        b_accuracy_dict[col] = helper.masked_accuracy(
            y_test[col], most_freq_df[col], col_nan_mask)
print('MAE', end="\n\n")
[print('{}: {:.3f}'.format(key, b_mae_dict[key])) for key in b_mae_dict]
print('MSE', end="\n\n")
[print('{}: {:.3f}'.format(key, b_mse_dict[key])) for key in b_mse_dict]
print('MAPE', end="\n\n")
[print('{}: {:.3f}'.format(key, b_mape_dict[key])) for key in b_mape_dict]
print('ACCURACY', end="\n\n")
[print('{}: {:.3f}'.format(key, b_accuracy_dict[key]))
 for key in b_accuracy_dict]


# %%
