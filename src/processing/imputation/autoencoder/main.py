# %%
from collections import Counter

import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from src import consts as const
from src import utils
from src.processing.imputation import helper
from src.processing.imputation.autoencoder.masked_ae import Autoencoder

sns.set(palette="Set2")


## %%
# Read data
df_incomplete, df_no_nan, df = helper.imputation_dataset(
    const.PROCESSED_DATA_DIR / 'enhanced_with_nan.csv')

## %%
# Split data
X_train, y_train, X_val, y_val, X_test, y_test = utils.split_data(
    df_incomplete, df_no_nan)

## %%
# Encode categorical using Target encoding
num_cols = y_train.select_dtypes(include=np.number).columns
cat_cols = y_train.select_dtypes(include=['object']).columns

## Encode categorical using Label Encoding
train_encoder = ce.ordinal.OrdinalEncoder(verbose=2, handle_missing=-1)
train_encoder = train_encoder.fit(y_train[cat_cols])
X_train[cat_cols] = train_encoder.transform(X_train[cat_cols])
y_train[cat_cols] = train_encoder.transform(y_train[cat_cols])
X_train[X_train[cat_cols] == -1] = np.nan

val_encoder = ce.ordinal.OrdinalEncoder(verbose=2, handle_missing=-1)
val_encoder = val_encoder.fit(y_val[cat_cols])
X_val[cat_cols] = val_encoder.transform(X_val[cat_cols])
y_val[cat_cols] = val_encoder.transform(y_val[cat_cols])
X_val[X_val[cat_cols] == -1] = np.nan

test_encoder = ce.ordinal.OrdinalEncoder(verbose=2, handle_missing=-1)
test_encoder = test_encoder.fit(y_test[cat_cols])
X_test[cat_cols] = test_encoder.transform(X_test[cat_cols])
X_test[X_test[cat_cols] == -1] = np.nan

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
autoencoder = Autoencoder(X_train_arr.shape[1])
autoencoder.train(X_train_arr, y_train_arr, X_val_arr,
                  y_val_arr, batch_size=512)

#autoencoder.recreate_model('autoencoder_weights.h5')

# %%
pred = autoencoder.infer(np.copy(X_test_arr))

# %%
# Revert normalization
imputed_arr = test_scaler.inverse_transform(pred)

# %%
# Revert categorical encoding
imputed_df = pd.DataFrame(imputed_arr, columns=y_train.columns)
imputed_df[cat_cols] = imputed_df[cat_cols].round()
imputed_df[cat_cols] = test_encoder.inverse_transform(imputed_df[cat_cols])


# %%
# Numerical error (MSE)
# Numerical mask
num_nan_mask = X_test[num_cols].apply(pd.isnull)
masked_y_test = y_test[num_cols]
masked_pred = imputed_df[num_cols]
mse = helper.masked_mse(
    masked_y_test, masked_pred, num_nan_mask)
mae = helper.masked_mae(
    masked_y_test, masked_pred, num_nan_mask)
baseline = mean_absolute_error(masked_y_test,
                               np.full(masked_y_test.values.shape, masked_y_test.mean()))


# %%
# Categorical error (accuracy)
cat_nan_mask = X_test[cat_cols].apply(pd.isnull)
accuracy = helper.masked_accuracy(
    y_test[cat_cols], imputed_df[cat_cols], cat_nan_mask)


# %%
print('MSE: {:.3f}'.format(mse))
print('MAE: {:.3f}'.format(mae))
print('Baseline (mean): {:.3f}'.format(baseline))
print('Accuracy: {:.3f}'.format(accuracy))


# %%
missing_cols = df.columns[df.isna().any()].tolist()
mae_dict = {}
mse_dict = {}
mape_dict = {}
accuracy_dict = {}
for col in missing_cols:
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


#%%
nan_num_cols = set(num_cols) & set(missing_cols)
true_df = masked_y_test[nan_num_cols].head(100)
pred_df = masked_pred[nan_num_cols].head(100)
true_df.reset_index(drop=True, inplace=True)
pred_df.reset_index(drop=True, inplace=True)
fig, ax = plt.subplots(figsize=(8, 6))

for col in nan_num_cols:
    true_col = true_df[col].sort_values()
    ax = sns.scatterplot(true_col.values, true_col.index)
    pred_col = pred_df[col].sort_values()
    ax = sns.scatterplot(pred_col.values, pred_col.index)


#%%
# Baseline Numerical
num_y_test = y_test[num_cols]
num_nan_mask = X_test[num_cols].apply(pd.isnull)
mean_df = X_test[num_cols].copy().fillna(X_test[num_cols].mean())

#%%
print(mean_absolute_error(num_y_test.values[num_nan_mask], mean_df.values[num_nan_mask]))
print(mean_squared_error(num_y_test.values[num_nan_mask], mean_df.values[num_nan_mask]))


#%%
# Baseline Categorical
cat_y_test = y_test[cat_cols]
cat_nan_mask = X_test[cat_cols].apply(pd.isnull)
most_freq_df = X_test[cat_cols].copy().apply(lambda x: x.fillna(x.value_counts().index[0]))

#%%
truth_values = [true == pred for true, pred in zip(cat_y_test.values[cat_nan_mask], most_freq_df.values[cat_nan_mask])]
truth_values = Counter(truth_values)
print(truth_values[True] / (truth_values[True] + truth_values[False]))


#%%
