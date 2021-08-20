"""Auxiliar methods in imputation process."""
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src import utils


def one_hot_encoding(df_incomplete, df_no_nan):
    assert df_incomplete.isnull().values.any()
    assert ~df_no_nan.isnull().values.any()

    # One-hot encode data
    dummy_df_inc = pd.get_dummies(df_incomplete)
    dummy_no_nan = pd.get_dummies(df_no_nan)

    # applying missing data points to newly created categorical features
    for col in df_no_nan.columns:
        missing_cols = dummy_df_inc.columns.str.startswith(str(col) + "_")
        dummy_df_inc.loc[df_incomplete[col].isnull(), missing_cols] = np.nan

    # Matching incomplete shape with complete shape. Adding categorical columns
    # that were left because of NaN setting as 0 filled (no occurrences)
    # Also adding columns from original that were erase of one-hot encoding
    missing_cols = set(dummy_no_nan.columns) - set(dummy_df_inc.columns)
    for col in missing_cols:
        dummy_df_inc[col] = 0.0

    return dummy_df_inc, dummy_no_nan


def imputation_dataset(file_path):
    df = pd.read_csv(file_path, sep='\t', parse_dates=utils.DATE_COLS)

    df.drop(['scheduled_block_time', 'scheduled_departure_date',
             'scheduled_arrival_date', 'delay_code'],
            axis=1, inplace=True)

    df_dt_cols = df.select_dtypes(include=['datetime64']).columns
    df = utils.decompose_dates(df, df_dt_cols)

    df_no_nan = df.dropna()

    # impute Nan values
    nan_columns = df.columns[df.isna().any()].tolist()
    incomplete_sample = df_no_nan.sample(frac=0.5, random_state=1)
    incomplete_sample = impute_nan(
        incomplete_sample, miss_rate=.5, cols=nan_columns)
    df_incomplete = df_no_nan.copy()
    df_incomplete.loc[incomplete_sample.index] = incomplete_sample

    return df_incomplete, df_no_nan, df


def impute_nan(df, miss_rate=.2, cols=None):
    # Randomly imputing Nan values in dataset
    if cols is None:
        cols = df.columns

    df_incomplete = df.copy()
    for col in cols:
        to_nan_idx = df[col].sample(frac=miss_rate).index
        df_incomplete.loc[to_nan_idx, col] = np.nan

    return df_incomplete


def masked_mse(true_df, pred_df, mask):
    y_true = true_df.values[mask]
    y_pred = pred_df.values[mask]

    return mean_squared_error(y_true, y_pred)


def masked_mae(true_df, pred_df, mask):
    y_true = true_df.values[mask]
    y_pred = pred_df.values[mask]

    return mean_absolute_error(y_true, y_pred)


def masked_accuracy(true_df, pred_df, mask):
    y_true = true_df.values[mask]
    y_pred = pred_df.values[mask]
    incorrect = [true == pred for true, pred in zip(y_true, y_pred)]

    # counts the elements' frequency
    frequency = Counter(incorrect)
    print(frequency)
    return frequency[True] / (frequency[True] + frequency[False])


def masked_mape(true_df, pred_df, mask):
    y_true = true_df.values[mask]
    y_pred = pred_df.values[mask]

    return utils.mape(y_true, y_pred)
