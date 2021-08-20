# %%
"""Auxiliar methods for whole project."""
import datetime
from collections import defaultdict

import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src import consts as const

sns.set(palette='Set2')

DATE_COLS = ['scheduled_departure_date',
             'off_block_date', 'take_off_date',
             'landing_date', 'on_block_date',
             'scheduled_arrival_date']


def normalize(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())


def revert_normalization(df, col_max, col_min):
    return df * (col_max - col_min) + col_min


def perc_error(true_val, pred_val):
    return (abs(pred_val - true_val) / abs(true_val)) * 100


def reverse_dummy(dummy_df):
    pos = defaultdict(list)
    vals = defaultdict(list)

    for i, c in enumerate(dummy_df.columns):
        if "_" in c:
            k, v = c.rsplit("_", 1)
            # print('i: {}, k: {}, v: {}'.format(i, k, v))
            pos[k].append(i)
            vals[k].append(v)
        else:
            pos["_"].append(i)

    df = pd.DataFrame({k: pd.Categorical.from_codes(
        np.argmax(dummy_df.iloc[:, pos[k]].values, axis=1),
        vals[k])
        for k in vals})

    df[dummy_df.columns[pos["_"]]] = dummy_df.iloc[:, pos["_"]]
    return df


def mle(row):
    # print(row)
    # print(np.argmax(row))
    res = np.zeros(row.shape[0])
    res[np.argmax(row)] = 1
    return res


def maximum_likelihood_categorical(data, dummy_cat_cols, skip_numerical=0):
    classes_count = {}
    for col in dummy_cat_cols:
        col_name = col.rsplit("_", 1)[0]
        classes_count[col_name] = classes_count.get(col_name, 0) + 1

    mle_complete = None
    col_classes = list(classes_count.values())

    # print(classes_count, col_classes)
    for i, key in enumerate(classes_count):
        cnt = classes_count[key]
        start_idx = int(sum(col_classes[0:i])) + skip_numerical
        # print(start_idx, (start_idx+cnt))
        col_completed = data[:, start_idx:start_idx+cnt]
        mle_completed = np.apply_along_axis(mle, axis=1, arr=col_completed)
        if mle_complete is None:
            mle_complete = mle_completed
        else:
            mle_complete = np.hstack([mle_complete, mle_completed])

    return mle_complete


def decompose_dates(df, cols=DATE_COLS.copy()):
    def decompose_date(df, name):
        df[name + '_year'] = df[name].dt.year.astype(int)
        df[name + '_month'] = df[name].dt.month.astype(int)
        df[name + '_day'] = df[name].dt.day.astype(int)
        if not (df[name].dt.time == datetime.time(0, 0)).all():
            df[name + '_hour'] = df[name].dt.hour.astype(int)
            df[name + '_minute'] = df[name].dt.minute.astype(int)
        df.drop(name, axis=1, inplace=True)
        return df

    for date in cols:
        df = decompose_date(df, date)
    return df


def reconstruct_dates(df, cols=DATE_COLS.copy()):
    rec_dict = {}
    for col in df.columns:
        if "_" in col:
            name, _ = col.rsplit("_", 1)
            if name in cols:
                if name in rec_dict:
                    rec_dict[name].append(col)
                else:
                    rec_dict[name] = [col]

    for name in rec_dict:
        print(name)
        df[rec_dict[name]] = df[rec_dict[name]].astype(int)
        df.rename({col: col.rsplit('_', 1)[
                  1] for col in rec_dict[name]}, axis='columns', inplace=True)

        time_names = ['year', 'month', 'day', 'hour', 'minute']
        df[name] = pd.to_datetime(df[time_names])
        df.drop(time_names, axis=1, inplace=True)

    return df


def remove_decomposed_dates(df, cols=DATE_COLS.copy()):
    for col in df.columns:
        if "_" in col:
            name, _ = col.rsplit("_", 1)
            if name in cols:
                df.drop(col, axis=1, inplace=True)
    return df


def split_data(X, y, imputed=False):
    if imputed:
        return split_data_imputed(X, y)
    else:
        return split_data_standard(X, y)


def split_data_standard(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2)
    return X_train, y_train, X_val, y_val, X_test, y_test


def split_data_imputed(X, y):
    df = pd.read_csv(const.PROCESSED_DATA_DIR /
                     'nan_dropped_indexed.csv', sep='\t', parse_dates=DATE_COLS)
    num_records = int(np.ceil(X.shape[0] * 0.3))
    test_idx = df.sample(num_records, random_state=1).iloc[:, 0].values
    X_train = X[~X.index.isin(test_idx)]
    X_test = X[X.index.isin(test_idx)]
    y_train = y[~y.index.isin(test_idx)]
    y_test = y[y.index.isin(test_idx)]
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2)
    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_for_modelling(df, target, fleet_type='whole'):
    if fleet_type == 'whole':
        pass
    if fleet_type == 'nb':
        df = df[df['actual_block_time'] < 360]
    if fleet_type == 'wb':
        df = df[df['actual_block_time'] >= 360]

    if target == 'taxi_in':
        df.drop(['origin_cloud_height', 'origin_cloud_coverage',
                 'origin_wind_direction', 'origin_wind_speed',
                 'origin_air_temperature',
                 'origin_visibility', 'actual_block_time',
                 'air_time', 'taxi_out'],
                axis=1, inplace=True)
    if target == 'taxi_out':
        df.drop(['destination_cloud_height', 'destination_cloud_coverage',
                 'destination_wind_direction', 'destination_wind_speed',
                 'destination_air_temperature',
                 'destination_visibility', 'actual_block_time',
                 'air_time', 'taxi_in'],
                axis=1, inplace=True)
    if target == 'air_time':
        df.drop(['taxi_in', 'taxi_out', 'actual_block_time'],
                axis=1, inplace=True)
    if target == 'actual_block_time':
        df.drop(['taxi_in', 'taxi_out', 'air_time'],
                axis=1, inplace=True)

    y = df.pop(target)
    X = df

    return X, y


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def plot_true_vs_pred(y_true, y_pred, fig_path):
    _, ax = plt.subplots()
    ax.scatter(y_true, y_pred)
    ax.plot([y_true.min(), y_true.max()], [
            y_true.min(), y_true.max()], 'k--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    plt.savefig(fig_path,
                bbox_inches='tight', facecolor='white')


def plot_perc_mape(y_test, y_pred, fig_path, interval=2):
    v_mape = np.vectorize(mape)
    mape_list = v_mape(y_test, y_pred)
    bin_mape = []
    bin_mae = []
    labels = []
    for i in range(0, interval * 10 + 1, interval):
        if i == 0:
            prev = i
            continue
        idx = np.argwhere((mape_list <= i) & (mape_list > prev))
        if idx.size == 0:
            continue
        labels.append(str(prev) + '-' + str(i))
        mae = mean_absolute_error(
            y_test.values[idx.flatten()], y_pred[idx.flatten()])
        bin_mape.append(mape_list[idx].shape[0])
        bin_mae.append(mae)
        prev = i
    for i in [50, 100]:
        idx = np.argwhere((mape_list <= i) & (mape_list > prev))
        if idx.size == 0:
            continue
        labels.append(str(prev) + '-' + str(i))
        mae = mean_absolute_error(
            y_test.values[idx.flatten()], y_pred[idx.flatten()])
        bin_mape.append(mape_list[idx].shape[0])
        bin_mae.append(mae)
        prev = i
        if i == 100:
            idx = np.argwhere(mape_list > i)
            if idx.size == 0:
                continue
            labels.append('> 100')
            mae = mean_absolute_error(
                y_test.values[idx.flatten()], y_pred[idx.flatten()])
            bin_mape.append(mape_list[idx].shape[0])
            bin_mae.append(mae)
            break

    # plot
    _, ax = plt.subplots(figsize=(8, 6))
    bin_mape_perc = np.array(bin_mape) / mape_list.shape[0] * 100
    ax = sns.barplot(x=labels, y=bin_mape_perc,
                     hue=np.around(bin_mae, 2), dodge=False)
    ax.legend(title='MAE', loc='upper right')
    ax.set_xlabel("Discretised MAPE")
    ax.set_ylabel("% of total observations")
    plt.savefig(fig_path,
                bbox_inches='tight', facecolor='white')


def plot_feature_importance(feature_tuple_list, fig_path):
    _, ax = plt.subplots()
    feat_imp_unzip = list(zip(*feature_tuple_list))
    ax.barh(feat_imp_unzip[0], feat_imp_unzip[1])
    plt.xlabel('Relative importance')
    plt.savefig(fig_path,
                bbox_inches='tight', facecolor='white')


def encode_categorical(X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None, encoder_name='target'):
    if encoder_name == 'target':
        encoder = ce.TargetEncoder(verbose=2)
    if encoder_name == 'leave_one_out':
        encoder = ce.LeaveOneOutEncoder(verbose=2)

    cat_cols = X_train.select_dtypes(include=['object']).columns

    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols], y_train)
    if X_test is not None:
        X_test[cat_cols] = encoder.transform(X_test[cat_cols], y_test)
        if X_val is not None:
            X_val[cat_cols] = encoder.transform(X_val[cat_cols], y_val)
            return X_train, X_val, X_test, encoder
        return X_train, X_test, encoder
    return X_train, encoder


def scale_data(X_train, X_val=None, X_test=None, scaler_name='Standard'):
    if scaler_name == 'MinMax':
        scaler = preprocessing.MinMaxScaler()
    if scaler_name == 'Standard':
        scaler = preprocessing.StandardScaler()
    if scaler_name == 'Robust':
        scaler = preprocessing.RobustScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        if X_val is not None:
            X_val_scaled = scaler.transform(X_val)
            return X_train_scaled, X_val_scaled, X_test_scaled, scaler
        return X_train_scaled, X_test_scaled, scaler
    return X_train_scaled, scaler
