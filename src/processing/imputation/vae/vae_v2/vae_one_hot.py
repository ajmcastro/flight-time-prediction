# %%
from __future__ import absolute_import, division, print_function

import argparse
import os

import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.layers import Dense, Input, Lambda
from keras.losses import binary_crossentropy, mse
from keras.models import Model
from keras.utils import plot_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src import consts as const
from src import utils
from src.processing.imputation import helper

sns.set(palette="Set2")


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


## %%
# Read data
df_incomplete, df_no_nan, df = helper.imputation_dataset(
    const.PROCESSED_DATA_DIR / 'enhanced_with_nan.csv')


## %%
dummy_df_inc, dummy_no_nan = helper.one_hot_encoding(df_incomplete, df_no_nan)


## %%
# Split data
_, X_test, y_train, y_test = train_test_split(
    dummy_df_inc, dummy_no_nan, test_size=0.3)


## %%
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include=['object']).columns


## %%
# Convert to arrays
y_train_arr = y_train.values.astype(float)
X_test_arr = X_test.values.astype(float)

## %%
# Normalize data
train_scaler = MinMaxScaler().fit(y_train_arr)
y_train_arr = train_scaler.transform(y_train_arr)

test_obs_row_idx = np.where(np.isfinite(np.sum(X_test_arr, axis=1)))
X_test_complete_records = X_test_arr[test_obs_row_idx]
test_scaler = MinMaxScaler().fit(X_test_complete_records)
X_test_arr = test_scaler.transform(X_test_arr)
del X_test_complete_records


## %%
original_dim = y_train.shape[1]

# network parameters
input_shape = (original_dim, )
intermediate_dim = original_dim * 2
batch_size = 256
latent_dim = original_dim // 4
epochs = 100

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')


# %%
models = (encoder, decoder)

reconstruction_loss = mse(inputs, outputs)

reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()
plot_model(vae,
           to_file='vae.png',
           show_shapes=True)

early_stopping = EarlyStopping(monitor='loss',
                                       min_delta=0.1,
                                       patience=5,
                                       verbose=1,
                                       mode='auto')

# train the autoencoder
vae.fit(y_train_arr,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping]
        #validation_data = (y_test, None)
        )


## %%
nan_row_idx = np.where(np.isnan(np.sum(X_test_arr, axis=1)))
x_miss_val = X_test_arr[nan_row_idx[0], :]

nan_idx = np.where(np.isnan(x_miss_val))
x_miss_val[nan_idx] = -1

for _ in range(25):
    _, _, z_sample = encoder.predict(x_miss_val, batch_size=batch_size)
    test_decoded = decoder.predict(z_sample)
    x_miss_val[nan_idx] = test_decoded[nan_idx]

X_test_arr[nan_row_idx, :] = x_miss_val


## %%
# impute missing values
imputed_arr = X_test_arr

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
imputed_df = pd.concat([imputed_num_subset, cat_subset], axis=1)


## %%
X_test = df_incomplete.loc[X_test.index]
y_test = df_no_nan.loc[y_test.index]
y_test.reset_index(drop=True, inplace=True)


## %%
imputed_df = imputed_df[y_test.columns]


## %%
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


## %%
# Categorical error (accuracy)
cat_nan_mask = X_test[cat_cols].apply(pd.isnull)
accuracy = helper.masked_accuracy(
    y_test[cat_cols], imputed_df[cat_cols], cat_nan_mask)


# %%
print('MSE: {:.3f}'.format(mse))
print('MAE: {:.3f}'.format(mae))
print('Baseline (mean): {:.3f}'.format(baseline))
print('Accuracy: {:.3f}'.format(accuracy))


## %%
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

# %%
