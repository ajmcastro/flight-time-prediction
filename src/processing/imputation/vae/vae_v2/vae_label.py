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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as a function of the 2D latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = (n - 1) * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


## %%
# Read data
df_incomplete, df_no_nan, df = helper.imputation_dataset(
    const.PROCESSED_DATA_DIR / 'enhanced_with_nan.csv')


## %%
# Split data
_, X_test, y_train, y_test = train_test_split(
    df_incomplete, df_no_nan, test_size=0.3)


## %%
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
train_scaler = MinMaxScaler().fit(y_train_arr)
y_train_arr = train_scaler.transform(y_train_arr)

test_obs_row_idx = np.where(np.isfinite(np.sum(X_test_arr, axis=1)))
X_test_complete_records = X_test_arr[test_obs_row_idx]
test_scaler = MinMaxScaler().fit(X_test_complete_records)
X_test_arr = test_scaler.transform(X_test_arr)
del X_test_complete_records


# %%
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
# plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
# plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')


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

early_stopping = EarlyStopping(monitor='loss',
                                       min_delta=0.005,
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
# vae.save_weights('vae_mlp_mnist.h5')

# plot_results(models,
#             data,
#             batch_size = batch_size,
#             model_name = "vae_mlp")


# %%
nan_row_idx = np.where(np.isnan(np.sum(X_test_arr, axis=1)))
x_miss_val = X_test_arr[nan_row_idx[0], :]

nan_idx = np.where(np.isnan(x_miss_val))
x_miss_val[nan_idx] = -1

for _ in range(25):
    _, _, z_sample = encoder.predict(x_miss_val, batch_size=batch_size)
    test_decoded = decoder.predict(z_sample)
    x_miss_val[nan_idx] = test_decoded[nan_idx]

X_test_arr[nan_row_idx, :] = x_miss_val


# %%
# impute missing values
imputed_arr = X_test_arr

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
imputed_df = imputed_df[y_test.columns]


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
