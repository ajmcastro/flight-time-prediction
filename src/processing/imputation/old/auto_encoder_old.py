import random
from collections import defaultdict

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.objectives import mse
from keras.regularizers import l1_l2
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder


def make_reconstruction_loss(n_features):

    def reconstruction_loss(input_and_mask, y_pred):
        X_values = input_and_mask[:, :n_features]
        #X_values.name = "$X_values"

        missing_mask = input_and_mask[:, n_features:]
        #missing_mask.name = "$missing_mask"
        observed_mask = 1 - missing_mask
        #observed_mask.name = "$observed_mask"

        X_values_observed = X_values * observed_mask
        #X_values_observed.name = "$X_values_observed"

        pred_observed = y_pred * observed_mask
        #pred_observed.name = "$y_pred_observed"

        return mse(y_true=X_values_observed, y_pred=pred_observed)
    return reconstruction_loss


def masked_mae(X_true, X_pred, mask):
    masked_diff = X_true[mask] - X_pred[mask]
    return np.mean(np.abs(masked_diff))


class Autoencoder:

    def __init__(self, data,
                 recurrent_weight=0.5,
                 optimizer="adam",
                 dropout_probability=0.5,
                 hidden_activation="relu",
                 output_activation="sigmoid",
                 init="glorot_normal",
                 l1_penalty=0,
                 l2_penalty=0):
        self.data = data.copy()
        self.recurrent_weight = recurrent_weight
        self.optimizer = optimizer
        self.dropout_probability = dropout_probability
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.init = init
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

    def _get_hidden_layer_sizes(self):
        n_dims = self.data.shape[1]
        return [
            min(2000, 8 * n_dims),
            min(500, 2 * n_dims),
            int(np.ceil(0.5 * n_dims)),
        ]

    def _create_model(self):

        hidden_layer_sizes = self._get_hidden_layer_sizes()
        first_layer_size = hidden_layer_sizes[0]
        n_dims = self.data.shape[1]

        model = Sequential()

        model.add(Dense(
            first_layer_size,
            input_dim=2 * n_dims,
            activation=self.hidden_activation,
            W_regularizer=l1_l2(self.l1_penalty, self.l2_penalty),
            init=self.init))
        model.add(Dropout(self.dropout_probability))

        for layer_size in hidden_layer_sizes[1:]:
            model.add(Dense(
                layer_size,
                activation=self.hidden_activation,
                W_regularizer=l1_l2(self.l1_penalty, self.l2_penalty),
                init=self.init))
            model.add(Dropout(self.dropout_probability))

        model.add(Dense(
            n_dims,
            activation=self.output_activation,
            W_regularizer=l1_l2(self.l1_penalty, self.l2_penalty),
            init=self.init))

        loss_function = make_reconstruction_loss(n_dims)

        model.compile(optimizer=self.optimizer, loss=loss_function)
        return model

    def fill(self, missing_mask):
        self.data[missing_mask] = -1

    def _create_missing_mask(self):
        if self.data.dtype != "f" and self.data.dtype != "d":
            self.data = self.data.astype(float)

        return np.isnan(self.data)

    def _train_epoch(self, model, missing_mask, batch_size):
        input_with_mask = np.hstack([self.data, missing_mask])
        n_samples = len(input_with_mask)
        n_batches = int(np.ceil(n_samples / batch_size))
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_shuffled = input_with_mask[indices]

        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx + 1) * batch_size
            batch_data = X_shuffled[batch_start:batch_end, :]
            model.train_on_batch(batch_data, batch_data)
        return model.predict(input_with_mask)

    def train(self, batch_size=256, train_epochs=100):
        missing_mask = self._create_missing_mask()
        self.fill(missing_mask)
        self.model = self._create_model()

        observed_mask = ~missing_mask

        for epoch in range(train_epochs):
            X_pred = self._train_epoch(self.model, missing_mask, batch_size)
            observed_mae = masked_mae(X_true=self.data,
                                      X_pred=X_pred,
                                      mask=observed_mask)
            if epoch % 50 == 0:
                print("observed mae:", observed_mae)

            old_weight = (1.0 - self.recurrent_weight)
            self.data[missing_mask] *= old_weight
            pred_missing = X_pred[missing_mask]
            self.data[missing_mask] += self.recurrent_weight * pred_missing
        return self.data.copy()
