
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Dropout
from keras.models import Sequential, load_model
from keras.objectives import mse
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l1_l2


def make_masked_mse(n_features):
    def masked_mse(y_true, y_pred):
        """
        true_values = tf.slice(y_true, [0, 0], [-1, n_features])

        mask_value = tf.slice(y_true, [0, n_features], [-1, -1])
        mask = K.all(K.equal(true_values, mask_value), axis=-1)
        mask = 1 - K.cast(mask, K.floatx())

        masked_true = true_values * mask
        masked_pred = y_pred * mask

        return losses.mean_squared_error(masked_true, masked_pred)
        """

        X_values = y_true[:, :n_features]

        missing_mask = y_true[:, n_features:]
        observed_mask = 1 - missing_mask

        X_values_observed = X_values * observed_mask

        pred_observed = y_pred * observed_mask

        return mse(y_true=X_values_observed, y_pred=pred_observed)
    return masked_mse


class Autoencoder:

    def __init__(self, n_dims,
                 dropout_probability=0.2,
                 init="glorot_normal"):
        self.n_dims = n_dims
        self.dropout_probability = dropout_probability
        self.init = init

    def _create_model(self):
        model = Sequential()

        model.add(Dense(4 * self.n_dims,
                        input_dim=2 * self.n_dims,
                        activation='relu',
                        kernel_initializer=self.init))

        model.add(Dropout(self.dropout_probability))

        model.add(Dense(2 * self.n_dims,
                        activation='relu',
                        kernel_initializer=self.init))
        model.add(Dropout(self.dropout_probability))

        model.add(Dense(self.n_dims,
                        activation='relu',
                        kernel_initializer=self.init))
        model.add(Dropout(self.dropout_probability))

        model.add(Dense(2 * self.n_dims,
                        activation='relu',
                        kernel_initializer=self.init))
        model.add(Dropout(self.dropout_probability))

        model.add(Dense(4 * self.n_dims,
                        activation='relu',
                        kernel_initializer=self.init))
        model.add(Dropout(self.dropout_probability))

        model.add(Dense(self.n_dims,
                        activation='sigmoid',
                        kernel_initializer=self.init))

        self.loss_function = make_masked_mse(self.n_dims)

        model.summary()
        #opt = SGD(lr=0.01, momentum=0.9)
        model.compile(optimizer=RMSprop(), loss=self.loss_function)
        return model

    def train(self, X_train, y_train, X_val, y_val, batch_size=256, epochs=100):
        self.model = self._create_model()
        train_nan_mask = create_missing_mask(X_train)
        X_train[train_nan_mask] = -1

        val_nan_mask = create_missing_mask(X_val)
        X_val[val_nan_mask] = -1

        X_train_masked = np.hstack([X_train, train_nan_mask])
        X_val_masked = np.hstack([X_val, val_nan_mask])
        y_train_masked = np.hstack([y_train, train_nan_mask])
        y_val_masked = np.hstack([y_val, val_nan_mask])

        es = EarlyStopping(monitor='loss', mode='auto',
                           verbose=1, patience=5, min_delta=0.002)
        history = self.model.fit(X_train_masked, y_train_masked, epochs=epochs, batch_size=batch_size,
                                 validation_data=(X_val_masked, y_val_masked), verbose=2, callbacks=[es])
        # self.model.save_weights('autoencoder_weights.h5')
        self._save_model('masked_autoencoder.h5')
        plot_history(history)

    def infer(self, pred_data):
        pred_copy = np.copy(pred_data)
        missing_mask = create_missing_mask(pred_copy)
        pred_copy[missing_mask] = -1
        n_dims = pred_copy.shape[1]

        input_with_mask = np.hstack([pred_copy, missing_mask])

        X_pred = self.model.predict(input_with_mask)
        X_pred = X_pred[:, :n_dims]

        pred_data[missing_mask] = X_pred[missing_mask]

        return pred_data

    def load_model(self, filename):
        return load_model(filename,
                          custom_objects={'masked_mse': self.loss_function})

    def _save_model(self, filename):
        # saving to and loading from file
        print(f"Save model to file {filename} ... ", end="")
        self.model.save(filename)
        print("OK")

        print(f"Load model from file {filename} ... ", end="")
        _ = load_model(filename,
                       custom_objects={'masked_mse': self.loss_function})
        print("OK")


def create_missing_mask(data):
    if data.dtype != "f" and data.dtype != "d":
        data = data.astype(float)

    return np.isnan(data)


def plot_history(history):
    sns.set(palette="Set2")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('mse_autoencoder.png', bbox_inches='tight', facecolor='w')
    plt.show()
