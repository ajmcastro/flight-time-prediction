
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Dropout
from keras.models import Sequential, load_model
from keras.objectives import mse
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l1_l2


class Autoencoder:

    def __init__(self, n_dims,
                 dropout_probability=0.2,
                 init="glorot_normal"):
        self.n_dims = n_dims
        self.dropout_probability = dropout_probability
        self.init = init

    def _create_model(self):
        model = Sequential()

        model.add(Dense(250,
                        input_dim=self.n_dims,
                        activation='relu',
                        kernel_initializer=self.init))
        model.add(Dropout(self.dropout_probability))

        """
        model.add(Dense(2 * self.n_dims,
                        activation='relu',
                        kernel_initializer=self.init))
        model.add(Dropout(self.dropout_probability))
        """

        model.add(Dense(100,
                        activation='relu',
                        kernel_initializer=self.init))
        model.add(Dropout(self.dropout_probability))

        model.add(Dense(250,
                        activation='relu',
                        kernel_initializer=self.init))
        model.add(Dropout(self.dropout_probability))

        """
        model.add(Dense(4 * self.n_dims,
                        activation='relu',
                        kernel_initializer=self.init))
        model.add(Dropout(self.dropout_probability))
        """

        model.add(Dense(self.n_dims,
                        activation='sigmoid',
                        kernel_initializer=self.init))

        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X_train, y_train, X_val, y_val, batch_size=256, epochs=100):
        self.model = self._create_model()
        train_nan_mask = create_missing_mask(X_train)
        X_train[train_nan_mask] = -1

        val_nan_mask = create_missing_mask(X_val)
        X_val[val_nan_mask] = -1

        es = EarlyStopping(monitor='loss', mode='auto',
                           verbose=1, patience=5, min_delta=0.0005)
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(X_val, y_val), verbose=2, callbacks=[es])
        self._save_model('autoencoder.h5')
        plot_history(history)

    def infer(self, pred_data):
        pred_copy = np.copy(pred_data)
        missing_mask = create_missing_mask(pred_copy)
        pred_copy[missing_mask] = -1

        X_pred = self.model.predict(pred_copy)

        pred_data[missing_mask] = X_pred[missing_mask]

        return pred_data

    def _save_model(self, filename):
        # saving to and loading from file
        print(f"Save model to file {filename} ... ", end="")
        self.model.save(filename)
        print("OK")

        print(f"Load model from file {filename} ... ", end="")
        _ = load_model(filename)
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
