"Class for Neural Network building."

import keras.backend as K
import tensorflow as tf
from keras.initializers import RandomUniform
from keras.layers import Dense, Dropout, Layer
from keras.models import Sequential
from keras.optimizers import RMSprop


class ModelBuilder():
    def __init__(self):
        pass

    def create_RBF(self, input_dim, initializer=None):
        """Implementation of a Radial Basis Function Network"""
        model = Sequential()
        """
        rbflayer = RBFLayer(10,
                            initializer=initializer,
                            betas=2.0,
                            input_shape=(input_dim,))
        """
        rbflayer = RBFLayer(100, initializer=initializer,
                            input_shape=(input_dim,))
        outputlayer = Dense(1, use_bias=False)

        model.add(rbflayer)
        model.add(outputlayer)

        model.compile(loss='mean_squared_error',
                      optimizer=RMSprop())
        return model

    def create_FFNN(self, input_dim=None):
        """Implementation of a Feed-Forward Neural Network"""
        model = Sequential()
        model.add(Dense(
            input_dim * 2, input_dim=input_dim, kernel_initializer='glorot_normal', activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(
            input_dim * 4, kernel_initializer='glorot_normal', activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(
            input_dim * 8, kernel_initializer='glorot_normal', activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(
            input_dim * 4, kernel_initializer='glorot_normal', activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(
            input_dim * 2, kernel_initializer='glorot_normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='glorot_normal'))
        # model.compile(loss='mse', optimizer='adam', metrics=[
        #              'mse', 'mae', 'mape', coeff_determination])
        model.compile(loss='mse', optimizer='adam')
        return model


def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


"""
class RBFLayer(Layer):
    Layer of Gaussian RBF units.

    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
        self.output_dim = output_dim
        # betas is either initializer object or float
        if isinstance(betas, Initializer):
            self.betas_initializer = betas
        else:
            self.betas_initializer = Constant(value=betas)
        self.initializer = initializer if initializer else RandomUniform(
            0.0, 1.0)
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=self.betas_initializer,
                                     # initializer='ones',
                                     trainable=True)
        super().build(input_shape)

    def call(self, x):
        C = tf.expand_dims(self.centers, -1)  # inserts a dimension of 1
        H = tf.transpose(C-tf.transpose(x))  # matrix of differences
        return tf.exp(tf.negative(self.betas) * tf.math.reduce_sum(H**2, axis=1))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
"""


class RBFLayer(Layer):
    """Layer of Gaussian RBF units."""
    def __init__(self, units, initializer=None, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.initializer = initializer if initializer else RandomUniform(
            0.0, 1.0)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(self.units, input_shape[1]),
                                  initializer=self.initializer,
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        C = tf.expand_dims(self.mu, -1)
        H = tf.transpose(C-tf.transpose(inputs))
        return tf.exp(-1 * tf.math.reduce_sum(H**2, axis=1))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        """get_config needs to be defined for model loading to work
        because of custom layer"""
        config = {
            'units': self.units
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
