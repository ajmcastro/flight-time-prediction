# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src import consts as const
from src import utils
from src.processing.imputation.auto_encoder import Autoencoder

# %%
# Read data
dummy_df_inc, dummy_df_no_nan, df_incomplete, original_df = utils.autoencoder_data(
    const.PROCESSED_DATA_DIR / 'simple_delay_code.csv')

# %%
dummy_df_full = pd.get_dummies(original_df)
aux = pd.get_dummies(original_df.drop(utils.IRRELEVANT_INPUT, axis=1))

# %%
num_cols = list(df_incomplete.select_dtypes(np.number))
num_numerical = len(num_cols)

cat_cols = list(df_incomplete.select_dtypes(np.object))
dummy_cat_cols = aux.iloc[:, num_numerical:].columns

# %%
X_train, y_train, X_val, y_val, X_test, y_test = utils.split_data(
    dummy_df_inc, dummy_df_no_nan)

# %%
X_train, y_train, X_val, y_val, X_test, y_test = utils.scale_data(
    X_train, y_train, X_val, y_val, X_test, y_test)

# %%
# network parameters
batch_size, n_epoch = 100, 100
n_hidden, z_dim = 256, 2

# %%
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from scipy.stats import norm

# %%

# encoder
x = Input(shape=(y_train.shape[1:]))
x_encoded = Dense(n_hidden, activation='relu')(x)
x_encoded = Dense(n_hidden//2, activation='relu')(x_encoded)

mu = Dense(z_dim)(x_encoded)
log_var = Dense(z_dim)(x_encoded)

# %%
# sampling function
def sampling(args):
    mu, log_var = args
    eps = K.random_normal(shape=(batch_size, z_dim), mean=0., stddev=1.0)
    return mu + K.exp(log_var) * eps

z = Lambda(sampling, output_shape=(z_dim,))([mu, log_var])

#%%
# decoder
z_decoder1 = Dense(n_hidden//2, activation='relu')
z_decoder2 = Dense(n_hidden, activation='relu')
y_decoder = Dense(y_train.shape[1], activation='sigmoid')

z_decoded = z_decoder1(z)
z_decoded = z_decoder2(z_decoded)
y = y_decoder(z_decoded)

# %%
# loss
reconstruction_loss = objectives.binary_crossentropy(x, y) * y_train.shape[1]
kl_loss = 0.5 * K.sum(K.square(mu) + K.exp(log_var) - log_var - 1, axis = -1)
vae_loss = reconstruction_loss + kl_loss

# build model
vae = Model(x, y)
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()

# %%
# train
vae.fit(y_train,
       shuffle=True,
       epochs=n_epoch,
       batch_size=batch_size,
       validation_data=(y_test, None), verbose=1)

# %%
# build encoder
encoder = Model(x, mu)
encoder.summary()

# %%
# Plot of the digit classes in the latent space
x_te_latent = encoder.predict(y_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_te_latent[:, 0], x_te_latent[:, 1], c=y_te)
plt.colorbar()
plt.show()

# %%
# build decoder
decoder_input = Input(shape=(z_dim,))
_z_decoded = z_decoder1(decoder_input)
_z_decoded = z_decoder2(_z_decoded)
_y = y_decoder(_z_decoded)
generator = Model(decoder_input, _y)
generator.summary()























# %%
"""
train_mask = 1 - np.isnan(X_train)
val_mask = 1 - np.isnan(X_val)
test_mask = 1 - np.isnan(X_test)

# %%


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# Hint Vector Generation


def sample_M(m, n, p):
    A = np.random.uniform(0., 1., size=[m, n])
    B = A > p
    C = 1.*B
    return C


# %% GAIN Architecture
# Hidden state dimensions
H_Dim1 = X_train.shape[1]
H_Dim2 = X_train.shape[1]

# %% 1. Input Placeholders
# 1.1. Data Vector
X = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]])
# 1.2. Mask Vector
M = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]])
# 1.3. Hint vector
H = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]])
# 1.4. X with missing values
New_X = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]])

# %% 2. Discriminator
# Data + Hint as inputs
D_W1 = tf.Variable(xavier_init([X_train.shape[1]*2, H_Dim1]))
D_b1 = tf.Variable(tf.zeros(shape=[H_Dim1]))

D_W2 = tf.Variable(xavier_init([H_Dim1, H_Dim2]))
D_b2 = tf.Variable(tf.zeros(shape=[H_Dim2]))

D_W3 = tf.Variable(xavier_init([H_Dim2, X_train.shape[1]]))
# Output is multi-variate
D_b3 = tf.Variable(tf.zeros(shape=[X_train.shape[1]]))

theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

# %% 3. Generator
# Data + Mask as inputs (Random Noises are in Missing Components)
G_W1 = tf.Variable(xavier_init([X_train.shape[1]*2, H_Dim1]))
G_b1 = tf.Variable(tf.zeros(shape=[H_Dim1]))

G_W2 = tf.Variable(xavier_init([H_Dim1, H_Dim2]))
G_b2 = tf.Variable(tf.zeros(shape=[H_Dim2]))

G_W3 = tf.Variable(xavier_init([H_Dim2, X_train.shape[1]]))
G_b3 = tf.Variable(tf.zeros(shape=[X_train.shape[1]]))

theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

# %% 1. Generator


def generator(new_x, m):
    inputs = tf.concat(axis=1, values=[new_x, m])  # Mask + Data Concatenate
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    # [0,1] normalized Output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)

    return G_prob

# %% 2. Discriminator


def discriminator(new_x, h):
    inputs = tf.concat(axis=1, values=[new_x, h])  # Hint + Data Concatenate
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)  # [0,1] Probability Output

    return D_prob

# %% 3. Other functions
# Random sample generator for Z


def sample_Z(m, n):
    return np.random.uniform(0., 0.01, size=[m, n])

# Mini-batch generation


def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx


# %% Structure
# Generator
G_sample = generator(New_X, M)

# Combine with original data
Hat_New_X = New_X * M + G_sample * (1-M)

# Discriminator
D_prob = discriminator(Hat_New_X, H)

# %% Loss
D_loss1 = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) +
                          (1-M) * tf.log(1. - D_prob + 1e-8))
G_loss1 = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))
MSE_train_loss = tf.reduce_mean(
    (M * New_X - M * G_sample)**2) / tf.reduce_mean(M)

D_loss = D_loss1
G_loss = G_loss1 + 10 * MSE_train_loss

# %% MSE Performance metric
MSE_test_loss = tf.reduce_mean(
    ((1-M) * X - (1-M)*G_sample)**2) / tf.reduce_mean(1-M)

# %% Solver
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

# Sessions
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# %% Start Iterations
for it in tqdm(range(5000)):

    # %% Inputs
    mb_idx = sample_idx(y_train.shape[0], 256)
    X_mb = y_train[mb_idx, :]

    Z_mb = sample_Z(256, X_train.shape[1])
    M_mb = train_mask[mb_idx, :]
    H_mb1 = sample_M(256, X_train.shape[1], 1-0.9)
    H_mb = M_mb * H_mb1

    New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce

    _, D_loss_curr = sess.run([D_solver, D_loss1], feed_dict={
                              M: M_mb, New_X: New_X_mb, H: H_mb})
    _, G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = sess.run([G_solver, G_loss1, MSE_train_loss, MSE_test_loss],
                                                                       feed_dict={X: X_mb, M: M_mb, New_X: New_X_mb, H: H_mb})

    # %% Intermediate Losses
    if it % 100 == 0:
        print('Iter: {}'.format(it))
        print('Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr)))
        print('Test_loss: {:.4}'.format(np.sqrt(MSE_test_loss_curr)))
        print()


# %% Final Loss

Z_mb = sample_Z(X_test.shape[0], X_train.shape[1])
M_mb = test_mask
X_mb = y_test

New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce

MSE_final, Sample = sess.run([MSE_test_loss, G_sample], feed_dict={
                             X: y_test, M: test_mask, New_X: New_X_mb})

print('Final Test RMSE: ' + str(np.sqrt(MSE_final)))

print(Sample)

# %%
Z_mb = sample_Z(X_test.shape[0], X_train.shape[1])
M_mb = test_mask
X_mb = y_test

New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce

_, complete_encoded = sess.run([MSE_test_loss, G_sample], feed_dict={
                               X: X_test, M: test_mask, New_X: New_X_mb})
print(X_test)
print(complete_encoded)
"""
# %%
# Normalize data
dummy_incomplete = utils.normalize(dummy_incomplete)

# %%
mask = dummy_incomplete.apply(pd.isnull)
y = full_imp.iloc[dummy_incomplete.index]

# %%
# x_train and y_train are slightly different because of normalization
x_train, x_test, y_train, y_test = train_test_split(
    dummy_incomplete, y, test_size=0.3)

# %%
# Build Autoencoder input
train_mask = mask.loc[x_train.index]
test_mask = mask.loc[x_test.index]

# %%
x_train[train_mask] = -1
x_test[test_mask] = -1

# %%
#x_train = np.hstack([x_train, train_mask])
#y_train = np.hstack([y_train, train_mask])
#x_test = np.hstack([x_test, test_mask])
#y_test = np.hstack([y_test, test_mask])

# %%
imputer = Autoencoder(x_train.shape[1])
imputer.train(x_train.values, y_train.values,
              x_test.values, y_test.values, epochs=20)

# %%
imputer.model.save('test.h5')

# %%
imputer = Autoencoder(y_train.shape[1])
imputer.model = load_model('test.h5')

# %%
complete_encoded = imputer.model.predict(x_test)

# %%
score = imputer.model.evaluate(x_test, y_test, verbose=0)
print('Test MSE:', score[0])

# %%
categorical_data = complete_encoded[:, num_numerical:]
numerical_data = complete_encoded[:, :num_numerical]

# %%
# Maximum likelihood categorical data
mle_complete = utils.maximum_likelihood_categorical(
    categorical_data, dummy_cat_cols)

# %%
# Build predicted dataframes
mle_df = pd.DataFrame(data=mle_complete, columns=dummy_cat_cols)
pred_cat_df = utils.reverse_dummy(mle_df)
pred_cat_df = pred_cat_df[cat_cols]

pred_num_df = pd.DataFrame(
    data=numerical_data, columns=num_cols)

# %%
missing = df_incomplete.apply(pd.isnull)
train_mask = missing.loc[x_train.index]
test_mask = missing.loc[x_test.index]

# %%
######## TRAIN CATEGORICAL ACCURACY ########
# Checking categorical attributes correctness
cat_true_df = original_df.loc[x_train.index, cat_cols].reset_index(drop=True)
cat_mask = train_mask[cat_cols].reset_index(drop=True)

accuracy = utils.categorical_accuracy(pred_cat_df, cat_true_df, cat_mask)

print("Train Accuracy: {0:.3f}".format(accuracy))

# %%
######## TRAIN NUMERICAL MSE ########
num_true_df = original_df.loc[x_train.index, num_cols].reset_index(drop=True)
num_mask = train_mask[num_cols].reset_index(drop=True)

mse = utils.numerical_mse(pred_num_df, num_true_df, num_mask)

print("Train MSE: {0:.3f}".format(mse))

# %%
######## TESTING ########
# %%
imputer = Autoencoder(x_train.values)
imputer.recreate_model('encoder_100_256.h5')

# %%
pred_testing = imputer.infer(x_test.values)

# %%
test_dummy_cat_cols = x_test.iloc[:, num_numerical:].columns
test_mle_complete = utils.maximum_likelihood_categorical(
    pred_testing, test_dummy_cat_cols, num_numerical)

# %%
# Build predicted dataframes
test_mle_df = pd.DataFrame(data=test_mle_complete, columns=test_dummy_cat_cols)
test_cat_df = utils.reverse_dummy(test_mle_df)
test_cat_df = test_cat_df[cat_cols]

test_num_df = pd.DataFrame(
    data=pred_testing[:, :num_numerical], columns=num_cols)

# %%
######## TEST CATEGORICAL ACCURACY ########
# Checking categorical attributes correctness
cat_true_df = original_df.loc[x_test.index, cat_cols].reset_index(
    drop=True)
cat_mask = test_mask[cat_cols].reset_index(drop=True)

test_accuracy = utils.categorical_accuracy(test_cat_df, cat_true_df, cat_mask)

print("Test Accuracy: {0:.3f}".format(test_accuracy))

# %%
######## TEST NUMERICAL MSE ########
num_true_df = original_df.loc[x_test.index, num_cols].reset_index(
    drop=True)
num_mask = test_mask[num_cols].reset_index(drop=True)

test_mse = utils.numerical_mse(test_num_df, num_true_df, num_mask)

print("Test MSE: {0:.3f}".format(test_mse))

# %%
######## INFERENCE ########
pred_encoded = imputer.infer(full_dummy.values)

# %%
dummy_cat_cols = x_train.iloc[:, num_numerical:].columns
pred_mle_complete = utils.maximum_likelihood_categorical(
    pred_encoded, dummy_cat_cols, num_numerical)

# %%
# Reverse dummy categorical data
pred_mle_df = pd.DataFrame(data=pred_mle_complete, columns=dummy_cat_cols)
pred_cat_df = utils.reverse_dummy(pred_mle_df)
pred_cat_df = pred_cat_df[cat_cols]

# %%
pred_num_df = pd.DataFrame(
    data=pred_encoded[:, :num_numerical], columns=num_cols)

# %%
pred_dataset = pd.concat([pred_num_df, pred_cat_df], axis=1)

# %%
df_no_nan = original_df.dropna()
df_no_nan.to_csv(const.PROCESSED_DATA_DIR / 'nan_dropped.csv',
                 sep='\t', encoding='utf-8', index=False)
pred_dataset.to_csv(const.PROCESSED_DATA_DIR / 'nan_predicted.csv',
                    sep='\t', encoding='utf-8', index=False)


# %%
def make_masked_mse(n_features):
    def masked_mse(y_true, y_pred):
        true_values = tf.slice(y_true, [0, 0], [-1, n_features])

        mask_value = tf.slice(y_true, [0, n_features], [-1, -1])
        mask = K.all(K.equal(true_values, mask_value), axis=-1)
        mask = 1 - K.cast(mask, K.floatx())

        masked_true = true_values * mask
        masked_pred = y_pred * mask

        return losses.mean_squared_error(masked_true, masked_pred)
    return masked_mse


# %%
y_true = np.array([[-1, -1, 4, 5, -1], [7, -1, 9, -1, -1], [12,
                                                            13, -1, -1, 16], [17, 18, 19, 20, -1], [22, -1, 24, -1, 26]])
y_pred = np.array([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11], [12, 13, 14, 15, 16], [
                  17, 18, 19, 20, 21], [22, 23, 24, 25, 26]])
mask = np.array([[1, 1, 0, 0, 1], [0, 1, 0, 1, 1], [
                0, 0, 1, 1, 0], [0, 0, 0, 0, 1], [0, 1, 0, 1, 0]])
join_true = np.hstack([y_true, mask])
join_pred = y_pred

# %%
y_true = K.constant(join_true)
y_pred = K.constant(join_pred)
loss = make_masked_mse(5)
mse = loss(y_true, y_pred)
print(tf.Session().run(mse))

# %%
