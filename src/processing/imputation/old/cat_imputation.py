# %%
import numpy as np
import pandas as pd

from src import consts as const
from src.processing.imputation import utils
from src.processing.imputation.auto_encoder import Autoencoder


def build_categorical_dataset(data, dummy_cat_cols, cat_cols):
    mle_complete = utils.maximum_likelihood_categorical(
        data, dummy_cat_cols)

    mle_df = pd.DataFrame(data=mle_complete, columns=dummy_cat_cols)
    cat_df = utils.reverse_dummy(mle_df)
    return cat_df[cat_cols]


# %%
# Read data
x_train, x_test, train_mask, test_mask, original_df_no_nan, original_df = utils.autoencoder_data(
    const.PROCESSED_DATA_DIR / 'simple_delay_code.csv')

full_dummy = pd.get_dummies(original_df)

# %%
# Find categorical columns
cat_cols = list(original_df_no_nan.select_dtypes(np.object))

# Find numerical columns
num_cols = list(original_df_no_nan.select_dtypes(np.number))
num_numerical = len(num_cols)

# %%
# Subset categorical data only
x_train = x_train.iloc[:, num_numerical:]
x_test = x_test.iloc[:, num_numerical:]
full_dummy = full_dummy.iloc[:, num_numerical:]

# %%
imputer = Autoencoder(x_train.values)
#complete_encoded = imputer.train(train_epochs=300, batch_size=256)
complete_encoded = imputer.train()

# %%
train_dummy_cat_cols = x_train.columns
pred_cat_df = build_categorical_dataset(
    complete_encoded, train_dummy_cat_cols, cat_cols)

# %%
# TRAIN CATEGORICAL ACCURACY
cat_true_df = original_df_no_nan.loc[x_train.index, cat_cols].reset_index(
    drop=True)
cat_mask = train_mask[cat_cols].reset_index(drop=True)

accuracy = utils.categorical_accuracy(pred_cat_df, cat_true_df, cat_mask)

print("Train Accuracy: {0:.3f}".format(accuracy))

# %%
######## TESTING ########
# %%
imputer = Autoencoder(x_train.values)
imputer.recreate_model(const.MODEL_DIR / 'cat_encoder_100_256.h5')

# %%
pred_testing = imputer.infer(x_test.values)

# %%
test_dummy_cat_cols = x_test.columns
test_cat_df = build_categorical_dataset(
    pred_testing, test_dummy_cat_cols, cat_cols)

# %%
# TEST CATEGORICAL ACCURACY
cat_true_df = original_df_no_nan.loc[x_test.index, cat_cols].reset_index(
    drop=True)
cat_mask = test_mask[cat_cols].reset_index(drop=True)

test_accuracy = utils.categorical_accuracy(test_cat_df, cat_true_df, cat_mask)

print("Test Accuracy: {0:.3f}".format(test_accuracy))

# %%
######## INFERENCE ########
pred_encoded = imputer.infer(full_dummy.values)

# %%
dummy_cat_cols = x_train.columns
pred_cat_df = build_categorical_dataset(pred_encoded, dummy_cat_cols, cat_cols)

# %%
cat_original = original_df[cat_cols]
actual_cat_mask = cat_original.apply(pd.isnull)
cat_original[actual_cat_mask] = pred_cat_df[actual_cat_mask]

# %%
cat_original.to_csv(const.PROCESSED_DATA_DIR / 'predicted_cat.csv',
                   sep='\t', encoding='utf-8')
