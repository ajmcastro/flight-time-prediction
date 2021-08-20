# %%
import numpy as np
import pandas as pd

from src import consts as const
from src.processing.imputation import utils
from src.processing.imputation.auto_encoder import Autoencoder

# %%
# Read data
x_train, x_test,train_mask, test_mask, original_df_no_nan, original_df = utils.autoencoder_data(
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
x_train = x_train.iloc[:, :num_numerical]
x_test = x_test.iloc[:, :num_numerical]
full_dummy = full_dummy.iloc[:, :num_numerical]

# %%
# Normalize data
x_train = utils.normalize(x_train)
x_test = utils.normalize(x_test)
full_dummy = utils.normalize(full_dummy)
original_df[num_cols] = utils.normalize(original_df[num_cols])

# %%
imputer = Autoencoder(x_train.values)
#complete_encoded = imputer.train(train_epochs=300, batch_size=256)
complete_encoded = imputer.train()

# %%
######## TRAIN NUMERICAL MSE ########
pred_num_df = pd.DataFrame(
    data=complete_encoded, columns=num_cols)
num_true_df = original_df_no_nan.loc[x_train.index, num_cols].reset_index(drop=True)
num_mask = train_mask[num_cols].reset_index(drop=True)

mse = utils.numerical_mse(pred_num_df, num_true_df, num_mask)

print("Train MSE: {0:.3f}".format(mse))

# %%
######## TESTING ########
# %%
imputer = Autoencoder(x_train.values)
imputer.recreate_model(const.MODEL_DIR / 'num_encoder_100_256.h5')

# %%
pred_testing = imputer.infer(x_test.values)

# %%
######## TEST NUMERICAL MSE ########
test_num_df = pd.DataFrame(
    data=pred_testing, columns=num_cols)
num_true_df = original_df_no_nan.loc[x_test.index, num_cols].reset_index(
    drop=True)
num_mask = test_mask[num_cols].reset_index(drop=True)

test_mse = utils.numerical_mse(test_num_df, num_true_df, num_mask)

print("Test MSE: {0:.3f}".format(test_mse))

# %%
######## INFERENCE ########
pred_encoded = imputer.infer(full_dummy.values)

# %%
dummy_cat_cols = x_train.columns
pred_num_df = pd.DataFrame(
    data=pred_encoded, columns=num_cols)

# %%
num_original = original_df[num_cols]
actual_num_mask = num_original.apply(pd.isnull)
num_original[actual_num_mask] = pred_num_df[actual_num_mask]

# %%
num_original.to_csv(const.PROCESSED_DATA_DIR / 'predicted_num.csv',
                   sep='\t', encoding='utf-8')



#%%
