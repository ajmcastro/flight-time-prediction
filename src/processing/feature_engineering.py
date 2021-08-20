# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import boxcox, boxcox_normmax, norm, probplot, skew
from sklearn.decomposition import PCA

from src import consts as const
from src import utils
from src.processing import attribute_builder as ab
from src.processing import plotting, refactor

sns.set(palette="Set2")


def backwardElimination(x, Y, sl, columns):
    ini = len(columns)
    numVars = x.shape[1]
    for i in range(0, numVars):
        Y = list(Y)
        regressor = sm.OLS(Y, x).fit()
        maxVar = max(regressor.pvalues)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor.pvalues[j].astype(float) == maxVar):
                    columns = np.delete(columns, j)
                    x = x.loc[:, columns]
    print(
        '\nSelect {:d} features from {:d} by best p-values.'.format(len(columns), ini))
    print(
        'The max p-value from the features selected is {:.3f}.'.format(maxVar))
    print(regressor.summary())

    # odds ratios and 95% CI
    conf = np.exp(regressor.conf_int())
    conf['Odds Ratios'] = np.exp(regressor.params)
    conf.columns = ['2.5%', '97.5%', 'Odds Ratios']
    print(conf)

    return columns, regressor


def cumsum_pca(data, title='pca', save=False):
    # Fitting the PCA algorithm
    pca = PCA().fit(data)
    # Plotting the Cumulative Summation of the Explained Variance
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')
    if save:
        plt.savefig(f'{title}.png', bbox_inches='tight', facecolor='w')


df = pd.read_csv(const.PROCESSED_DATA_DIR / 'basic_eda.csv',
                 sep='\t', parse_dates=utils.DATE_COLS)

# %%
# Outlier removal
df = refactor.weather_outlier_removal(df)

# %%
# Numerical Binning
df = refactor.wind_dir_binning(df)


# %%
plotting.simple_bar(df, 'service_type')

# %%
plotting.simple_bar(df, 'aircraft_owner_code', save=True)

# %%
plotting.delay_calculated_source(df, save=True)

# %%
plotting.missing_values(df, save=True)

# %%
# Create attributes
df = ab.create_targets(df)
# df = ab.departure_delay_creation(df)
df = ab.scheduled_block_time_creation(df)
df = ab.is_night_creation(df)
df = ab.create_scheduled_rotation_time(df)
df = ab.create_previous_delay_code(df)

# %%
# Dropping observations with service type different than J
df.drop(df[df['service_type'] != 'J'].index, axis=0, inplace=True)

# %%
# Dropping unnecessary columns
df.drop(['flight_date', 'aircraft_owner_code', 'aircraft_type_code',
         'delay_minutes', 'service_type', 'registered_delay_date',
         'off_block_date', 'take_off_date', 'landing_date', 'on_block_date',
         'delay_code'],
        axis=1, inplace=True)


# %%
# Create dates segmentation
df_dt_cols = df.select_dtypes(include=['datetime64']).columns
df = utils.decompose_dates(df, df_dt_cols)

# %%
# Correcting types
df['flight_number'] = df['flight_number'].astype(int).astype('str')
df['is_night'] = df['is_night'].astype('uint8')

# %%
df.describe().transpose()

#####################################################


def plot_correlations(df, save=False):
    copy = df.copy()
    y = copy['air_time']
    categorical_cols = copy.select_dtypes(include=['object']).columns
    numerical_cols = copy.select_dtypes(include=['number']).columns

    X_aux = utils.encode_categorical(copy, y, encoder_name='leave_one_out')
    targets = ['air_time', 'taxi_in', 'taxi_out', 'actual_block_time']
    # Correlations
    plt.figure(1)
    corr = df[numerical_cols].corr('spearman')
    sns.heatmap(corr, xticklabels=True, yticklabels=True)
    if save:
        plt.savefig('numerical_corr.png', bbox_inches='tight', facecolor='w')
    plt.figure(2)
    corr = X_aux[list(categorical_cols)+targets].corr('spearman')
    sns.heatmap(corr, xticklabels=True, yticklabels=True)
    if save:
        plt.savefig('categorical_corr.png', bbox_inches='tight', facecolor='w')
    plt.figure(3)
    corr = pd.DataFrame(np.zeros([len(numerical_cols)+len(targets), len(categorical_cols)+len(targets)]),
                        index=list(numerical_cols)+targets, columns=list(categorical_cols)+targets)
    for q1 in list(numerical_cols)+targets:
        for q2 in list(categorical_cols)+targets:
            corr.loc[q1, q2] = df[q1].corr(X_aux[q2], method='spearman')
    sns.heatmap(corr, xticklabels=True, yticklabels=True)
    if save:
        plt.savefig('all_corr.png', bbox_inches='tight', facecolor='w')


"""Uncomment to see a feature ranking with recursive feature elimination 
and cross-validated selection of the best number of features. """

"""
cols = X_train.columns.values
scaler = RobustScaler()
df = pd.DataFrame(scaler.fit_transform(X_train[cols]), columns=cols)
ls = Lasso(alpha=0.0005, selection='cyclic',
           tol=0.002, random_state=101)
rfecv = RFECV(estimator=ls, n_jobs=-1, step=1,
              scoring='neg_mean_squared_error', cv=5)
rfecv.fit(df, y_train)

select_features_rfecv = rfecv.get_support()
RFEcv = cols[select_features_rfecv]
print('{:d} Features Select by RFEcv:\n{:}'.format(
    rfecv.n_features_, list(RFEcv)))
"""

# Removal of non-existent features for each predictive task


def get_dataset(df, target):
    df = df.dropna()
    if target == 'air_time':
        df.drop(['taxi_in', 'taxi_out',
                 'actual_block_time'],
                axis=1, inplace=True)
    if target == 'taxi_out':
        df.drop(['destination_cloud_height', 'destination_cloud_coverage',
                 'destination_wind_direction', 'destination_wind_speed',
                 'destination_air_temperature',
                 'destination_visibility', 'actual_block_time',
                 'air_time', 'taxi_in'],
                axis=1, inplace=True)
    if target == 'taxi_in':
        df.drop(['origin_cloud_height', 'origin_cloud_coverage',
                 'origin_wind_direction', 'origin_wind_speed',
                 'origin_air_temperature',
                 'origin_visibility', 'actual_block_time',
                 'air_time', 'taxi_out'],
                axis=1, inplace=True)
    if target == 'actual_block_time':
        df.drop(['taxi_in', 'taxi_out', 'air_time'],
                axis=1, inplace=True)

    y = df.pop(target)
    X = df
    return X, y


"""Uncomment to see a feature ranking with recursive feature elimination 
and cross-validated selection of the best number of features. """

"""
X, y = get_dataset(df, 'air_time')
X_aux = utils.encode_categorical(X, y, encoder_name='leave_one_out')
cols = X_aux.columns.values
X_aux, _ = utils.scale_data(X_aux)
X_aux_df = pd.DataFrame(X_aux, columns=cols)
SL = 0.051
pv_cols, LR = backwardElimination(X_aux_df, y, SL, cols)
"""

"""Uncomment to get cumulative sum of explained variance"""
# cumsum_pca(X_aux_df, "pca_taxi_in", save=True)

# %%
# Save data
df = df.dropna()
df.to_csv(const.PROCESSED_DATA_DIR / 'cleaned_dropped_nan.csv',
          sep='\t', encoding='utf-8', index=False)
