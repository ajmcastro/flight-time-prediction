import numpy as np

### CLOUD COVERAGE ###


def _fix_nan_coverages(df, coverage):
    for i in range(3):
        df[coverage[i]] = df[coverage[i]].replace({'M': np.nan, '///': np.nan})

    return df


def fix_cloud_data(df):
    ori_cov = ['origin_cloud_coverage_lvl_1',
               'origin_cloud_coverage_lvl_2',
               'origin_cloud_coverage_lvl_3']

    dest_cov = ['destination_cloud_coverage_lvl_1',
                'destination_cloud_coverage_lvl_2',
                'destination_cloud_coverage_lvl_3']

    df = _fix_nan_coverages(df, ori_cov)
    df = _fix_nan_coverages(df, dest_cov)

    return df

### FLEET ###


def _join_nb_fleets(df):
    df['fleet'] = np.where(
        (df['fleet'] == 'PGA') | (df['fleet'] == 'WI'), 'NB', df['fleet'])
    return df


def _fill_missing_fleet(df):
    nb_codes = ['M83', '738', '752', '737',
                '757', '734', 'BEH', 'ER4', '100', 'AT4']
    df.loc[df['aircraft_type_code'].isin(nb_codes), 'fleet'] = 'NB'

    wb_codes = ['772', '763', '744', '342', '313', '345']
    df.loc[df['aircraft_type_code'].isin(wb_codes), 'fleet'] = 'WB'
    return df


def adjust_fleets(df):
    # changing NB fleets to NB
    df = _join_nb_fleets(df)
    # changing NaN fleets to proper ones (domain expert feedback)
    df = _fill_missing_fleet(df)
    return df

### DELAY CODES ###


def build_delay_codes(df):
    df.loc[df['delay_code'].isna()
           & df['delay_sub_code'].isna(), 'delay_code'] = 0

    subset = df.loc[df['delay_code'].notna() & df['delay_sub_code'].notna()]
    df.loc[df['delay_code'].notna() & df['delay_sub_code'].notna(), 'delay_code'] = np.where(
        subset['delay_sub_code'] != '0',
        subset['delay_code'].astype(int).astype(
            str) + subset['delay_sub_code'].astype(str),
        subset['delay_code'].astype(int).astype(str))
    df.drop('delay_sub_code', axis=1, inplace=True)
    return df


### ALL ###
def remove_cols_nan_based(df, threshold):
    """Remove columns in DataFrame based on a threshold for filling factor

    Arguments:
        df {DataFrame (pandas)} -- DataFrame with NaN-valued columns
        threshold {float} -- maximum acceptable percentage of NaN values in columns

    Returns:
        DataFrame (pandas) -- DataFrame without columns that exceded the NaN percentage threshold
    """
    df = df.loc[:, df.isna().mean() < threshold]
    return df

### OUTLIER DELETION ###


def weather_outlier_removal(df):
    def outlier_removal(df, attr_name, threshold):
        mask = df[attr_name] >= threshold
        print(
            f'{attr_name} has {df[mask].shape[0]} values greater than {threshold}.')
        return df[~mask]

    df = outlier_removal(df, 'origin_wind_speed', 30)
    df = outlier_removal(df, 'destination_wind_speed', 30)

    df = outlier_removal(df, 'origin_cloud_height', 10000)
    print(df[df['origin_cloud_height'] == 0].shape[0])
    df.drop(df[df['origin_cloud_height'] == 0].index, axis=0, inplace=True)
    df = outlier_removal(df, 'destination_cloud_height', 10000)
    print(df[df['destination_cloud_height'] == 0].shape[0])
    df.drop(df[df['destination_cloud_height']
               == 0].index, axis=0, inplace=True)

    df = outlier_removal(df, 'destination_visibility', 100)
    print(df[df['destination_visibility'] == 0].shape[0])
    df.drop(df[df['destination_visibility'] == 0].index, axis=0, inplace=True)

    return df

### CATEGORICAL BINNING ###


def categorical_binning(df, attr_name):
    counts = df[attr_name].value_counts()
    mask = (counts/counts.sum() * 100).lt(1)
    df[attr_name] = np.where(df[attr_name].isin(
        counts[mask].index), 'other', df[attr_name])
    return df


"""
# Delay code binning
counts = df['delay_code'].value_counts()
mask = (counts/counts.sum() * 100).lt(1)
df['delay_code'] = np.where(df['delay_code'].isin(
    counts[mask].index), 'other', df['delay_code'])

# Origin airport binning
counts = df['origin_airport'].value_counts()
mask = (counts/counts.sum() * 100).lt(1)
df['origin_airport'] = np.where(df['origin_airport'].isin(
    counts[mask].index), 'other', df['origin_airport'])

# Destination airport binning
counts = df['destination_airport'].value_counts()
mask = (counts/counts.sum() * 100).lt(1)
df['destination_airport'] = np.where(df['destination_airport'].isin(
    counts[mask].index), 'other', df['destination_airport'])


# delay_code             242
# origin_airport         129
# destination_airport    151

# then
# delay_code             20
# origin_airport         15
# destination_airport    16
"""

### NUMERICAL BINNING ###


def wind_dir_binning(df):
    def degrees_to_cardinal(d):
        if np.isnan(d):
            return d
        '''
        note: this is highly approximate...
        '''
        dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        ix = int((d + 11.25)/22.5 - 0.02)
        return dirs[ix % 16]
    df['origin_wind_direction'] = df['origin_wind_direction'].apply(
        degrees_to_cardinal)
    df['destination_wind_direction'] = df['destination_wind_direction'].apply(
        degrees_to_cardinal)
    return df
