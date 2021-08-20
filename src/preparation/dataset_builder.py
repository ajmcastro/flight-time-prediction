"""Dataset building with flight and weather data."""

import multiprocessing
import os
import re

import pandas as pd

from src import consts as const
from src.preparation import utils, weather_scraper


def build_weather_row(airport, block_date):
    """Extracts weather data for a given airport IATA code and date.

    Arguments:
        airport {string} -- IATA code for airport
        block_date {datetime} -- datetime (e.g., off block or on block dates)

    Raises:
        err: raises exception if weather data download fails

    Returns:
        Series (pandas) -- Series with weather data
    """
    icao_code = utils.get_icao_from_iata(airport)
    start_timestamp = pd.to_datetime(block_date)
    try:
        return weather_scraper.get_weather_data(icao_code, start_timestamp)
    except ValueError as err:
        raise err


def get_weather_cols_names():
    """Builds list with column names for weather data at origin
    and destination airports.

    Returns:
        list -- list of origin and destination weather column names
    """
    weather_cols = ['air_temperature', 'wind_direction', 'wind_speed', 'visibility', 'wind_gust',
                    'cloud_coverage_lvl_1', 'cloud_coverage_lvl_2', 'cloud_coverage_lvl_3', 'cloud_height_lvl_1',
                    'cloud_height_lvl_2', 'cloud_height_lvl_3', 'ice_accretion_1h', 'ice_accretion_3h',
                    'ice_accretion_6h', 'peak_wind_gust', 'peak_wind_direction']
    origin_weather_cols = ['origin_' + c for c in weather_cols]
    destination_weather_cols = ['destination_' + c for c in weather_cols]
    return origin_weather_cols + destination_weather_cols


def get_row_weather(row):
    """Builds DataFrame with weather data from origin and destination
    airports at off block and on block dates, respectively.

    Arguments:
        row {Series (pandas)} -- record of flight on-time performance dataset

    Raises:
        err: raises exception if weather data download fails

    Returns:
        array (numpy) -- weather values for origin and destination airports
    """
    try:
        origin_weather = build_weather_row(
            row['origin_airport'], row['off_block_date'])
        origin_weather = origin_weather.add_prefix('{}_'.format('origin'))
        destination_weather = build_weather_row(
            row['destination_airport'], row['on_block_date'])
        destination_weather = destination_weather.add_prefix(
            '{}_'.format('destination'))
        return pd.concat([origin_weather, destination_weather]).values
    except ValueError as err:
        raise err


def remove_irrelevant_attr(flight_data):
    """Removes data in the DataFrame not relevant for the predictive task.
    Such as irrelevant columns, duplicates and surface vehicle activities' records.

    Arguments:
        flight_data {DataFrame (pandas)} -- DataFrame with irrelevant data

    Returns:
        DataFrame (pandas) -- cleaned DataFrame
    """
    # removing irrelevant attributes
    flight_data.drop(['Unnamed: 0', 'CARR_CD', 'EET', 'TAXI_TIME_OUT', 'TAXI_TIME_IN',
                      'EST_OFFBLK_DATE', 'EST_AIRB_DATE', 'EST_LNDNG_DATE', 'EST_ONBLK_DATE', 'BUS_SEATS',
                      'BUS_SALE_SEATS', 'ECON_SEATS', 'ECON_SALE_SEATS', 'FLEG_STAT', 'DLY_REMARKS',
                      'DLY_DESCRIPTION'], axis=1, inplace=True)
    # removing duplicates
    flight_data.drop_duplicates(keep='first', inplace=True)
    # removing earlier delay records
    flight_data = flight_data.sort_values('DLY_REGISTERED_DATE_TIME', ascending=False).drop_duplicates(
        ['FLT_NBR', 'FROM_AIRP_CD', 'TO_AIRP_CD', 'SCHD_DEP_DATE']).sort_index()
    # removing surface vehicles
    flight_data.drop(flight_data.loc[flight_data['DESCR']
                                     == 'Surface Vehicle'].index, axis=0, inplace=True)
    return flight_data


def adjust_flight_data_types(flight_data):
    """Adjust types of flight on-time performance data in DataFrame.

    Arguments:
        flight_data {DataFrame (pandas)} -- DataFrame with mismatching data types

    Returns:
        DataFrame (pandas) -- DataFrame with adjusted data types
    """
    date_cols = ['FLT_DATE', 'SCHD_DEP_DATE', 'ACTL_OFFBLK_DATE', 'ACTL_AIRB_DATE',
                 'ACTL_LNDNG_DATE', 'ACTL_ONBLK_DATE', 'SCHD_ARR_DATE', 'DLY_REGISTERED_DATE_TIME']
    flight_data[date_cols] = flight_data[date_cols].apply(
        pd.to_datetime, errors='coerce')
    return flight_data


def adjust_weather_data_types(df):
    """Adjust types of weather data in DataFrame.

    Arguments:
        df {DataFrame (pandas)} -- DataFrame with mismatching data types

    Returns:
        DataFrame (pandas) -- DataFrame with adjusted data types
    """
    numeric_cols = ['air_temperature', 'wind_direction', 'wind_speed', 'visibility',
                    'wind_gust', 'cloud_height_lvl_1', 'cloud_height_lvl_2', 'cloud_height_lvl_3',
                    'ice_accretion_1h', 'ice_accretion_3h', 'ice_accretion_6h', 'peak_wind_gust',
                    'peak_wind_direction']
    origin_cols = ['origin_' + c for c in numeric_cols]
    destination_cols = ['destination_' + c for c in numeric_cols]
    joined_cols = origin_cols + destination_cols
    df[joined_cols] = df[joined_cols].apply(pd.to_numeric, errors='coerce')
    return df


def iterations_to_skip():
    """Search file system for existing file chunks to skip when
    restarting the weather extraction proccess.

    Returns:
        list -- list with the already computed final indexes of each chunk
    """
    folder_path = const.WEATHER_DIR
    folder_files = [filenames for _, _, filenames in os.walk(folder_path)]
    skip_starts = []
    for file_name in folder_files[0]:
        m = re.search('backup_(.+?)-', file_name)
        if m:
            found = m.group(1)
            skip_starts.append(int(found))
    return skip_starts


def chunk_processing(df):
    """Extract, parse, clean and impute weather data into dataset subset.

    Arguments:
        df {DataFrame (pandas)} -- subset of main dataset with flight on-time
        performance data

    Raises:
        err: raises exception if weather data download fails

    Returns:
        DataFrame (pandas) -- subset with flight and weather data
    """
    print("Starting #{} compilation...".format(df.iloc[0].name))
    joined_cols = get_weather_cols_names()
    for index, row in df.iterrows():
        try:
            df.loc[index, joined_cols] = get_row_weather(row)
        except ValueError as err:
            raise err
    df = adjust_weather_data_types(df)
    file_name = 'backup_{}-{}.csv'.format(
        str(df.iloc[0].name), str(df.iloc[-1].name))
    df.to_csv(os.path.join('data', 'weather', file_name),
              sep='\t', encoding='utf-8')
    print("\'{}\' writing finished".format(file_name))
    return df


def build_dataset_chunks():
    """Split dataset in chunks for weather imputation and computes each
    in parallel using multi processing.
    """
    # read flight info file
    flight_data = pd.read_csv(
        const.PROCESSED_DATA_DIR / 'flight_info.csv', sep='\t')

    # pre-processing flight data
    flight_data = remove_irrelevant_attr(flight_data)
    flight_data = adjust_flight_data_types(flight_data)
    # renaming columns
    flight_data.rename(columns={'FLT_DATE': 'flight_date', 'FLT_NBR': 'flight_number', 'AIRC_REG': 'tail_number', 'ACTYP_CD': 'aircraft_type_code',
                                'DESCR': 'aircraft_model', 'FMLY_CD': 'size_code', 'FROM_AIRP_CD': 'origin_airport', 'SCHD_DEP_DATE': 'scheduled_departure_date',
                                'ACTL_OFFBLK_DATE': 'off_block_date', 'ACTL_AIRB_DATE': 'take_off_date', 'TO_AIRP_CD': 'destination_airport', 'ACTL_LNDNG_DATE': 'landing_date',
                                'ACTL_ONBLK_DATE': 'on_block_date', 'SCHD_ARR_DATE': 'scheduled_arrival_date', 'SRVCE_TYP': 'service_type', 'AIRC_OWNR_CD': 'aircraft_owner_code',
                                'DLY_REGISTERED_DATE_TIME': 'registered_delay_date', 'DLY_MINS': 'delay_minutes', 'DLY_CD': 'delay_code', 'DLY_SUB_CD': 'delay_sub_code'}, inplace=True)

    joined_cols = get_weather_cols_names()
    # dataset with new columns
    df = pd.concat([flight_data, pd.DataFrame(
        columns=joined_cols)], sort=False)
    df.reset_index(drop=True, inplace=True)

    chunk_size = 5000
    chunks = [df.iloc[i:i + chunk_size]
              for i in range(0, df.shape[0], chunk_size)]
    skip_iterations = iterations_to_skip()
    undone_chunks = [
        df for df in chunks if df.iloc[0].name not in skip_iterations]

    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)
    pool.map(chunk_processing, undone_chunks)


def main():
    """
    Main for testing purposes.
    """
    try:
        build_dataset_chunks()
    except ValueError as err:
        print(err.args)


if __name__ == '__main__':
    main()
