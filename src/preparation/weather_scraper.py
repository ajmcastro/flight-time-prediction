"""Scraper for weather data from the IEM ASOS service"""

import datetime
import re
import time
from urllib.request import urlopen

import pandas as pd

from src.preparation import utils

MAX_ATTEMPTS = 6
SERVICE = "http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"


def _build_service_uri(station, start_timestamp):
    """Build URI to fetch data.

    Arguments:
        station {string} -- station ICAO code
        start_timestamp {datetime} -- starting day

    Returns:
        string -- URL query string
    """
    end_timestamp = start_timestamp + datetime.timedelta(days=1)

    # output file configuration
    service = SERVICE + "tz=Etc%2FUTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=1&report_type=2&"
    # start date
    service += start_timestamp.strftime('year1=%Y&month1=%m&day1=%d&')
    # one day after
    service += end_timestamp.strftime('year2=%Y&month2=%m&day2=%d&')
    # station
    service += "station=" + station + "&"
    # data
    service += "data=tmpc&data=drct&data=sknt&data=vsby&data=gust&data=skyc1&data=skyc2&data=skyc3&data=skyl1&data=skyl2&data=skyl3&data=ice_accretion_1hr&data=ice_accretion_3hr&data=ice_accretion_6hr&data=peak_wind_gust&data=peak_wind_drct"
    return service


def _download_data(uri):
    """Fetch data from Iowa Environmental Mesonet.

    Arguments:
        uri {string} -- URL to fetch

    Raises:
        ValueError: raises exception if weather data download fails

    Returns:
        string -- weather data in tabular format
    """
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        try:
            data = urlopen(uri, timeout=300).read().decode('utf-8')
            if data is not None and not data.startswith('ERROR'):
                return data
        except Exception as exp:
            print("download_data(%s) failed with %s" % (uri, exp))
            time.sleep(5)
        attempt += 1
    raise ValueError("Exhausted attempts to download data.")


def _parse_data(data):
    """Parse data into a dataframe.

    Arguments:
        data {string} -- Fetched data

    Returns:
        DataFrame (pandas) -- data formatted as a DataFrame
    """
    iterrow = iter(data.split('\n'))
    next(iterrow)  # skipping column names
    row_list = []
    for row in iterrow:
        if not re.match(r'^\s*$', row):  # ignore empty lines
            line = row.split(',')
            line.pop(0)  # removing station
            row_list.append(line)

    df = pd.DataFrame(row_list, columns=['time', 'air_temperature', 'wind_direction', 'wind_speed',
                                         'visibility', 'wind_gust', 'cloud_coverage_lvl_1', 'cloud_coverage_lvl_2', 'cloud_coverage_lvl_3',
                                         'cloud_height_lvl_1', 'cloud height_lvl_2', 'cloud_height_lvl_3', 'ice_accretion_1h', 'ice_accretion_3h',
                                         'ice_accretion_6h', 'peak_wind_gust', 'peak_wind_direction'])
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df


def get_weather_data(station, start_timestamp):
    """Build URI, download, parse and select closer instance to
    starting timestamp.

    Arguments:
      station {string} -- station ICAO code
      start_timestamp {datetime} -- starting day

    Raises:
        ValueError: raises exception if weather data download fails

    Returns:
      Series (pandas) -- closer instance to starting timestamp for given station
    """
    if not station:
        """in case of empty station name"""
        df = pd.DataFrame(columns=['time', 'air_temperature', 'wind_direction', 'wind_speed',
                                   'visibility', 'wind_gust', 'cloud_coverage_lvl_1', 'cloud_coverage_lvl_2', 'cloud_coverage_lvl_3',
                                   'cloud_height_lvl_1', 'cloud height_lvl_2', 'cloud_height_lvl_3', 'ice_accretion_1h', 'ice_accretion_3h',
                                   'ice_accretion_6h', 'peak_wind_gust', 'peak_wind_direction'])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df = df.append(pd.Series(), ignore_index=True)
        return df.iloc[0, :]

    uri = _build_service_uri(station, start_timestamp)
    print('Downloading: %s' % (station, ))
    try:
        data = _download_data(uri)
    except ValueError as err:
        raise err
    df = _parse_data(data)
    if df.empty:
        df = df.append(pd.Series(), ignore_index=True)
        return df.iloc[0, :]
    else:
        nearest = utils.nearest(df.index, start_timestamp)
        return df.loc[df.index == nearest].iloc[0, :]


def main():
    """
    Main for testing purposes.
    """
    startts = datetime.datetime(2012, 8, 1)
    station = ''
    row = get_weather_data(station, startts)
    print(row)


if __name__ == '__main__':
    main()
