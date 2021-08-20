"""Auxiliar methods in data preparation."""

import os
import pandas as pd


def is_float(value):
    """[summary]

    Arguments:
        value {primitive type} -- value

    Returns:
        bool -- Float value check truth value
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def nearest(items, pivot):
    """Get closest item, in list of items, to pivot

    Arguments:
        items {list} -- list of items
        pivot {primitive type} -- item to compare to

    Returns:
        [primitive type] -- closest item to pivot
    """
    return min(items, key=lambda x: abs(x - pivot))


def get_icao_from_iata(iata_code):
    """Get ICAO code from an IATA code.

    Arguments:
        iata_code {string} -- IATA code

    Returns:
        string -- ICAO code
    """
    airport_codes = pd.read_csv(
        os.getcwd() + "/data/airport_codes.csv", sep=';', encoding='latin_1')
    airport_codes.dropna(inplace=True)
    if iata_code in airport_codes['airportIATACode'].values:
        return airport_codes.loc[airport_codes['airportIATACode'] == iata_code]['airportICAOCode'].values[0]
    else:
        return ""
