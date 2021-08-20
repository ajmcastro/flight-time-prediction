"""Utility function for file merging."""
import os

import pandas as pd
from tqdm import tqdm

from src import consts as const


def concat_csv(folder_path):
    """Concatenate files in folder.

    Arguments:
        folder_path {string} -- path to folder containing the files to concatenate

    Returns:
        DataFrame (pandas) -- concatenated files
    """
    frames = []
    for file_name in tqdm(os.listdir(folder_path)):
        file_path = folder_path / file_name
        if file_name.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        frames.append(df)
    return pd.concat(frames)


def generate_flight_info():
    """Generate flight on-time performance DataFrame.

    Returns:
        DataFrame (pandas) -- concatenated files
    """
    flights1 = const.OTP_DIR / 'export_dados_5_anos_SET_DEZaJAN_com_dly_codes'
    flights2 = const.OTP_DIR / 'export_dados_5_anos_restantes_meses_com_dly_codes'
    df_flights1 = concat_csv(flights1)
    df_flights2 = concat_csv(flights2)
    return pd.concat([df_flights1, df_flights2])


def main():
    """
    Main for testing purposes.
    """
    full_info = concat_csv(const.WEATHER_DIR)
    full_info.to_csv(const.PROCESSED_DATA_DIR / 'full_info.csv',
                     sep=',', encoding='utf-8')


if __name__ == '__main__':
    main()
