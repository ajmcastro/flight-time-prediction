"""Project constants"""

from pathlib import Path


### DIRECTORIES ###
ROOT_DIR = Path(__file__).resolve().parent.parent  # MScThesis/src
# DATA DIRECTORIES
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'
OTP_DIR = RAW_DATA_DIR / 'on-time_performance'
WEATHER_DIR = RAW_DATA_DIR / 'weather'
# MODEL DIRECTORIES
MODEL_DIR = ROOT_DIR / 'model'
# SOURCE DIRECTORIES
MODELING_DIR = ROOT_DIR / 'src' / 'modeling'
# IMAGES
IMAGES_DIR = ROOT_DIR / 'images'
