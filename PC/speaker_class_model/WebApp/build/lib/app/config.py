import os

CORS_HEADERS = 'Content-Type'

RAW_TRAIN_FOLDER = os.path.join(os.getcwd(), '../train_data/voices_raw')
PROCESSED_TRAIN_FOLDER = os.path.join(os.getcwd(), '../train_data/voices_processed')

RAW_TEST_FOLDER = os.path.join(os.getcwd(), '../test_data/voices_raw')
PROCESSED_TEST_FOLDER = os.path.join(os.getcwd(), '../test_data/voices_processed')

BASE_URL = 'https://localhost:5000'