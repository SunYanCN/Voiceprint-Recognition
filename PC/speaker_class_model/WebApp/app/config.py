import os

CORS_HEADERS = 'Content-Type'

RAW_TRAIN_FOLDER = os.path.join(os.getcwd(), '../data/voice/train_voice')
PROCESSED_TRAIN_FOLDER = os.path.join(os.getcwd(), '../data')

RAW_TEST_FOLDER = os.path.join(os.getcwd(), '../data/voice/test')
PROCESSED_TEST_FOLDER = os.path.join(os.getcwd(), '../data')

RAW_ENROLL_FOLDER = os.path.join(os.getcwd(), '../data/voice/enroll_voice')
PROCESSED_ENROLL_FOLDER = os.path.join(os.getcwd(), '../data')

BASE_URL = 'https://localhost:5000'