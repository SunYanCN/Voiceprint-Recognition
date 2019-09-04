from flask import current_app as app

import os
import numpy as np
from collections import Counter
from kws import merge_mfcc_file

def preprocess_all():
    preprocess_train()
    preprocess_enroll()
    preprocess_test()


def preprocess_train():
    train_path = app.config['RAW_TRAIN_FOLDER']
    dest = app.config['PROCESSED_TRAIN_FOLDER']
    x_train, y_train = merge_mfcc_file(input_path=train_path, sig_len=32000, enroll=False, mix_noise=False, augmentation= False, augmentation_num= 1)
    np.save(os.path.join(dest,'train_data.npy'), x_train)
    np.save(os.path.join(dest,'train_label.npy'), y_train)

def preprocess_enroll():
    enroll_path = app.config['RAW_ENROLL_FOLDER']
    dest = app.config['PROCESSED_ENROLL_FOLDER']
    x_enroll, y_enroll = merge_mfcc_file(input_path=enroll_path, sig_len=32000, enroll=True, mix_noise=False, augmentation= False)
    np.save(os.path.join(dest,'enroll_data.npy'), x_enroll)
    np.save(os.path.join(dest,'enroll_label.npy'), y_enroll)

def preprocess_test():
    test_path = app.config['RAW_TEST_FOLDER']
    dest = app.config['PROCESSED_TEST_FOLDER']
    x_test, y_test = merge_mfcc_file(input_path=test_path, sig_len=32000, enroll=True, mix_noise=False,
                                       augmentation=False)
    np.save(os.path.join(dest, "test_data.npy"), x_test)
    np.save(os.path.join(dest, "test_label.npy"), y_test)

def get_all_speakers():
    preprocess_train()
    directory = app.config['PROCESSED_TRAIN_FOLDER']
    y_train = np.load(os.path.join(directory, 'train_label.npy'))
    speakers = list(dict(Counter(y_train)).keys())
    return speakers

def get_test_speaker_name(index):
    directory = app.config['PROCESSED_TRAIN_FOLDER']
    y_train = np.load(os.path.join(directory, 'train_label.npy'))
    speakers = list(dict(Counter(y_train)).keys())
    return speakers[index]

def remove_test_file():
    file_path = os.path.join(app.config['PROCESSED_TEST_FOLDER'], 'test.wav')
    cmd = "rm -rf {}".format(file_path)
    os.system(cmd)