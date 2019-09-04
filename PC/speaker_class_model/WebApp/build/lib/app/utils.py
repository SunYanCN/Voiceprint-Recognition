from flask import current_app as app

import os
import scipy.io.wavfile as wav


def preprocess_all():
    dest = app.config['PROCESSED_TRAIN_FOLDER']
    cmd = "rm {}".format(os.path.join(dest, '*.wav'))
    os.system(cmd);
    source = app.config['RAW_TRAIN_FOLDER']
    for index, file in enumerate(os.listdir(source)):
        preprocess_train(file, index)


def preprocess_train(file, index):
    source = app.config['RAW_TRAIN_FOLDER']
    dest = app.config['PROCESSED_TRAIN_FOLDER']
    source = os.path.join(source, file)
    dest = os.path.join(dest, '{:03d}_{}'.format(index, file))
    preprocess_audio(source, dest)


def preprocess_test(file):
    source = app.config['RAW_TEST_FOLDER']
    dest = app.config['PROCESSED_TEST_FOLDER']
    source = os.path.join(source, file)
    dest = os.path.join(dest, file)
    preprocess_audio(source, dest)


def preprocess_audio(source, dest):
    print(source)
    (rate,sig) = wav.read(source)
    if int(rate) > 8000:
        cmd = "sox {} {} silence 1 0.3 -45d -1 0.1 1% lowpass 7000".format(source, dest)
        os.system(cmd);
    else:
        cmd = "sox {} {} silence 1 0.3 -55d -1 0.1 1% lowpass 3500".format(source, dest)
        os.system(cmd);
    # cmd = "sox {} {} silence 1 0.3 -45d -1 0.1 1% lowpass 5500".format(source, dest)
    # os.system(cmd);


def remove_test_file():
    file_path = os.path.join(app.config['PROCESSED_TEST_FOLDER'], 'test.wav')
    cmd = "rm {}".format(file_path)
    os.system(cmd);


def get_all_speakers():
    speakers = []
    directory = os.listdir(app.config['PROCESSED_TRAIN_FOLDER'])
    for file in directory:
    	speakers.append(file)
    return speakers


def find_majority(k):
    myMap = {}
    maximum = ( '', 0 ) # (occurring element, occurrences)
    for n in k:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1

        # Keep track of maximum on the go
        if myMap[n] > maximum[1]: maximum = (n,myMap[n])

    return maximum

def get_test_speaker_name(index):
    source = app.config['PROCESSED_TRAIN_FOLDER']
    speaker_dict = {}
    for file in os.listdir(source):
        split_name = file.split("_")
        print(split_name)
        speaker_dict[int(split_name[0])] = split_name[1].split(".")[0]

    return speaker_dict[index]
