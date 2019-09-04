from flask import current_app as app

import os
# import requests

import numpy as numpy
from keras.models import Sequential
from keras.layers import Convolution1D, MaxPooling1D, Dense, Dropout, Activation, Flatten, Reshape
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import Callback

from app.feature_extraction import get_feature_vectors_with_index, get_feature_vectors, no_of_columns
from app.utils import find_majority, get_test_speaker_name


cnn_model = None
classes = None
mean = None
std_deviation = None
speaker_name = "None"

def trainCNN():
    global cnn_model, classes, mean, std_deviation
    directory = app.config['PROCESSED_TRAIN_FOLDER']
    no_of_frames = 800
    start_frame = 10
    classes = len(os.listdir(directory)) + 2
    print(classes)
    dataset = numpy.empty([0, no_of_columns + 1])
    
    for file in os.listdir(directory):
        dataset = numpy.concatenate((dataset, get_feature_vectors_with_index(file, directory, no_of_frames, start_frame)), axis=0)

    my_data = dataset
    numpy.random.shuffle(my_data)

    print(my_data.shape)
    Y = numpy.copy(my_data[:, no_of_columns:])
    print(Y.shape)
    
    X = numpy.copy(my_data[:, :no_of_columns])
    print(X.shape)
    
    mean = X.mean(0, keepdims=True)
    print(mean.shape)
    
    std_deviation = numpy.std(X, axis=0, keepdims=True)
    print(std_deviation.shape)
    
    normalized_X = (X - mean) / std_deviation
    print(normalized_X.shape)
    
    one_hot_labels = np_utils.to_categorical(Y, num_classes=classes+1)
    print(one_hot_labels)
    
    cnn_model = cnn_train(normalized_X, one_hot_labels, classes)
    # app.config['CNN_MODEL'] = cnn_model
    # app.config['CLASSES'] = classes
    # test_cnn(cnn_model, classes, mean, std_deviation)


def cnn_train(normalized_X, one_hot_labels, classes):
    
    temp = normalized_X.reshape(normalized_X.shape[0], no_of_columns, 1)
    
    model = Sequential()
    # 13 7 1 1 0.25 60 0.25 10 - 70%
    
    model.add(Convolution1D(52, 13, activation='tanh', input_shape=(no_of_columns,1)))
    print(model.output_shape)
    model.add(Convolution1D(52, 7, activation='tanh'))
    print(model.output_shape)
    model.add(Convolution1D(13, 3, activation='tanh'))
    print(model.output_shape)
    
    # stride = 2 - 70
    # 20, 10, 17 op - 64
    
    model.add(MaxPooling1D(pool_size=(1)))
#     model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1000, activation='tanh'))
    model.add(Dropout(0.25))
    
    # 0.4 70
    model.add(Dense(classes+1, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(temp, one_hot_labels, epochs=10, batch_size=100, verbose=1, callbacks=[TrainCallback()])
    return model


class TrainCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        print(epoch)


def testCNN():
    # model = app.config['CNN_MODEL']
    global cnn_model, classes, mean, std_deviation, speaker_name
    model = cnn_model
    directory = app.config['PROCESSED_TEST_FOLDER']
    no_of_frames = 50
    test_frames = 50
    start_frame = 1
    test_model = numpy.empty([0, no_of_columns])
    
    test_model = numpy.concatenate((test_model, get_feature_vectors('test.wav', directory, no_of_frames, start_frame)), axis=0)
    
    test_X = test_model

    normalized_test_X = (test_X - mean) / std_deviation
    
    test_X = test_X.reshape(test_X.shape[0], no_of_columns, 1)
    normalized_test_X = normalized_test_X.reshape(normalized_test_X.shape[0], no_of_columns, 1)
    
    predictions = model.predict(normalized_test_X)

    b = [sum(predictions[current: current+test_frames]) for current in range(0, len(predictions), test_frames)]
    predicted_Y = []
    for row in b:
        predicted_Y.append(row.argmax(axis=0))
    
    indices = numpy.argmax(predictions, axis=1)
    majority = []
    
    for i in range(0, len(indices), test_frames):
        majority.append(find_majority(indices[i:i + test_frames]))

    for p, m in zip(predicted_Y, majority):
        print(p, m[0])

    speaker_name = get_test_speaker_name(p)
    print(speaker_name)


def test_cnn(model, classes, mean, std_deviation):

    directory = app.config['PROCESSED_TEST_FOLDER']
    no_of_frames = 50
    test_frames = 50
    start_frame = 1
    test_model = numpy.empty([0, no_of_columns + 1])
    
    for file in os.listdir(directory):
        test_model = numpy.concatenate((test_model, get_feature_vectors_with_index(file, directory, no_of_frames, start_frame)), axis=0)
    
#     print(test_model.shape)

    test_X = numpy.copy(test_model[:, :no_of_columns])
#     print(test_X.shape)

    normalized_test_X = (test_X - mean) / std_deviation
#     print(normalized_test_X.shape)

    test_Y = numpy.copy(test_model[:, no_of_columns:])
#     print(test_Y.shape)
    test_labels = np_utils.to_categorical(test_Y, num_classes=classes+1)
    
    test_X = test_X.reshape(test_X.shape[0], no_of_columns, 1)
    normalized_test_X = normalized_test_X.reshape(normalized_test_X.shape[0], no_of_columns, 1)
    
    print(model.test_on_batch(normalized_test_X, test_labels, sample_weight=None))
    print(model.metrics_names)
    predictions = model.predict(normalized_test_X)

    b = [sum(predictions[current: current+test_frames]) for current in range(0, len(predictions), test_frames)]
    predicted_Y = []
    for row in b:
        predicted_Y.append(row.argmax(axis=0))

    # print(predicted_Y)
    # print(test_Y[::40].T)
    
    indices = numpy.argmax(predictions, axis=1)
    majority = []
    
    for i in range(0, len(indices), test_frames):
        majority.append(find_majority(indices[i:i + test_frames]))
        
#     majority = 
    for t, p, m in zip(test_Y[::test_frames].T[0], predicted_Y, majority):
        print(int(t), p, m[0])
    
#     for t, p in zip(test_Y.T[0], indices):
#         print(int(t), p)  
    
    diff = predicted_Y - test_Y[::test_frames].T[0]
    maj_diff = numpy.array(majority)[:, 0] - test_Y[::test_frames].T[0]

    numerator = sum(x == 0 for x in diff)
    denominator = len(predicted_Y)
    
    numerator2 = sum(x == 0 for x in maj_diff)
    denominator2 = len(maj_diff)  

    print("Accuracy prob_diff: {} of {} - {}".format(numerator, denominator, numerator/denominator))
    
    print("Accuracy maj_diff: {} of {} - {}".format(numerator2, denominator2, numerator2/denominator2))


def get_speaker_name():
    global speaker_name
    return speaker_name

def reset_speaker_name():
    global speaker_name
    speaker_name = "None"