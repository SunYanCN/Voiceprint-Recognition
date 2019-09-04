import matplotlib.pyplot as plt
import keras
from keras.models import load_model,Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint,TensorBoard
from nnom_utils import *
from mfcc import *
import tensorflow as tf
from keras import backend as K

from flask import current_app as app
from utils import get_test_speaker_name
import os

model_path = 'model.h5'
speaker_name = "None"

def mfcc_plot(x, label= None):
    mfcc_feat = np.swapaxes(x, 0, 1)
    ig, ax = plt.subplots()
    cax = ax.imshow(mfcc_feat, interpolation='nearest', origin='lower', aspect=1)#, cmap=cm.coolwarm)
    if label is not None:
        ax.set_title(label)
    else:
        ax.set_title('MFCC')
    plt.show()

def label_to_category(label, selected):
    category = []
    for word in label:
        category.append(selected.index(word))
    return np.array(category)

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def amsoftmax_loss(y_true, y_pred, scale=10, margin=10.0):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)

def train(x_train, y_train, type, batch_size=64, epochs=100,labels=None):
    inputs = Input(shape=x_train.shape[1:])
    x = Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='valid')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 1), strides=(2, 1), padding="valid")(x)

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 1),strides=(2, 1), padding="valid")(x)


    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 1), strides=(2, 1), padding="valid")(x)

    x = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding="valid")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(rate=0.2)(x)

    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 1), strides=(2, 1), padding="valid")(x)
    x = Dropout(rate=0.5)(x)

    x = Flatten(name='flatten')(x)
    x = Dense(type)(x)
    # predictions = Softmax()(x)

    model = Model(inputs=inputs, outputs=x)
    # loss = 'categorical_crossentropy',
    from keras.optimizers import SGD
    model.compile(loss=amsoftmax_loss,
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    # save best
    checkpoint = ModelCheckpoint(filepath=model_path,
            monitor='val_acc',
            verbose=0,
            save_best_only='True',
            mode='auto',
            period=1)

    #write_images=1, histogram_freq=1
    tb = TensorBoard(log_dir="./logs")
    callback_lists = [checkpoint, tb]

    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.2,
              shuffle=True, callbacks=callback_lists)

    del model
    K.clear_session()

    return history

def proress(x_train):
    x_train = x_train[:, :, 1:]
    # expand on channel axis because we only have one channel
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    print('x_train shape:', x_train.shape, 'max', x_train.max(), 'min', x_train.min())

    # fake quantised
    # instead of using maximum value for quantised, we allows some saturation to save more details in small values.
    quantise_factor = pow(2, 4)
    print("quantised by", quantise_factor)
    x_train = (x_train / quantise_factor)
    # saturation to -1 to 1
    x_train = np.clip(x_train, -1, 1)
    # -1 to 1 quantised to 256 level (8bit)
    x_train = (x_train * 128).round() / 128
    print('quantised', 'x_train shape:', x_train.shape, 'max', x_train.max(), 'min', x_train.min())
    return x_train

def train_cnn(selected_lable):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)
    dest = app.config['PROCESSED_TRAIN_FOLDER']
    train_data_path = os.path.join(dest,'train_data.npy')
    train_label_path = os.path.join(dest,'train_label.npy')

    x_train = np.load(train_data_path, allow_pickle=True)
    y_train = np.load(train_label_path ,allow_pickle=True)

    epochs = 80
    batch_size = 64
    num_type = len(selected_lable)
    x_train = proress(x_train)
    # word label to number label
    y_train = label_to_category(y_train, selected_lable)

    # number label to onehot
    y_train = keras.utils.to_categorical(y_train, num_type)

    # shuffle data
    permutation = np.random.permutation(x_train.shape[0])
    x_train = x_train[permutation, :]
    y_train = y_train[permutation]

    history = train(x_train, y_train, type=num_type, batch_size=batch_size,
                    epochs=epochs, labels=selected_lable)

    # reload the best model
    model = load_model(model_path, custom_objects={'f1': f1, 'amsoftmax_loss': amsoftmax_loss})

    layerName = "flatten"
    targetModel = Model(inputs=model.input, outputs=model.get_layer(layerName).output)
    targetModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    targetModel.save("dvector.h5")

    del model
    del targetModel
    K.clear_session()


def test_cnn():
    global speaker_name
    dest = app.config['PROCESSED_TEST_FOLDER']
    test_data_path = os.path.join(dest, 'test_data.npy')
    x_test = np.load(test_data_path)
    x_test = proress(x_test)
    model = load_model(model_path, custom_objects={'f1': f1, 'amsoftmax_loss': amsoftmax_loss})
    model._make_predict_function()
    y_vector = model.predict(x_test)
    index = np.argmax(y_vector)
    speaker_name = get_test_speaker_name(index)

    del model
    K.clear_session()

def get_speaker_name():
    global speaker_name
    return speaker_name

def reset_speaker_name():
    global speaker_name
    speaker_name = "None"

