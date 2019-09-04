import matplotlib.pyplot as plt
import tensorflow as tf

import keras
from keras.models import load_model,Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint,TensorBoard
from nnom_utils import *

from mfcc import *

from wandb.keras import WandbCallback
import wandb

import tensorflow as tf
from keras import backend as K
from keras.layers import Dropout

model_path = 'model.h5'

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


    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 1), strides=(2, 1), padding="valid")(x)

    x_shortcut = x
    x = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding="valid")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Add()([x, x_shortcut])

    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 1), strides=(2, 1), padding="valid")(x)

    x = Flatten(name='flatten')(x)
    x = Dense(type)(x)
    x = Dropout(rate=0.5)(x)

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

    wandb_call = WandbCallback(data_type="image", labels=labels)
    #write_images=1, histogram_freq=1
    tb = TensorBoard(log_dir="./logs")
    callback_lists = [checkpoint,wandb_call, tb]

    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.2,
              shuffle=True, callbacks=callback_lists)

    del model
    K.clear_session()

    return history

def main():

    # fixed the gpu error
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)


    x_train = np.load('train_data.npy')
    y_train = np.load('train_label.npy')


    # label: the selected label will be recognised, while the others will be classified to "unknow".
    #selected_lable = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    #selected_lable = ['marvin', 'sheila', 'yes', 'no', 'left', 'right', 'forward', 'backward', 'stop', 'go']

    # selected_lable = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight','five', 'follow', 'forward',
    #                   'four','go','happy','house','learn','left','marvin','nine','no','off','on','one','right',
    #                   'seven','sheila','six','stop','three','tree','two','up','visual','yes','zero']

    selected_lable = ['F070', 'M018', 'M011', 'F002', 'F004', 'F089', 'F085', 'F009', 'F096', 'M056', 'M024', 'F097', 'M035', 'F069', 'M093', 'F039', 'F012', 'M053', 'F081', 'F065', 'M068', 'M078', 'M080', 'F024', 'M006', 'M052', 'M044', 'F082', 'M020', 'M088', 'F099', 'F064', 'F079', 'F037', 'M067', 'F040', 'M091', 'F051', 'M072', 'F056', 'F059', 'F062', 'F005', 'F006', 'F090', 'M062', 'M004', 'M034', 'M060', 'M049', 'F003', 'M066', 'F031', 'F087', 'F063', 'M040', 'F042', 'M065', 'M097', 'M092', 'F014', 'M094', 'M041', 'M079', 'M036', 'M061', 'M031', 'F058', 'F029', 'M089', 'F050', 'F049', 'M099', 'M051', 'M083', 'M095', 'F028', 'F025', 'M022', 'F048', 'M014', 'M007', 'M015', 'M075', 'M082', 'F074', 'F098', 'F008', 'M084', 'M055', 'M046', 'M059', 'F016', 'F094', 'M037', 'F084', 'M009', 'M029', 'F015', 'F061', 'M086', 'F077', 'M023', 'F027', 'F095', 'M073', 'F010', 'M019', 'F052', 'F088', 'F034', 'F080', 'F041', 'F047', 'M048', 'M003', 'M054', 'F078', 'M070', 'F073', 'F021', 'F019', 'M043', 'M001', 'F033', 'F035', 'M012', 'M096', 'F055', 'M013', 'M076', 'F083', 'M042', 'M008', 'F011', 'F022', 'F007', 'M058', 'M028', 'F046', 'M017', 'M069', 'F092', 'M005', 'M085', 'M027', 'M016', 'F086', 'F068', 'M057', 'F045', 'F075', 'F054', 'F020', 'M090', 'M030', 'M050', 'M010', 'M047', 'F071', 'M098', 'M064', 'M074', 'F001', 'F018', 'F030', 'M033', 'M032', 'F100', 'M071', 'F038', 'M081', 'F053', 'M002', 'F067', 'M021', 'M025', 'M038', 'F057', 'F017', 'F072', 'M100', 'F043', 'M063', 'F060', 'F044', 'F036', 'F091', 'M026', 'F076', 'M045', 'F023', 'F026', 'F032', 'F093', 'M039', 'M087', 'M077', 'F013', 'F066']

    print(len(selected_lable))

    wandb.init(project="Deep_Vector_CNN")
    config = wandb.config

    # parameters
    config.epochs = 100
    config.batch_size = 512
    num_type = len(selected_lable)

    # only take 2~13 coefficient. 1 is destructive.
    x_train = x_train[:, :, 1:]

    # expand on channel axis because we only have one channel
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    print('x_train shape:', x_train.shape, 'max', x_train.max(), 'min', x_train.min())

    # fake quantised
    # instead of using maximum value for quantised, we allows some saturation to save more details in small values.
    quantise_factor = pow(2, 4)
    print("quantised by", quantise_factor)

    x_train = (x_train / quantise_factor)

    # training data enforcement
    # x_train = np.vstack((x_train, x_train*0.8))
    # y_train = np.hstack((y_train, y_train))
    print(y_train.shape)

    # saturation to -1 to 1
    x_train = np.clip(x_train, -1, 1)

    # -1 to 1 quantised to 256 level (8bit)
    x_train = (x_train * 128).round()/128

    print('quantised', 'x_train shape:', x_train.shape, 'max', x_train.max(), 'min', x_train.min())

    # test, if you want to see a few random MFCC imagea.
    # which = 100
    # mfcc_plot(x_train[which].reshape((255, 12))*128, y_train[which])

    # word label to number label
    y_train = label_to_category(y_train, selected_lable)

    # number label to onehot
    y_train = keras.utils.to_categorical(y_train, num_type)

    # shuffle data
    permutation = np.random.permutation(x_train.shape[0])
    x_train = x_train[permutation, :]
    y_train = y_train[permutation]

    # generate test data for MCU
    # generate_test_bin(x_test * 127, y_test, 'test_data.bin')
    # generate_test_bin(x_train * 127, y_train, 'train_data.bin')

    # do the job
    history = train(x_train, y_train, type=num_type, batch_size=config.batch_size,
                    epochs=config.epochs, labels=selected_lable)

    # reload the best model
    model = load_model(model_path, custom_objects={'f1': f1,'amsoftmax_loss':amsoftmax_loss})

    layerName = "flatten"
    targetModel = Model(inputs=model.input, outputs=model.get_layer(layerName).output)
    targetModel.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    targetModel.save("dvector.h5")

    # evaluate_model(model, x_test, y_test)

    # generate_model(model, np.vstack((x_test, x_val)), name="weights.h")

    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    #
    # plt.plot(range(0, config.epochs), acc, color='red', label='Training acc')
    # plt.plot(range(0, config.epochs), val_acc, color='green', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()
    #
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    #
    # plt.plot(range(0, config.epochs), loss, color='red', label='Training acc')
    # plt.plot(range(0, config.epochs), val_loss, color='green', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()


if __name__=='__main__':
    main()
    """
    top1_acc: 0.16010165184243966
    top3_acc: 0.31300296484540446
    top5_acc: 0.40258365099534094
    """