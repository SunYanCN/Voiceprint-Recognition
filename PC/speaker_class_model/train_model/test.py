
from keras.models import load_model

from mfcc import *

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

def euclidean_similarity(vector_a, vector_b):
    return np.sqrt(np.sum((vector_a-vector_b)**2))

def cosine_similarity(vector_a, vector_b):
    """ 计算两个向量x和y的余弦相似度 """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def eu_knn(v, M, y,topk):
    distances = [euclidean_similarity(m,v) for m in M]
    nearest = np.argsort(distances)
    topK_dis = [distances[i] for i in nearest[:topk]]
    topK_y = [y[i] for i in nearest[:topk]]
    return topK_y,topK_dis

def cos_knn(v, M, y,topk):
    distances = [cosine_similarity(m,v) for m in M]
    nearest = np.argsort(distances)[::-1]
    topK_dis = [distances[i] for i in nearest[:topk]]
    topK_y = [y[i] for i in nearest[:topk]]
    return topK_y,topK_dis

import keras.backend as K

def triplet_loss(y_true, y_pred):
    """
    Triplet Loss的损失函数
    """

    anc, pos, neg = y_pred[:, 0:128], y_pred[:, 128:256], y_pred[:, 256:]

    # 欧式距离
    pos_dist = K.sum(K.square(anc - pos), axis=-1, keepdims=True)
    neg_dist = K.sum(K.square(anc - neg), axis=-1, keepdims=True)
    basic_loss = pos_dist - neg_dist + 1.0

    loss = K.maximum(basic_loss, 0.0)
    print("[INFO] model - triplet_loss shape: %s" % str(loss.shape))
    return loss


if __name__ == '__main__':
    import numpy as np
    # enroll_path = "/home/CAIL/Speaker_R/data/voice/enroll"
    # x_enroll, y_enroll = merge_mfcc_file(input_path=enroll_path, sig_len=16000)
    # np.save('/home/CAIL/Speaker_R/data/enroll_data.npy', x_enroll)
    # np.save('/home/CAIL/Speaker_R/data/enroll_label.npy', y_enroll)

    x_test  = np.load('test_data.npy')
    y_test = np.load('test_label.npy')

    x_enroll = np.load('enroll_data.npy')
    y_enroll = np.load('enroll_label.npy')

    x_enroll = proress(x_enroll)
    x_test = proress(x_test)

    print(x_enroll.shape, x_test.shape)

    from kws import f1,amsoftmax_loss

    model_path = "dvector.h5"
    # model = load_model(model_path, custom_objects={'f1': f1,"AMSoftmax": AMSoftmax,"amsoftmax_loss": amsoftmax_loss})
    model = load_model(model_path, custom_objects={'f1': f1,'amsoftmax_loss':amsoftmax_loss})

    # model_path = "/home/CAIL/Speaker_R/triplet-loss/experiments/triplet_loss/checkpoints/triplet_loss_model.h5"
    # model = load_model(model_path, custom_objects={'triplet_loss': triplet_loss})
    # model.summary()
    #
    #
    # def predict(x):
    #     anchor = np.reshape(x, (-1, 63, 12, 1))
    #     X = {
    #         'anc_input': anchor,
    #         'pos_input': np.zeros(anchor.shape),
    #         'neg_input': np.zeros(anchor.shape)
    #     }
    #     y = model.predict(X, batch_size=1)[:, :128]
    #
    #     return y

    y_vector = model.predict(x_enroll)
    y_pred = model.predict(x_test)

    print(y_vector.shape, y_pred.shape)

    top1_acc_num = 0
    top3_acc_num = 0
    top5_acc_num = 0

    from tqdm import tqdm

    for i in tqdm(range(y_pred.shape[0])):
        topK,topK_dis = eu_knn(y_pred[i],y_vector,y_enroll,topk=5)
        # print("True:",y_test[i],"Predict:",topK,"Distance:",topK_dis)
        if y_test[i] in topK[:1]:
            top1_acc_num +=1
        if y_test[i] in topK[:3]:
            top3_acc_num +=1
        if y_test[i] in topK[:5]:
            top5_acc_num +=1

    print("top1_acc:", top1_acc_num / y_pred.shape[0])
    print("top3_acc:", top3_acc_num / y_pred.shape[0])
    print("top5_acc:", top5_acc_num / y_pred.shape[0])

    # topK_y,topK_dis = knn(np.array([0,0.2]), np.array([[0,0.1],[1,1]]), ['A','B'], 1)
    # print(topK_y, topK_dis)



