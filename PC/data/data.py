import os
import shutil
from tqdm import tqdm
import scipy.io.wavfile as wav

def move_file(path):
    files = os.listdir(path)

    for file in tqdm(files):
        folder = os.path.join(path, file.split('_')[0])
        if not os.path.exists(folder):
            os.makedirs(folder)
        shutil.move(path + '/' + file, folder)

def wav_info(audio_name):
    samplimg_freq, audio = wav.read(audio_name)

    # plt.plot(range(len(audio)), audio)
    #
    # plt.show()

    nframes = audio.shape[0]

    """计算文件的时间长度"""
    time = nframes/samplimg_freq
    """计算文件的大小"""
    size = nframes*2/1024

    # print("采样频率：{}，采样点：{}，音频时长：{}，音频文件大小：{}".format(samplimg_freq,nframes, time, size))

    return samplimg_freq,nframes, time, size

import pandas as pd

def data_info(input_path):

    filenames = []
    samplimg_freqs = []
    nframes = []
    times = []
    sizes = []

    files = os.listdir(input_path)
    for fi in tqdm(files):
        fi_d = os.path.join(input_path, fi)
        for f in os.listdir(fi_d):
            filename = f
            if ('wav' not in filename):
                continue
            f = os.path.join(fi_d, f)
            samplimg_freq, nframe, time, size = wav_info(f)

            filenames.append(filename)
            samplimg_freqs.append(samplimg_freq)
            nframes.append(nframe)
            times.append(time)
            sizes.append(size)

    return pd.DataFrame({'文件名称':filenames,'采样频率':samplimg_freqs,'采样点': nframes,'音频时长':times,'音频文件大小':sizes})


if __name__ == '__main__':
    train_path = "voice/train_voice"
    test_path = "voice/test_voice"
    move_file(train_path)
    move_file(test_path)
    
    train_pd = data_info(train_path)
    train_pd.to_csv('train_info.csv')
    
    test_pd = data_info(test_path)
    test_pd.to_csv('test_info.csv')

    import numpy as np
    
    y_train = np.load('/home/CAIL/Speaker_R/data/train_label.npy')
    y_test = np.load('/home/CAIL/Speaker_R/data/test_label.npy')
    from collections import Counter
    
    print(list(dict(Counter(y_train)).keys()))
    print(list(dict(Counter(y_test)).keys()))
    
    print(len(dict(Counter(y_train))))
    print(len(dict(Counter(y_test))))

    enroll_path = "voice/enroll"
    move_file(enroll_path)

    print(wav_info("/home/CAIL/Speaker_R/speaker_class_model/WebApp/data/voice/train_voice/sunyan/sunyan_001.wav"))

    samplimg_freq, audio = wav.read("/home/sunyan/Downloads/sunyan_001.wav")
    # wav.write("sunyan_002.wav",rate=16000,data=audio)
    print(samplimg_freq)
