from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import random
from tqdm import tqdm

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

augmenter = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
])

def load_noise(path='/home/CAIL/Speaker_R/data/voice/background_noise/'):
    noise = []
    files = os.listdir(path)
    for f in files:
        filename = f
        if ('wav' not in filename):
            continue
        f = os.path.join(path, f)
        (rate, sig) = wav.read(f)
        noise.append(sig)
    return  noise

def generate_mfcc(sig, rate, sig_len, noise=None, noise_weight=0.1, winlen=0.03125, winstep=0.03125/2, numcep=13, nfilt=26, nfft=512, lowfreq=20, highfreq=4000, winfunc=np.hanning, ceplifter=0, preemph=0.97):
    if(len(sig) != sig_len):
        if(len(sig)< sig_len):
            sig = np.pad(sig, (0, sig_len - len(sig)), 'constant')
        if(len(sig) >sig_len):
            sig = sig[0:sig_len]
    # i dont know, 'tensorflow' normalization
    sig = sig.astype('float') / 32768

    if(noise is not None):
        noise = noise[random.randint(0, len(noise)-1)] # pick a noise
        start = random.randint(0, len(noise)-sig_len) # pick a sequence
        noise = noise[start:start+sig_len]
        noise = noise.astype('float')/32768
        sig = sig * (1-noise_weight) + noise * noise_weight
        #wav.write('noise_test.wav', rate, sig)
    mfcc_feat = mfcc(sig, rate, winlen=winlen, winstep=winstep, numcep=numcep, nfilt=nfilt, nfft=nfft, lowfreq=lowfreq,
                     highfreq=highfreq, winfunc=winfunc, ceplifter=ceplifter, preemph=preemph)
    mfcc_feat = mfcc_feat.astype('float32')
    return mfcc_feat

def merge_mfcc_file(input_path='dat/', mix_noise=True, sig_len=16000, winlen=0.03125, winstep=0.03125/2, numcep=13, nfilt=26, nfft=512,
                    lowfreq=20, highfreq=4000, winfunc=np.hanning, ceplifter=0, preemph=0.97, enroll = False, augmentation =True, augmentation_num = 20):

    train_data = []
    train_lablel = []

    if mix_noise:
        noise = load_noise()
    else:
        noise = None

    files = os.listdir(input_path)
    for fi in tqdm(files):
        fi_d = os.path.join(input_path, fi)
        # # folders of each cmd
        if os.path.isdir(fi_d):
            label = fi_d.split('/')[-1] # get the label from the dir
            # dataset
            for f in os.listdir(fi_d):
                f = os.path.join(fi_d, f)
                (rate, sig) = wav.read(f)

                if augmentation:
                    sig = sig.astype(np.float32)
                    augmentation_data = [augmenter(samples=sig, sample_rate=rate) for _ in range(augmentation_num)] + [sig]
                else:
                    augmentation_data = [sig]

                for sig in augmentation_data:
                    if enroll:
                        data = generate_mfcc(sig, rate, sig_len, noise=noise, winlen=winlen,
                                             winstep=winstep, numcep=numcep, nfilt=nfilt, nfft=nfft, lowfreq=lowfreq,
                                             highfreq=highfreq, winfunc=winfunc, ceplifter=ceplifter, preemph=preemph)
                        data = np.array(data)
                        train_data.append(data)
                        train_lablel.append(label)
                    else:
                        for i in range(len(sig)//sig_len):
                            data = generate_mfcc(sig[i * sig_len:(i + 1) * sig_len], rate, sig_len, noise=noise,
                                                 winlen=winlen, winstep=winstep, numcep=numcep, nfilt=nfilt,
                                                 nfft=nfft, lowfreq=lowfreq,
                                                 highfreq=highfreq, winfunc=winfunc, ceplifter=ceplifter,
                                                 preemph=preemph)
                            data = np.array(data)
                            train_data.append(data)
                            train_lablel.append(label)
    # finalize
    train_data = np.array(train_data)

    return train_data, train_lablel

if __name__ == "__main__":

    train_path = "/home/CAIL/Speaker_R/data/voice/train_voice"
    test_path = "/home/CAIL/Speaker_R/data/voice/test_voice"
    enroll_path = "/home/CAIL/Speaker_R/data/voice/enroll_voice"

    x_train, y_train = merge_mfcc_file(input_path=train_path, sig_len=16000, enroll=False, mix_noise=False, augmentation= True, augmentation_num= 3)
    x_test, y_test = merge_mfcc_file(input_path=test_path, sig_len=16000, enroll=False, mix_noise=False, augmentation= False)
    x_enroll, y_enroll = merge_mfcc_file(input_path=enroll_path, sig_len=16000, enroll=True, mix_noise=False, augmentation= False)

    np.save('train_data.npy', x_train)
    np.save('train_label.npy', y_train)
    np.save('test_data.npy', x_test)
    np.save('test_label.npy', y_test)
    np.save('enroll_data.npy', x_enroll)
    np.save('enroll_label.npy', y_enroll)

    print('x_train shape:', x_train.shape, 'max', x_train.max(), 'min', x_train.min())
    print('x_test shape:', x_test.shape, 'max', x_test.max(), 'min', x_test.min())
    print(x_enroll.shape, np.shape(y_enroll))

    import pandas as pd
    train_pd = pd.DataFrame({'train_speaker':y_train})
    print(train_pd.apply(pd.value_counts))

    test_pd = pd.DataFrame({'train_speaker': y_test})
    print(test_pd.apply(pd.value_counts))

    enroll_pd = pd.DataFrame({'train_speaker': y_enroll})
    print(enroll_pd.apply(pd.value_counts))

    #
    # mfcc_feat = x_train[1000]
    # mfcc_feat = np.swapaxes(mfcc_feat, 0, 1)
    # ig, ax = plt.subplots()
    # cax = ax.imshow(mfcc_feat, interpolation='nearest', origin='lower', aspect='auto')
    # ax.set_title('MFCC')
    # plt.show()


