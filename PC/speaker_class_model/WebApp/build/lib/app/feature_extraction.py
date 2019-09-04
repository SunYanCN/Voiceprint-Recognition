from flask import current_app as app

import os
import numpy as numpy
import scipy.io.wavfile as wav
from python_speech_features import mfcc, delta, logfbank


no_of_features = 13
no_of_fbank_features = 13
no_of_columns = (3 * no_of_features) + no_of_fbank_features


def get_feature_vectors(file, directory, no_of_frames, start_frame):
    (rate,sig) = wav.read(os.path.join(directory, file))
    fbank_feat = logfbank(sig,rate,nfft=2048)
    mfcc_feat = mfcc(sig,rate,winlen=0.032,winstep=0.016,numcep=13,nfft=2048)

    d_mfcc_feat = delta(mfcc_feat, 2)
    dd_mfcc_feat = delta(d_mfcc_feat, 2)
    
    mfcc_vectors = mfcc_feat[start_frame:start_frame+no_of_frames,:no_of_features]
    dmfcc_vectors = d_mfcc_feat[start_frame:start_frame+no_of_frames,:no_of_features]
    ddmfcc_vectors = dd_mfcc_feat[start_frame:start_frame+no_of_frames,:no_of_features]
    fbank_vectors = fbank_feat[start_frame:start_frame+no_of_frames,:no_of_fbank_features]
    
    feature_vectors = numpy.hstack((mfcc_vectors, dmfcc_vectors, ddmfcc_vectors, fbank_vectors))
    return feature_vectors


def get_feature_vectors_with_index(file, directory, no_of_frames, start_frame):

    feature_vectors = get_feature_vectors(file, directory, no_of_frames, start_frame)
    # print(feature_vectors)
    
    # get speaker index from filename
    speaker_index = int(file.split("_")[0])
    # if speaker_index[0] == 'M':
    #    speaker_index = 5 + int(speaker_index[3:])
    # else:
    #    speaker_index = int(speaker_index[3:])

    #append speaker index to feature vectors
    np_speaker_index = numpy.array([speaker_index])
    # print(np_speaker_index)
    temp = numpy.tile(np_speaker_index[numpy.newaxis,:], (feature_vectors.shape[0],1))
    # print(temp)
    concatenated_feature_vector = numpy.concatenate((feature_vectors,temp), axis=1)
    # print(concatenated_feature_vector)
    return concatenated_feature_vector