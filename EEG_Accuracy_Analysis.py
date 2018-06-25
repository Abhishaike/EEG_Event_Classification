from sklearn.metrics import confusion_matrix
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn import metrics


def load_pickle(filename):
    with open(filename, 'rb') as input:
        object = pickle.load(input)
    return object


def one_hot_encode(Y):
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    dummy_y = np_utils.to_categorical(encoded_Y)
    return dummy_y

def six_way_confusion(PredictedLabels, TrueLabels):
    ct = pd.crosstab(TrueLabels_argmax, PredictedLabels_argmax, rownames=['True'], colnames=['Predicted'], normalize='index')
    ct = ct.reindex_axis([4, 5, 3, 1, 2, 0]).T.reindex_axis([4, 5, 3, 1, 2, 0]).T


    return ct

def two_way_conversion(PredicatedLabels, TrueLabels):
    for label_num in range(len(PredicatedLabels)):
        if PredicatedLabels[label_num] >= 4:
            PredicatedLabels[label_num] = 0
        else:
            PredictedLabels_argmax[label_num] = 1
        if PredicatedLabels[label_num] >= 4:
            TrueLabels[label_num] = 0
        else:
            TrueLabels[label_num] = 1

    return PredicatedLabels, TrueLabels

def two_way_confusion(PredicatedLabels, TrueLabels):
    cm = pd.crosstab(TrueLabels, PredicatedLabels, rownames=['True'], colnames=['Predicted'], normalize='index')
    cm = cm.reindex_axis([1,0]).T.reindex_axis([1,0]).T
    return cm

for best_weights_filepath in ['Weights/BestWeights_nochannel_twoway_dil.h5',
                              'Weights/BestWeights_nochannel_sixway.h5',
                              'Weights/BestWeights_nochannel_twoway_nodil.h5',
                              'Weights/BestWeights_nochannel_sixway_nodil.h5',
                              'Weights/BestWeights_channel_twoway.h5',
                              'Weights/BestWeights_channel_sixway.h5',
                              'Weights/BestWeights_channel_twoway_nodil.h5',
                              'Weights/BestWeights_channel_sixway_nodil.h5']:

    if 'nochannel' in best_weights_filepath:
        PredictedLabels = load_pickle("Labels/PredictedLabels_" + best_weights_filepath.split('/')[1].split('.')[0])  # you need the event prediction, not the channel
        TrueLabels_onehot = one_hot_encode( load_pickle("Labels/TrueLabels_" + best_weights_filepath.split('/')[1].split('.')[0]))
        PredictedLabels_argmax = PredictedLabels.argmax(1)
        TrueLabels_argmax = TrueLabels_onehot.argmax(1)
    else:
        PredictedLabels = load_pickle("Labels/PredictedLabels_" + best_weights_filepath.split('/')[1].split('.')[0])[0]  # you need the event prediction, not the channel
        TrueLabels_onehot = one_hot_encode( load_pickle("Labels/TrueLabels_" + best_weights_filepath.split('/')[1].split('.')[0]))
        PredictedLabels_argmax = PredictedLabels.argmax(1)
        TrueLabels_argmax = TrueLabels_onehot.argmax(1)

    print(best_weights_filepath)
    if 'sixway' in best_weights_filepath:
        print(six_way_confusion(PredictedLabels_argmax, TrueLabels_argmax))
    else:
        print(two_way_confusion(PredictedLabels_argmax, TrueLabels_argmax))
        fpr, tpr, thresholds = metrics.roc_curve(TrueLabels_argmax, PredictedLabels_argmax, pos_label=1)
        print(fpr[1])



