from keras.layers import Dense
from keras.layers import Dropout
import keras
from keras.layers import SeparableConv1D, Input, BatchNormalization, Flatten
from keras.models import Model
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.utils import class_weight
import numpy as np
import collections
from model_architectures import *
import os
from keras.models import load_model

def save_pickle(object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(object, f)


def poly_decay(epoch):
    maxEpochs = 60
    baseLR = 5e-3
    power = 1.0
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    return alpha


def one_hot_encode(Y):
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    dummy_y = np_utils.to_categorical(encoded_Y)
    return dummy_y


def load_pickle(filename):
    with open(filename, 'rb') as input:
        object = pickle.load(input)
    return object


def get_class_weights(y, smooth_factor=0):
    counter = collections.Counter(y)
    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p
    majority = max(counter.values())
    return {cls: float(majority / count) for cls, count in counter.items()}


def make_two_way(labels):
    for label_num in range(len(labels)):
        if labels[label_num] >= 4:
            labels[label_num] = 0
        else:
            labels[label_num] = 1
    return labels


for best_weights_filepath, CompleteModel in [('Weights/BestWeights_nochannel_twoway_dil.h5', nochannel_dil(2)),
                                             ('Weights/BestWeights_nochannel_sixway.h5', nochannel_dil(6)),
                                             ('Weights/BestWeights_nochannel_twoway_nodil.h5', nochannel_nodil(2)),
                                             ('Weights/BestWeights_nochannel_sixway_nodil.h5', nochannel_nodil(6)),
                                             ('Weights/BestWeights_channel_twoway.h5',channel_dil(2)),
                                             ('Weights/BestWeights_channel_sixway.h5', channel_dil(6)),
                                             ('Weights/BestWeights_channel_twoway_nodil.h5', channel_nodil(2)),
                                             ('Weights/BestWeights_channel_sixway_nodil.h5', channel_nodil(6))]:
    if os.path.exists(best_weights_filepath):
        continue
    print(best_weights_filepath)
    TrainFeatures = load_pickle('ExtractedData/TrainFeatures')
    TrainLabels = load_pickle('ExtractedData/TrainLabels')
    TrainOffendingChannel = load_pickle('ExtractedData/TrainOffendingChannel')

    EvalFeatures = load_pickle('ExtractedData/EvalFeatures')
    EvalLabels = load_pickle('ExtractedData/EvalLabels')
    EvalOffendingChannel = load_pickle('ExtractedData/EvalOffendingChannel')

    if 'twoway' in best_weights_filepath:
        TrainLabels = make_two_way(TrainLabels)
        EvalLabels = make_two_way(EvalLabels)

    TrainLabels_onehot = one_hot_encode(TrainLabels)
    TrainOffendingChannel_onehot = one_hot_encode(TrainOffendingChannel)

    EvalLabels_onehot = one_hot_encode(EvalLabels)
    EvalOffendingChannel_onehot = one_hot_encode(EvalOffendingChannel)

    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=20,
                                                  verbose=1,
                                                  mode='auto')

    saveBestModel = keras.callbacks.ModelCheckpoint(best_weights_filepath,
                                                    monitor='val_loss',
                                                    verbose=1,
                                                    save_best_only=True,
                                                    mode='auto')

    TrainLabels_ints = [y.argmax() for y in TrainLabels_onehot]
    EvalOffendingChannel_ints = [y.argmax() for y in EvalOffendingChannel_onehot]
    if 'nochannel' in best_weights_filepath:
        class_weight_dict = {'event_prediction': get_class_weights(y=TrainLabels_ints)} #evenly weigh events
        CompleteModel.fit(TrainFeatures,
                          [TrainLabels_onehot],
                          batch_size=128,
                          epochs=30,
                          validation_data=([EvalFeatures], [EvalLabels_onehot]),
                          callbacks=[saveBestModel, earlyStopping, LearningRateScheduler(poly_decay)],
                          class_weight=class_weight_dict)

        PredictedLabels = CompleteModel.predict(EvalFeatures)

    else:
        class_weight_dict = {'event_prediction': get_class_weights(y=TrainLabels_ints),
                             'channel_prediction': get_class_weights(y=EvalOffendingChannel_ints)} #evenly weigh events AND channels
        CompleteModel.fit(TrainFeatures,
                          [TrainLabels_onehot, TrainOffendingChannel_onehot],
                          batch_size=128,
                          epochs=30,
                          validation_data=([EvalFeatures], [EvalLabels_onehot, EvalOffendingChannel_onehot]),
                          callbacks=[saveBestModel, earlyStopping, LearningRateScheduler(poly_decay)],
                          class_weight=class_weight_dict)

        PredictedLabels = CompleteModel.predict([EvalFeatures])


    save_pickle(PredictedLabels, "Labels/PredictedLabels_" + best_weights_filepath.split('/')[1].split('.')[0])
    save_pickle(EvalLabels, "Labels/TrueLabels_" + best_weights_filepath.split('/')[1].split('.')[0])
