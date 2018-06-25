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

def nochannel_dil(classes):
    InputSignal = Input(shape=(250, 22))

    ConvBlock1 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=1, kernel_regularizer='l2')(InputSignal)
    ConvBlock1 = BatchNormalization()(ConvBlock1)
    ConvBlock1 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=1, kernel_regularizer='l2')(ConvBlock1)
    ConvBlock1 = BatchNormalization()(ConvBlock1)
    for _ in range(0, 4):
        ConvBlock1 = SeparableConv1D(filters=32, kernel_size=9, strides=2, padding='same', activation='relu',
                                     dilation_rate=1, kernel_regularizer='l2')(ConvBlock1)
        ConvBlock1 = BatchNormalization()(ConvBlock1)

    ConvBlock2 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=3, kernel_regularizer='l2')(InputSignal)
    ConvBlock2 = BatchNormalization()(ConvBlock2)
    ConvBlock2 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=6, kernel_regularizer='l2')(ConvBlock2)
    ConvBlock2 = BatchNormalization()(ConvBlock2)
    for _ in range(0, 4):
        ConvBlock2 = SeparableConv1D(filters=32, kernel_size=9, strides=2, padding='same', activation='relu',
                                     dilation_rate=1, kernel_regularizer='l2')(ConvBlock2)
        ConvBlock2 = BatchNormalization()(ConvBlock2)

    ConvBlock3 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=6, kernel_regularizer='l2')(InputSignal)
    ConvBlock3 = BatchNormalization()(ConvBlock3)
    ConvBlock3 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=9, kernel_regularizer='l2')(ConvBlock3)
    ConvBlock3 = BatchNormalization()(ConvBlock3)
    for _ in range(0, 4):
        ConvBlock3 = SeparableConv1D(filters=32, kernel_size=9, strides=2, padding='same', activation='relu',
                                     dilation_rate=1, kernel_regularizer='l2')(ConvBlock3)
        ConvBlock3 = BatchNormalization()(ConvBlock3)

    Concat_Layer = keras.layers.concatenate([ConvBlock1, ConvBlock2, ConvBlock3])
    Concat_Layer = Flatten()(Concat_Layer)
    FinalOutput = Dense(1024, activation='relu', kernel_regularizer='l2')(Concat_Layer)
    FinalOutput = BatchNormalization()(FinalOutput)
    FinalOutput = Dropout(.5)(FinalOutput)
    FinalOutput = Dense(512, activation='relu', kernel_regularizer='l2')(FinalOutput)
    FinalOutput = BatchNormalization()(FinalOutput)
    FinalOutput = Dropout(.5)(FinalOutput)
    FinalOutput = Dense(256, activation='relu', kernel_regularizer='l2')(FinalOutput)
    FinalOutput = BatchNormalization()(FinalOutput)
    FinalOutput = Dropout(.5)(FinalOutput)

    EventPrediction = Dense(64, activation='relu', kernel_regularizer='l2')(FinalOutput)
    EventPrediction = BatchNormalization()(EventPrediction)
    EventPrediction = Dropout(.5)(EventPrediction)
    EventPrediction = Dense(classes, activation='softmax', kernel_regularizer='l2', name='event_prediction')(
        EventPrediction)

    CompleteModel = Model(inputs=(InputSignal), outputs=[EventPrediction])
    opt = keras.optimizers.sgd(lr=1e-3, momentum=.9, nesterov=True)
    CompleteModel.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    return CompleteModel


def nochannel_nodil(classes):
    InputSignal = Input(shape=(250, 22))

    ConvBlock1 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=1, kernel_regularizer='l2')(InputSignal)
    ConvBlock1 = BatchNormalization()(ConvBlock1)
    ConvBlock1 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=1, kernel_regularizer='l2')(ConvBlock1)
    ConvBlock1 = BatchNormalization()(ConvBlock1)
    for _ in range(0, 4):
        ConvBlock1 = SeparableConv1D(filters=32, kernel_size=9, strides=2, padding='same', activation='relu',
                                     dilation_rate=1, kernel_regularizer='l2')(ConvBlock1)
        ConvBlock1 = BatchNormalization()(ConvBlock1)

    ConvBlock2 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=1, kernel_regularizer='l2')(InputSignal)
    ConvBlock2 = BatchNormalization()(ConvBlock2)
    ConvBlock2 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=1, kernel_regularizer='l2')(ConvBlock2)
    ConvBlock2 = BatchNormalization()(ConvBlock2)
    for _ in range(0, 4):
        ConvBlock2 = SeparableConv1D(filters=32, kernel_size=9, strides=2, padding='same', activation='relu',
                                     dilation_rate=1, kernel_regularizer='l2')(ConvBlock2)
        ConvBlock2 = BatchNormalization()(ConvBlock2)

    ConvBlock3 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=1, kernel_regularizer='l2')(InputSignal)
    ConvBlock3 = BatchNormalization()(ConvBlock3)
    ConvBlock3 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=1, kernel_regularizer='l2')(ConvBlock3)
    ConvBlock3 = BatchNormalization()(ConvBlock3)
    for _ in range(0, 4):
        ConvBlock3 = SeparableConv1D(filters=32, kernel_size=9, strides=2, padding='same', activation='relu',
                                     dilation_rate=1, kernel_regularizer='l2')(ConvBlock3)
        ConvBlock3 = BatchNormalization()(ConvBlock3)

    Concat_Layer = keras.layers.concatenate([ConvBlock1, ConvBlock2, ConvBlock3])
    Concat_Layer = Flatten()(Concat_Layer)
    FinalOutput = Dense(1024, activation='relu', kernel_regularizer='l2')(Concat_Layer)
    FinalOutput = BatchNormalization()(FinalOutput)
    FinalOutput = Dropout(.5)(FinalOutput)
    FinalOutput = Dense(512, activation='relu', kernel_regularizer='l2')(FinalOutput)
    FinalOutput = BatchNormalization()(FinalOutput)
    FinalOutput = Dropout(.5)(FinalOutput)
    FinalOutput = Dense(256, activation='relu', kernel_regularizer='l2')(FinalOutput)
    FinalOutput = BatchNormalization()(FinalOutput)
    FinalOutput = Dropout(.5)(FinalOutput)

    EventPrediction = Dense(64, activation='relu', kernel_regularizer='l2')(FinalOutput)
    EventPrediction = BatchNormalization()(EventPrediction)
    EventPrediction = Dropout(.5)(EventPrediction)
    EventPrediction = Dense(classes, activation='softmax', kernel_regularizer='l2', name='event_prediction')(
        EventPrediction)

    CompleteModel = Model(inputs=(InputSignal), outputs=[EventPrediction])
    opt = keras.optimizers.sgd(lr=1e-3, momentum=.9, nesterov=True)
    CompleteModel.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    return CompleteModel


def channel_dil(classes):
    InputSignal = Input(shape=(250, 22))

    ConvBlock1 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=1, kernel_regularizer='l2')(InputSignal)
    ConvBlock1 = BatchNormalization()(ConvBlock1)
    ConvBlock1 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=1, kernel_regularizer='l2')(ConvBlock1)
    ConvBlock1 = BatchNormalization()(ConvBlock1)
    for _ in range(0, 4):
        ConvBlock1 = SeparableConv1D(filters=32, kernel_size=9, strides=2, padding='same', activation='relu',
                                     dilation_rate=1, kernel_regularizer='l2')(ConvBlock1)
        ConvBlock1 = BatchNormalization()(ConvBlock1)

    ConvBlock2 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=3, kernel_regularizer='l2')(InputSignal)
    ConvBlock2 = BatchNormalization()(ConvBlock2)
    ConvBlock2 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=6, kernel_regularizer='l2')(ConvBlock2)
    ConvBlock2 = BatchNormalization()(ConvBlock2)
    for _ in range(0, 4):
        ConvBlock2 = SeparableConv1D(filters=32, kernel_size=9, strides=2, padding='same', activation='relu',
                                     dilation_rate=1, kernel_regularizer='l2')(ConvBlock2)
        ConvBlock2 = BatchNormalization()(ConvBlock2)

    ConvBlock3 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=6, kernel_regularizer='l2')(InputSignal)
    ConvBlock3 = BatchNormalization()(ConvBlock3)
    ConvBlock3 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=9, kernel_regularizer='l2')(ConvBlock3)
    ConvBlock3 = BatchNormalization()(ConvBlock3)
    for _ in range(0, 4):
        ConvBlock3 = SeparableConv1D(filters=32, kernel_size=9, strides=2, padding='same', activation='relu',
                                     dilation_rate=1, kernel_regularizer='l2')(ConvBlock3)
        ConvBlock3 = BatchNormalization()(ConvBlock3)

    Concat_Layer = keras.layers.concatenate([ConvBlock1, ConvBlock2, ConvBlock3])
    Concat_Layer = Flatten()(Concat_Layer)
    FinalOutput = Dense(1024, activation='relu', kernel_regularizer='l2')(Concat_Layer)
    FinalOutput = BatchNormalization()(FinalOutput)
    FinalOutput = Dropout(.5)(FinalOutput)
    FinalOutput = Dense(512, activation='relu', kernel_regularizer='l2')(FinalOutput)
    FinalOutput = BatchNormalization()(FinalOutput)
    FinalOutput = Dropout(.5)(FinalOutput)
    FinalOutput = Dense(256, activation='relu', kernel_regularizer='l2')(FinalOutput)
    FinalOutput = BatchNormalization()(FinalOutput)
    FinalOutput = Dropout(.5)(FinalOutput)

    EventPrediction = Dense(64, activation='relu', kernel_regularizer='l2')(FinalOutput)
    EventPrediction = BatchNormalization()(EventPrediction)
    EventPrediction = Dropout(.5)(EventPrediction)
    EventPrediction = Dense(classes, activation='softmax', kernel_regularizer='l2', name='event_prediction')(EventPrediction)

    ChannelPrediction = Dense(64, activation='relu')(FinalOutput)
    ChannelPrediction = BatchNormalization()(ChannelPrediction)
    ChannelPrediction = Dropout(.5)(ChannelPrediction)
    ChannelPrediction = Dense(22, activation='softmax', name='channel_prediction')(ChannelPrediction)

    CompleteModel = Model(inputs=(InputSignal), outputs=[EventPrediction, ChannelPrediction])
    opt = keras.optimizers.sgd(lr=1e-3, momentum=.9, nesterov=True)
    CompleteModel.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    return CompleteModel


def channel_nodil(classes):
    InputSignal = Input(shape=(250, 22))

    ConvBlock1 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=1, kernel_regularizer='l2')(InputSignal)
    ConvBlock1 = BatchNormalization()(ConvBlock1)
    ConvBlock1 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=1, kernel_regularizer='l2')(ConvBlock1)
    ConvBlock1 = BatchNormalization()(ConvBlock1)
    for _ in range(0, 4):
        ConvBlock1 = SeparableConv1D(filters=32, kernel_size=9, strides=2, padding='same', activation='relu',
                                     dilation_rate=1, kernel_regularizer='l2')(ConvBlock1)
        ConvBlock1 = BatchNormalization()(ConvBlock1)

    ConvBlock2 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=1, kernel_regularizer='l2')(InputSignal)
    ConvBlock2 = BatchNormalization()(ConvBlock2)
    ConvBlock2 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=1, kernel_regularizer='l2')(ConvBlock2)
    ConvBlock2 = BatchNormalization()(ConvBlock2)
    for _ in range(0, 4):
        ConvBlock2 = SeparableConv1D(filters=32, kernel_size=9, strides=2, padding='same', activation='relu',
                                     dilation_rate=1, kernel_regularizer='l2')(ConvBlock2)
        ConvBlock2 = BatchNormalization()(ConvBlock2)

    ConvBlock3 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=1, kernel_regularizer='l2')(InputSignal)
    ConvBlock3 = BatchNormalization()(ConvBlock3)
    ConvBlock3 = SeparableConv1D(filters=32, kernel_size=9, strides=1, padding='same', activation='relu',
                                 dilation_rate=1, kernel_regularizer='l2')(ConvBlock3)
    ConvBlock3 = BatchNormalization()(ConvBlock3)
    for _ in range(0, 4):
        ConvBlock3 = SeparableConv1D(filters=32, kernel_size=9, strides=2, padding='same', activation='relu',
                                     dilation_rate=1, kernel_regularizer='l2')(ConvBlock3)
        ConvBlock3 = BatchNormalization()(ConvBlock3)

    Concat_Layer = keras.layers.concatenate([ConvBlock1, ConvBlock2, ConvBlock3])
    Concat_Layer = Flatten()(Concat_Layer)
    FinalOutput = Dense(1024, activation='relu', kernel_regularizer='l2')(Concat_Layer)
    FinalOutput = BatchNormalization()(FinalOutput)
    FinalOutput = Dropout(.5)(FinalOutput)
    FinalOutput = Dense(512, activation='relu', kernel_regularizer='l2')(FinalOutput)
    FinalOutput = BatchNormalization()(FinalOutput)
    FinalOutput = Dropout(.5)(FinalOutput)
    FinalOutput = Dense(256, activation='relu', kernel_regularizer='l2')(FinalOutput)
    FinalOutput = BatchNormalization()(FinalOutput)
    FinalOutput = Dropout(.5)(FinalOutput)

    EventPrediction = Dense(64, activation='relu', kernel_regularizer='l2')(FinalOutput)
    EventPrediction = BatchNormalization()(EventPrediction)
    EventPrediction = Dropout(.5)(EventPrediction)
    EventPrediction = Dense(classes, activation='softmax', kernel_regularizer='l2', name='event_prediction')(EventPrediction)

    ChannelPrediction = Dense(64, activation='relu')(FinalOutput)
    ChannelPrediction = BatchNormalization()(ChannelPrediction)
    ChannelPrediction = Dropout(.5)(ChannelPrediction)
    ChannelPrediction = Dense(22, activation='softmax', name='channel_prediction')(ChannelPrediction)

    CompleteModel = Model(inputs=(InputSignal), outputs=[EventPrediction, ChannelPrediction])
    opt = keras.optimizers.sgd(lr=1e-3, momentum=.9, nesterov=True)
    CompleteModel.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    return CompleteModel
