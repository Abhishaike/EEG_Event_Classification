import mne
import numpy as np
import os
import pickle

def BuildEvents(signals, times, EventData):
    [numEvents, z] = (EventData.shape)  # numEvents is equal to # of rows of the .rec file
    fs = 250.0
    [numChan, numPoints] = signals.shape
    for i in range(numChan):  # standardize each channel
        if np.std(signals[i, :]) > 0:
            signals[i, :] = (signals[i, :] - np.mean(signals[i, :])) / np.std(signals[i, :])
    features = np.zeros([numEvents, numChan, int(fs)])
    offending_channel = np.zeros([numEvents, 1]) #channel that had the detected thing
    labels = np.zeros([numEvents, 1])
    for i in range(numEvents):  # for each event
        chan = int(EventData[i, 0])  # chan is channel
        start = np.where((times) >= EventData[i, 1])[0][0]
        end = np.where((times) >= EventData[i, 2])[0][0]
        features[i, :] = signals[:, start:end]
        offending_channel[i, :] = int(chan)
        labels[i, :] = int(EventData[i, 3])
    return [features, offending_channel, labels]


def convert_signals(signals, Rawdata):
    signal_names = {k: v for (k, v) in zip(Rawdata.info['ch_names'], list(range(len(Rawdata.info['ch_names']))))}
    new_signals = np.vstack((signals[signal_names['EEG FP1-REF']] - signals[signal_names['EEG F7-REF']],  # 0
                             (signals[signal_names['EEG F7-REF']] - signals[signal_names['EEG T3-REF']]),  # 1
                             (signals[signal_names['EEG T3-REF']] - signals[signal_names['EEG T5-REF']]),  # 2
                             (signals[signal_names['EEG T5-REF']] - signals[signal_names['EEG O1-REF']]),  # 3
                             (signals[signal_names['EEG FP2-REF']] - signals[signal_names['EEG F8-REF']]),  # 4
                             (signals[signal_names['EEG F8-REF']] - signals[signal_names['EEG T4-REF']]),  # 5
                             (signals[signal_names['EEG T4-REF']] - signals[signal_names['EEG T6-REF']]),  # 6
                             (signals[signal_names['EEG T6-REF']] - signals[signal_names['EEG O2-REF']]),  # 7
                             (signals[signal_names['EEG A1-REF']] - signals[signal_names['EEG T3-REF']]),  # 8
                             (signals[signal_names['EEG T3-REF']] - signals[signal_names['EEG C3-REF']]),  # 9
                             (signals[signal_names['EEG C3-REF']] - signals[signal_names['EEG CZ-REF']]),  # 10
                             (signals[signal_names['EEG CZ-REF']] - signals[signal_names['EEG C4-REF']]),  # 11
                             (signals[signal_names['EEG C4-REF']] - signals[signal_names['EEG T4-REF']]),  # 12
                             (signals[signal_names['EEG T4-REF']] - signals[signal_names['EEG A2-REF']]),  # 13
                             (signals[signal_names['EEG FP1-REF']] - signals[signal_names['EEG F3-REF']]),  # 14
                             (signals[signal_names['EEG F3-REF']] - signals[signal_names['EEG C3-REF']]),  # 15
                             (signals[signal_names['EEG C3-REF']] - signals[signal_names['EEG P3-REF']]),  # 16
                             (signals[signal_names['EEG P3-REF']] - signals[signal_names['EEG O1-REF']]),  # 17
                             (signals[signal_names['EEG FP2-REF']] - signals[signal_names['EEG F4-REF']]),  # 18
                             (signals[signal_names['EEG F4-REF']] - signals[signal_names['EEG C4-REF']]),  # 19
                             (signals[signal_names['EEG C4-REF']] - signals[signal_names['EEG P4-REF']]),  # 20
                             (signals[signal_names['EEG P4-REF']] - signals[signal_names['EEG O2-REF']]))) # 21
    return new_signals


def readEDF(fileName):
    Rawdata = mne.io.read_raw_edf(fileName)
    signals, times = Rawdata[:]
    RecFile = fileName[0:-3] + 'rec'
    eventData = np.genfromtxt(RecFile, delimiter=",")
    Rawdata.close()
    return [signals, times, eventData, Rawdata]

def load_up_objects(BaseDir, Features, OffendingChannels, Labels):
    for dirName, subdirList, fileList in os.walk(BaseDir):
        print('Found directory: %s' % dirName)
        for fname in fileList:
            if fname[-4:] == '.edf':
                print('\t%s' % fname)
                try:
                    [signals, times, event, Rawdata] = readEDF(
                        dirName + '/' + fname)  # event is the .rec file in the form of an array
                    signals = convert_signals(signals, Rawdata)
                except (ValueError, KeyError):
                    print('something funky happened in ' + dirName + '/' + fname)
                    continue
                [features, offending_channel, labels] = BuildEvents(signals, times, event)
                Features = np.append(Features, features, axis=0)
                OffendingChannels = np.append(OffendingChannels, offending_channel, axis=0)
                Labels = np.append(Labels, labels, axis=0)
    return Features, Labels, OffendingChannels


def save_pickle(object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(object, f)

BaseDirTrain = 'EEGData/v1.0.0/edf/train'
fs = 250
TrainFeatures = np.empty((0, 22, fs)) #0 for lack of intialization, 22 for channels, fs for num of points
TrainLabels = np.empty([0, 1])
TrainOffendingChannel = np.empty([0, 1])
TrainFeatures, TrainLabels, TrainOffendingChannel = load_up_objects(BaseDirTrain, TrainFeatures, TrainLabels, TrainOffendingChannel)
TrainFeatures = np.rollaxis(TrainFeatures, 2, 1) #switch around axis for Keras

BaseDirEval = 'EEGData/v1.0.0/edf/eval'
fs = 250
EvalFeatures = np.empty((0, 22, fs)) #0 for lack of intialization, 22 for channels, fs for num of points
EvalLabels = np.empty([0, 1])
EvalOffendingChannel = np.empty([0, 1])
EvalFeatures, EvalLabels, EvalOffendingChannel = load_up_objects(BaseDirEval, EvalFeatures, EvalLabels, EvalOffendingChannel)
EvalFeatures = np.rollaxis(EvalFeatures, 2, 1) #switch around axis for Keras


save_pickle(TrainFeatures, 'ExtractedData/TrainFeatures')
save_pickle(TrainLabels, 'ExtractedData/TrainLabels')
save_pickle(TrainOffendingChannel, 'ExtractedData/TrainOffendingChannel')

save_pickle(EvalFeatures, 'ExtractedData/EvalFeatures')
save_pickle(EvalLabels, 'ExtractedData/EvalLabels')
save_pickle(EvalOffendingChannel, 'ExtractedData/EvalOffendingChannel')
