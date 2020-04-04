import pandas as pd
import numpy as np
from sklearn import preprocessing


def normalize(dataset):
    return preprocessing.normalize(dataset, axis=1)


def prepare_data(filename, normalization=True):
    dataframe = pd.read_csv(filename)

    np_dataset = np.array(dataframe)

    features, labels = separate_features_labels(np_dataset)

    normalized_features = normalize(features) if normalization == True else features

    np_dataset = np.concatenate((normalized_features, np.reshape(labels, (-1, 1))), axis=1)

    return np_dataset, normalized_features, labels


def separate_features_labels (dataset):
    features = dataset[:, :-1]
    labels = dataset[:, -1]

    return features, labels


def read_from_file (filename):
    dataframe = pd.read_csv(filename)

    np_dataset = np.array(dataframe)

    features, labels = separate_features_labels(np_dataset)

    return features, labels