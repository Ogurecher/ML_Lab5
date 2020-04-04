import numpy as np


def train_test_split (dataset, train_ratio):
    np.random.shuffle(dataset)

    split_point = round(len(dataset) * train_ratio)

    train_set = dataset[:split_point]
    test_set = dataset[split_point:]

    train_features, train_labels = features_labels_split(train_set)
    test_features, test_labels = features_labels_split(test_set)

    return train_set, train_features, train_labels, test_set, test_features, test_labels


def features_labels_split (dataset):
    features = dataset[:, :-1]
    labels = dataset[:, -1]

    return features, labels
