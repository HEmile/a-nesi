import gzip
import os

from typing import Tuple

import torch
from torch.utils.data import random_split
import numpy as np

TRAIN_INPUTS_FILENAME = 'train_puzzle_pixels.txt'
TEST_INPUTS_FILENAME = 'test_puzzle_pixels.txt'

TRAIN_LABELS_FILENAME = 'train_puzzle_labels.txt'
TEST_LABELS_FILENAME = 'test_puzzle_labels.txt'

OPTIONS_FILENAME = 'options.json'

SOURCE_DIR = '/ViSudo-PC/'

# MNIST images are 28 x 28 = 784.
MNIST_DIMENSION = 28


def writeFile(path, data, dtype=str, compress=False):
    if (compress):
        file = gzip.open(path, 'wt')
    else:
        file = open(path, 'w')

    for row in data:
        file.write('\t'.join([str(dtype(item)) for item in row]) + "\n")

    file.close()


def convertToInts(dataset: np.array):
    values = list(sorted(set(dataset.flatten().tolist())))

    valueMap = {}
    for value in values:
        valueMap[value] = len(valueMap)

    trainOut = []

    # for (outData, inData) in [[trainOut, train], [testOut, test]]:
    for row in dataset:
        trainOut.append([valueMap[value] for value in row])

    train = np.stack(trainOut)

    return train


def load_data(dataDir: str, partition: str):
    features_file = f'{partition}_puzzle_pixels.txt'

    labels_file = f'{partition}_puzzle_labels.txt'
    labels = np.loadtxt(os.path.join(dataDir, labels_file), delimiter="\t", dtype=str)

    labels = convertToInts(labels)

    features = np.loadtxt(os.path.join(dataDir, features_file), delimiter="\t", dtype=float)

    return features, labels


class PuzzleDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            dataset_dir: str,
            partition: str,
            N: int = 9,
            use_negative: bool=True
    ):
        super(PuzzleDataset, self).__init__()
        self.features, self.labels = load_data(dataset_dir, partition)
        self.features = torch.tensor(self.features, dtype=torch.float)
        self.labels = 1 - torch.max(torch.tensor(self.labels), dim=1)[1]
        if not use_negative:
            self.features = self.features[self.labels == 1]
            self.labels = self.labels[self.labels == 1]
        self.N = N

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        grids = self.features[index]
        grids = grids.reshape(-1, self.N * self.N, MNIST_DIMENSION * MNIST_DIMENSION)
        label = self.labels[index]
        return grids, label


def get_datasets(split: int, basepath: str=".", dimension: int = 9, numTrain: str = "00100", overlap: str = "0.00", use_negative_train=True) -> Tuple[
    PuzzleDataset, PuzzleDataset, PuzzleDataset]:
    spl = str(split)
    if split < 10:
        spl = "0" + spl
    dataDir = os.path.join(basepath + SOURCE_DIR, f"ViSudo-PC_dimension::{dimension}_datasets::mnist_strategy::simple/"
                                       f"dimension::{dimension}/datasets::mnist/strategy::simple/strategy::simple/"
                                       f"numTrain::{numTrain}/numTest::00100/numValid::00100/corruptChance::0.50/"
                                       f"overlap::{overlap}/split::{spl}")
    train = PuzzleDataset(dataDir, 'train', dimension, use_negative=use_negative_train)
    valid = PuzzleDataset(dataDir, 'valid', dimension)
    test = PuzzleDataset(dataDir, 'test', dimension)
    return train, valid, test
