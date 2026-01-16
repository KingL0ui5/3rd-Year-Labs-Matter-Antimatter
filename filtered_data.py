"""
Task 1 09/01
Plot invariant mass histograms for 2011 dataset 
"""

import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
sns.set_style('darkgrid')
sns.set_context('paper')
sns.set_palette("colorblind")


with open('datasets/dataset_2011.pkl', 'rb') as infile:
    __data_2011 = pickle.load(infile)

__data_2011.insert('index', range(len(__data_2011)))  # add index column


def raw_data():
    """
    Return a copy of the full 2011 dataset.
    """
    return __data_2011.copy()


def samesign():
    data = __data_2011['Same-sign muon assumed particle type']
    hist = plt.hist(data, bins=150)
    plt.xlabel(r'Same-sign muon invariant mass / MeV/$c^2$')
    plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
    plt.show()
    return data


class seperate:
    def __init__(self, drop_cols: list = None, k: int = None, plot: bool = False):
        """
        Separate the 2011 dataset into signal and background based on B invariant mass.
        drop_cols: list
            List of columns to drop from the dataset in addition to correlated ones.
        k: int
            Number of folds to separate the data into.
        plot: bool
            Whether to plot the histograms of the signal and background datasets.
        """
        # from correlation import drop_correlated
        # dataset = drop_correlated('B invariant mass', __data_2011, threshold=0.5)
        self.__datasets = []

        dataset = __data_2011.copy()
        signal = dataset[dataset['dimuon-system invariant mass'].between(3070, 3200) |
                         dataset['dimuon-system invariant mass'].between(3600, 3750)]
        background = dataset[(dataset['B invariant mass'] > 5400)]

        if plot is True:
            hist = plt.hist(signal['dimuon-system invariant mass'], bins=50)
            plt.xlabel(r'B candidate mass) / MeV/$c^2$')
            plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
            plt.show()

            hist = plt.hist(background['B invariant mass'], bins=100)

            plt.xlabel(r'B candidate mass / MeV/$c^2$')
            plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
            plt.show()

        if drop_cols:
            signal = signal.drop(columns=drop_cols)
            background = background.drop(columns=drop_cols)

        if k is not None:
            signal_shuffled = signal.sample(
                frac=1, random_state=42).reset_index(drop=True)
            self.__signal_parts = np.array_split(signal_shuffled, k)

            background_shuffled = background.sample(
                frac=1, random_state=42).reset_index(drop=True)
            self.__background_parts = np.array_split(background_shuffled, k)

            for i in range(k):
                dataset_k = pd.concat(
                    [self.__signal_parts[i], self.__background_parts[i]])
                self.__datasets.append(dataset_k)

        if k is None:
            self.__signal_parts = signal
            self.__background_parts = background

    def data(self):
        return self.__signal_parts, self.__background_parts

    def dataset_k(self, k):
        if k >= len(self.__datasets):
            k = len(self.__datasets) - k

        return self.__datasets[k]


# %% Initial B invariant mass filtering
def __task1():
    hist = plt.hist(__data_2011['B invariant mass'], bins=100)
    peaks = find_peaks(hist[0], height=1e5)[0]
    plt.vlines(__data_2011['B invariant mass'][peaks], 0, max(hist[0]),
               colors='r', linestyles='dashed', label='peaks')
    plt.xlabel(r'B candidate mass / MeV/$c^2$')
    plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
    plt.show()

    print(
        f"B invariant mass peaks: {__data_2011['B invariant mass'][peaks].values}")

    #  remove charm anticharm meson J/psi (dominanat interaction)
    plt.hist(__data_2011[abs(__data_2011['dimuon-system invariant mass'] -
                             3097) > 100]['B invariant mass'], bins=100)
    plt.xlabel(r'B candidate mass / MeV/$c^2$')
    plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
    plt.show()

    #  remove charm anticharm meson J/psi and psi(2S) (other dominanat interaction)
    plt.hist(__data_2011[(abs(__data_2011['dimuon-system invariant mass'] - 3097) > 100) &
                         (abs(__data_2011['dimuon-system invariant mass'] - 3686) > 100)]['B invariant mass'], bins=100)
    plt.xlabel(r'B candidate mass / MeV/$c^2$')
    plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
    plt.show()

    #  left with rare decays only


def get_dimuon_peaks(data):
    """
    Detect peaks in dimuon-system invariant mass distribution.
    Returns the indices of peaks in the histogram bins.
    """
    hist = plt.hist(data['dimuon-system invariant mass'], bins=100)
    peaks = find_peaks(hist[0], height=1e3, distance=1, prominence=50)[0]
    plt.close()
    return peaks, hist


def __task2():
    hist = plt.hist(__data_2011['dimuon-system invariant mass'], bins=100)

    peaks = find_peaks(hist[0], height=1e3, distance=1, prominence=50)[0]

    plt.vlines(__data_2011['dimuon-system invariant mass'][peaks], 0, max(hist[0]),
               colors='r', linestyles='dashed', label='peaks')

    plt.xlabel(r'Dimuon invariant mass / MeV/$c^2$')
    plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
    plt.show()

    print(
        f"Invariant mass peaks: {__data_2011['dimuon-system invariant mass'][peaks].values}")
