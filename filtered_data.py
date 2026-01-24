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


def load_samesign():
    """
    Load the full 2011 dataset.
    Returns:
    data_2011 : pd.DataFrame
        The full 2011 dataset.
    """
    with open('datasets/samesign_2011.pkl', 'rb') as infile:
        samesign_2011 = pickle.load(infile)

    with open('datasets/samesign_2012.pkl', 'rb') as infile:
        samesign_2012 = pickle.load(infile)

    samesign = pd.concat([samesign_2011, samesign_2012], ignore_index=True)
    # samesign.hist(column='B invariant mass', bins=100)

    return samesign


def load_magnet_data():
    """
    Load the 2012 datasets for Magnet Up and Magnet Down.
    Returns:
    magnet_up_data : pd.DataFrame
        The dataset for Magnet Up.
    magnet_down_data : pd.DataFrame
        The dataset for Magnet Down.
    """
    with open('datasets/dataset_2012_MagnetUp.pkl', 'rb') as infile:
        magnet_up_data = pickle.load(infile)

    with open('datasets/dataset_2012_MagnetDown.pkl', 'rb') as infile:
        magnet_down_data = pickle.load(infile)

    #  concatinate both datasets with polarity column for easier processing

    magnet_up_data['polarity'] = 1
    magnet_down_data['polarity'] = 0

    dataset = pd.concat([magnet_up_data, magnet_down_data], ignore_index=True)

    return dataset


def load_2011_data():
    """
    Load the full 2011 dataset.
    Returns:
    data_2011 : pd.DataFrame
        The full 2011 dataset.
    """
    with open('datasets/dataset_2011.pkl', 'rb') as infile:
        dataset_2011 = pickle.load(infile)

    return dataset_2011

    # print('\n'.join(dataset.keys()))


class seperate:
    def __init__(self, k: int = None, plot: bool = False, dataset: str = '2011'):
        """
        Separate the 2011 dataset into signal and background based on B invariant mass.
        Parameters
        ----------
        k: int
            Number of folds to separate the data into.
        plot: bool
            Whether to plot the histograms of the signal and background datasets.
        """
        # from correlation import drop_correlated
        # dataset = drop_correlated('B invariant mass', _dataset, threshold=0.5)
        self.__splits = []

        self.__dataset = dataset
        if dataset == '2011':
            dataset = load_2011_data()

        elif dataset == '2012':
            dataset = load_magnet_data()

        else:
            raise ValueError(
                "dataset must be either '2011' or '2012'")

        # is_signal = (dataset['dimuon-system invariant mass'].between(3070, 3200)
        #              ) | dataset['dimuon-system invariant mass'].between(3600, 3750)

        is_signal = dataset['dimuon-system invariant mass'].between(3600, 3750)

        is_background = (dataset['B invariant mass'] > 5400)

        #  not used unless k=1
        signal = dataset[is_signal]
        background = dataset[is_background]

        if plot is True:
            hist = plt.hist(signal['dimuon-system invariant mass'], bins=50)
            plt.xlabel(r'B candidate mass / MeV/$c^2$')
            plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
            plt.show()

            hist = plt.hist(background['B invariant mass'], bins=100)

            plt.xlabel(r'B candidate mass / MeV/$c^2$')
            plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
            plt.show()

        dataset['label'] = -1
        dataset.loc[is_signal, 'label'] = 1
        dataset.loc[is_background, 'label'] = 0
        dataset = dataset[dataset['label'] != -1].reset_index(drop=True)

        samesign = load_samesign()
        samesign['label'] = 0  # all background
        dataset = pd.concat([dataset, samesign],
                            ignore_index=True)

        if k is not None:
            data_shuffled = dataset.sample(
                frac=1, random_state=42).reset_index(drop=True)
            full_parts = np.array_split(data_shuffled, k)

            self.__splits = []
            self.__signal_parts = []
            self.__background_parts = []

            for i in range(k):
                current_fold = full_parts[i]

                sig_k = current_fold[current_fold['label'] == 1]
                bkg_k = current_fold[current_fold['label'] == 0]
                sig_k = sig_k.drop(columns=['label'])
                bkg_k = bkg_k.drop(columns=['label'])
                current_fold = current_fold.drop(columns=['label'])
                self.__signal_parts.append(sig_k)
                self.__background_parts.append(bkg_k)
                self.__splits.append(current_fold)

        if k is None:
            self.__signal_parts = signal
            self.__background_parts = background

    def data(self, drop_cols: list = None,):
        """
        Return the separated signal and background datasets.
        Parameters
        ----------
        drop_cols: list
            List of columns to drop from the datasets.
        """
        signal = self.__signal_parts
        background = self.__background_parts

        if drop_cols:
            if isinstance(signal, list):
                signal = [df.drop(columns=drop_cols, errors='ignore')
                          for df in signal]
                background = [df.drop(columns=drop_cols, errors='ignore')
                              for df in background]

            else:
                signal = signal.drop(columns=drop_cols, errors='ignore')
                background = background.drop(
                    columns=drop_cols, errors='ignore')

        return signal, background

    def dataset_k(self, k, drop_cols: list = None):
        """
        Return the k-th fold of the separated dataset.

        Parameters
        ----------
        k: int
            The fold index to return.
        drop_cols: list (optional)
            List of columns to drop from the returned dataframe.
        """
        idx = k % len(self.__splits)

        df = self.__splits[idx]
        if drop_cols:
            return df.drop(columns=drop_cols, errors='ignore')
        return df

    def which_dataset(self):
        return self.__dataset


# %% Initial B invariant mass filtering
def __task1():
    dataset = load_2011_data()
    hist = plt.hist(dataset['B invariant mass'], bins=100)
    peaks = find_peaks(hist[0], height=1e5)[0]
    plt.vlines(dataset['B invariant mass'][peaks], 0, max(hist[0]),
               colors='r', linestyles='dashed', label='peaks')
    plt.xlabel(r'B candidate mass / MeV/$c^2$')
    plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
    plt.show()

    print(
        f"B invariant mass peaks: {dataset['B invariant mass'][peaks].values}")

    #  remove charm anticharm meson J/psi (dominanat interaction)
    plt.hist(dataset[abs(dataset['dimuon-system invariant mass'] -
                         3097) > 100]['B invariant mass'], bins=100)
    plt.xlabel(r'B candidate mass / MeV/$c^2$')
    plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
    plt.show()

    #  remove charm anticharm meson J/psi and psi(2S) (other dominanat interaction)
    plt.hist(dataset[(abs(dataset['dimuon-system invariant mass'] - 3097) > 100) &
                     (abs(dataset['dimuon-system invariant mass'] - 3686) > 100)]['B invariant mass'], bins=100)
    plt.xlabel(r'B candidate mass / MeV/$c^2$')
    plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
    plt.show()

    #  left with rare decays only


def __task2():
    dataset = load_2011_data()
    hist = plt.hist(dataset['dimuon-system invariant mass'], bins=100)

    peaks = find_peaks(hist[0], height=1e3, distance=1, prominence=50)[0]

    plt.vlines(dataset['dimuon-system invariant mass'][peaks], 0, max(hist[0]),
               colors='r', linestyles='dashed', label='peaks')

    plt.xlabel(r'Dimuon invariant mass / MeV/$c^2$')
    plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
    plt.show()

    print(
        f"Invariant mass peaks: {dataset['dimuon-system invariant mass'][peaks].values}")


if __name__ == "__main__":
    load_samesign()
