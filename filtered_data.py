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
sns.set_context('talk')
sns.set_palette("colorblind")


def load_simulation_data():
    """
    Load pure signal simulation dataset.
    """

    with open('datasets/rapidsim_Kmumu.pkl', 'rb') as infile:
        simulation_data = pickle.load(infile)

    # print('\n'.join(simulation_data.keys()))
    return simulation_data


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

    magnet_up_data['Magnet polarity'] = 1
    magnet_down_data['Magnet polarity'] = -1

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

    # print('\n'.join(dataset_2011.keys()))

    dataset_2011 = dataset_2011[dataset_2011['Magnet polarity'] == 1]

    return dataset_2011


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

        # elif dataset == 'simulation':
        #     dataset = load_simulation_data()

        else:
            raise ValueError(
                "dataset must be either '2011' or '2012'")

        #  signal selection criteria
        is_signal = (dataset['dimuon-system invariant mass'].between(2950, 3200)
                     ) | dataset['dimuon-system invariant mass'].between(3600, 3750)

        # is_signal = dataset['dimuon-system invariant mass'].between(3070, 3200)

        #  background selection criteria
        is_background = (dataset['B invariant mass'] > 5370)

        #  not used unless k=1
        signal = dataset[is_signal]
        background = dataset[is_background]

        if plot is True:
            fig, ax = plt.subplots(2, 1, figsize=(10, 6))
            col_name = 'dimuon-system invariant mass'
            x_min = dataset[col_name].min()
            x_max = dataset[col_name].max()
            common_bins = np.linspace(x_min, x_max, 150)

            ax[0].hist(dataset[col_name],
                       bins=common_bins,
                       color='lightgray',
                       label='Total Dataset',
                       edgecolor='black')

            ax[0].hist(signal[col_name],
                       bins=common_bins,
                       color='tab:blue',       # distinct color for signal
                       alpha=0.8,              # high opacity to make it pop
                       label='Signal',
                       edgecolor='black',
                       linewidth=0.5)

            ax[0].set_xlabel(
                r'Dimuon-system Invariant Mass [MeV/$c^2$]', fontsize=14)
            # Dynamically update the label with the correct width
            ax[0].set_ylabel(
                f'log(Count)', fontsize=14)
            ax[0].set_title(
                'Signal and Background selection', fontsize=16)
            ax[0].legend(fontsize=12, loc='upper right')
            ax[0].set_yscale('log')

            x_min = dataset['B invariant mass'].min()
            x_max = dataset['B invariant mass'].max()
            common_bins = np.linspace(x_min, x_max, 150)

            # 2. Plot the Full Dataset first (in the background, lighter color)
            ax[1].hist(dataset['B invariant mass'],
                       bins=common_bins,
                       color='lightgray',      # Neutral color
                       label='Total Dataset',
                       edgecolor='black')       # No edges for cleaner look

            # 3. Plot the Background Data on top (Highlighted)
            ax[1].hist(background['B invariant mass'],
                       bins=common_bins,
                       color='tab:red',        # Bright color to highlight
                       alpha=0.7,              # Slight transparency
                       label='Background',
                       edgecolor='black',      # Add edge to pop out
                       linewidth=0.5)

            ax[1].legend(fontsize=12, loc='upper right')
            ax[1].set_yscale('log')
            ax[1].set_xlabel(r'B Invariant Mass [MeV/$c^2$]', fontsize=14)
            ax[1].set_ylabel(r'log(Count)')
            plt.show()

            background.hist(column='dimuon-system invariant mass', bins=100)
            plt.xlabel(r'Dimuon invariant mass / MeV/$c^2$')
            plt.ylabel(r'Candidates / (23 MeV/$c^2$)')
            plt.title("Background Dataset")
            plt.yscale('log')
            plt.show()
            signal.hist(column='B invariant mass', bins=100)
            plt.xlabel(r'Dimuon invariant mass / MeV/$c^2$')
            plt.ylabel(r'Candidates / (23 MeV/$c^2$)')
            plt.title("Signal Dataset")
            plt.yscale('log')
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
            # 1. Shuffle and split the data into k equal-sized chunks
            data_shuffled = dataset.sample(
                frac=1, random_state=42).reset_index(drop=True)
            full_parts = np.array_split(data_shuffled, k)

            self.__splits = []
            self.__signal_parts = []
            self.__background_parts = []

            for i in range(k):
                evaluation_fold = full_parts[i]
                training_parts = [full_parts[j] for j in range(k) if j != i]
                training_set = pd.concat(training_parts)

                sig_train = training_set[training_set['label'] == 1].drop(columns=[
                                                                          'label'])
                bkg_train = training_set[training_set['label'] == 0].drop(columns=[
                                                                          'label'])

                self.__signal_parts.append(sig_train)
                self.__background_parts.append(bkg_train)

                self.__splits.append(evaluation_fold.drop(columns=['label']))

        elif k is None:
            # Standard logic for no folding
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
    seperate = seperate(k=5, plot=True, dataset='2011')
