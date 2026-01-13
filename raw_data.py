"""
Task 1 09/01
Plot invariant mass histograms for 2011 dataset 
"""

import pickle
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
sns.set_style('darkgrid')
sns.set_context('paper')
sns.set_palette("colorblind")

with open('data/dataset_2011.pkl', 'rb') as infile:
    __data_2011 = pickle.load(infile)


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


def data():
    dataset = __data_2011
    signal = dataset[(dataset['B invariant mass'] <= 5400) &
                     (dataset['B invariant mass'] >= 5200)]

    hist = plt.hist(signal, bins=100)
    plt.xlabel(r'B candidate mass / MeV/$c^2$')
    plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
    plt.show()

    background = dataset[(dataset['B invariant mass'] > 5400)
                         | (dataset['B invariant mass'] < 5200)]
    hist = plt.hist(background, bins=100)

    plt.xlabel(r'B candidate mass / MeV/$c^2$')
    plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
    plt.show()

    return signal, background


def samesign():
    data = __data_2011['Same-sign muon assumed particle type']
    hist = plt.hist(data, bins=150)
    plt.xlabel(r'Same-sign muon invariant mass / MeV/$c^2$')
    plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
    plt.show()
    return data


if __name__ == "__main__":
    samesign()
