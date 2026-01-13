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


def __task1():
    with open('data/dataset_2011.pkl', 'rb') as infile:
        data_2011 = pickle.load(infile)

    hist = plt.hist(data_2011['B invariant mass'], bins=100)
    peaks = find_peaks(hist[0], height=1e5)[0]
    plt.vlines(data_2011['B invariant mass'][peaks], 0, max(hist[0]),
               colors='r', linestyles='dashed', label='peaks')
    plt.xlabel(r'B candidate mass / MeV/$c^2$')
    plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
    plt.show()

    print(
        f"B invariant mass peaks: {data_2011['B invariant mass'][peaks].values}")

    #  remove charm anticharm meson J/psi (dominanat interaction)
    plt.hist(data_2011[abs(data_2011['dimuon-system invariant mass'] -
                           3097) > 100]['B invariant mass'], bins=100)
    plt.xlabel(r'B candidate mass / MeV/$c^2$')
    plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
    plt.show()

    #  remove charm anticharm meson J/psi and psi(2S) (other dominanat interaction)
    plt.hist(data_2011[(abs(data_2011['dimuon-system invariant mass'] - 3097) > 100) &
                       (abs(data_2011['dimuon-system invariant mass'] - 3686) > 100)]['B invariant mass'], bins=100)
    plt.xlabel(r'B candidate mass / MeV/$c^2$')
    plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
    plt.show()

    #  left with rare decays only


def __task2():
    with open('data/dataset_2011.pkl', 'rb') as infile:
        data_2011 = pickle.load(infile)

    hist = plt.hist(data_2011['dimuon-system invariant mass'], bins=100)

    peaks = find_peaks(hist[0], height=1e3, distance=1, prominence=50)[0]

    plt.vlines(data_2011['dimuon-system invariant mass'][peaks], 0, max(hist[0]),
               colors='r', linestyles='dashed', label='peaks')

    plt.xlabel(r'Dimuon invariant mass / MeV/$c^2$')
    plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
    plt.show()

    print(
        f"Invariant mass peaks: {data_2011['dimuon-system invariant mass'][peaks].values}")


def signal_data():
    with open('data/dataset_2011.pkl', 'rb') as infile:
        data_2011 = pickle.load(infile)

    data = data_2011['B invariant mass']
    signal = data[(data <= 5400) & (data >= 5200)]

    hist = plt.hist(signal, bins=100)
    plt.xlabel(r'B candidate mass / MeV/$c^2$')
    plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
    plt.show()

    return signal


def background_data():
    with open('data/dataset_2011.pkl', 'rb') as infile:
        data_2011 = pickle.load(infile)

    data = data_2011['B invariant mass']
    background = data[(data > 5400) | (data < 5200)]
    hist = plt.hist(background, bins=100)
    plt.xlabel(r'B candidate mass / MeV/$c^2$')
    plt.ylabel(r'Candidates / (23 MeV/$c^2)$')
    plt.show()

    return background


if __name__ == "__main__":
    background_data()
