"""
Count the number of B mesons in the dataset, and seperate them into B+ and B- mesons.
20/01 - created
"""

import pickle
import math
import dimuon_binning
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style('darkgrid')
sns.set_context('paper')


# %% Data Loading Functions


def __load_signal_data():
    """
    Load the cleaned signal 2011 dataset after background fitting and weighting.
    Returns:
    pd.DataFrame
        The cleaned dataset with event weights applied.
    """
    with open('data/cleaned_data_2011.pkl', 'rb') as f:
        cleaned_data = pickle.load(f)
    return cleaned_data


def __load_cleaned_mag_data():
    """
    Load the cleaned magnetic 2012 dataset after background fitting and weighting.
    Returns:
    pd.DataFrame
        The cleaned dataset with event weights applied.
    """
    with open('data/cleaned_data_2012.pkl', 'rb') as f:
        cleaned_data = pickle.load(f)

    mag_up = cleaned_data[cleaned_data['polarity'] == 1]
    mag_down = cleaned_data[cleaned_data['polarity'] == 0]
    return mag_up, mag_down


# %% CP Asymmetry Calculations

def compute_b_asymmetry(B_plus_count, B_minus_count, N_plus_uncertainty, N_minus_uncertainty):
    weighted_total = B_plus_count + B_minus_count
    cp_asy = (B_plus_count - B_minus_count) / weighted_total


    uncertainty = 2/(weighted_total**2) * math.sqrt(
        (B_minus_count * N_plus_uncertainty)**2 +
        (B_plus_count * N_minus_uncertainty)**2
    )
    return cp_asy, uncertainty


# %% Main Execution

def compute_asymmetry(data, plot: bool = False, n_bins: int = 10):
    counts, uncertaintes = dimuon_binning.B_counts(data, n_bins=n_bins)

    asy = []
    for bin_counts, count_uncertainty in zip(counts, uncertaintes):
        N_plus, N_minus = bin_counts
        N_plus_unc, N_minus_unc = count_uncertainty
        cp_asy, uncertainty = compute_b_asymmetry(N_plus, N_minus, N_plus_unc, N_minus_unc)
        asy.append((cp_asy, uncertainty))

    if plot:
        plt.errorbar(range(len(asy)), [a[0] for a in asy], yerr=[a[1] for a in asy], fmt='o')
        plt.xlabel('Dimuon Mass Bin')
        plt.ylabel('CP Asymmetry')
        plt.title('CP Asymmetry vs Dimuon Mass Bin (2011 Signal Data)')
        plt.axhline(0, color='red', linestyle='--')
        plt.show()

    return asy

if __name__ == "__main__":
    n_bins = 20
    signal_data = __load_signal_data()
    compute_asymmetry(signal_data, plot=True, n_bins=n_bins)
    
    mag_up, mag_down = __load_cleaned_mag_data()
    compute_asymmetry(mag_up, plot=True, n_bins=n_bins)
    compute_asymmetry(mag_down, plot=True, n_bins=n_bins)
