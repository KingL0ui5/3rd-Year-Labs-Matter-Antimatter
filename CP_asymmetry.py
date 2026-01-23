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

    print(len(cleaned_data))
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
    
    # Safety Check: If the bin is empty, return 0 for both asymmetry and uncertainty
    if weighted_total <= 0:
        return 0.0, 0.0

    cp_asy = (B_plus_count - B_minus_count) / weighted_total

    # Standard error propagation
    uncertainty = 2/(weighted_total**2) * math.sqrt(
        (B_minus_count * N_plus_uncertainty)**2 +
        (B_plus_count * N_minus_uncertainty)**2
    )
    return cp_asy, uncertainty


# %% Main Execution

def compute_asymmetry(data, plot: bool = False):
    n_bins = 20
    counts, uncertainties, inv_mass = dimuon_binning.B_counts(data, n_bins)

    asy = []
    for bin_counts, count_uncertainty in zip(counts, uncertainties):
        B_plus, B_minus = bin_counts

        # print(f"Total counts in bin: {B_plus + B_minus}")
        B_plus_unc, B_minus_unc = count_uncertainty
        
        cp_asy, uncertainty = compute_b_asymmetry(B_plus, B_minus, B_plus_unc, B_minus_unc)
        asy.append((cp_asy, uncertainty))

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
        data.hist('dimuon-system invariant mass', bins=100, alpha=0.5, ax=ax[0])
        ax[0].vlines(inv_mass, ymin=0, ymax=10e5, colors='red', linestyles='dashed', label='Bin Edges')
        ax[0].set_yscale('log')
        ax[0].set_ylabel('Counts')
        ax[0].set_title('Dimuon Invariant Mass Distribution (Signal Data)')

        ax[1].errorbar(inv_mass, [a[0] for a in asy], yerr=[a[1] for a in asy], fmt='-o', markersize=3, linewidth=1)
        ax[1].set_xlabel('Invariant mass')
        ax[1].set_ylabel('CP Asymmetry')
        ax[1].axhline(0, color='red', linestyle='--')
        ax[1].set_ylim(-0.25,0.25)
        plt.title('CP Asymmetry vs Dimuon Mass Bin (Signal Data)')
        plt.show()

    return asy

if __name__ == "__main__":
    signal_data = __load_signal_data()
    compute_asymmetry(signal_data, plot=True)
    
    mag_up, mag_down = __load_cleaned_mag_data()
    compute_asymmetry(mag_up, plot=True)
    compute_asymmetry(mag_down, plot=True)
