"""
Count the number of B mesons in the dataset, and seperate them into B+ and B- mesons.
20/01 - created
"""

import pickle
import math

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


def __load_rare_decay_data():
    """
    Load the cleaned rare decay dataset after background fitting and weighting.
    Returns:
    pd.DataFrame
        The cleaned dataset with event weights applied.
    """
    with open('data/rare_decay_data.pkl', 'rb') as f:
        cleaned_data = pickle.load(f)
    return cleaned_data


def __load_simulation_data():
    """
    Load the simulation datasets for J/psi K and K mu mu.
    Returns:
    JpsiK_data : pd.DataFrame
        The dataset for the J/psi K simulation.
    Kmumu_data : pd.DataFrame
        The dataset for the K mu mu simulation."""
    with open('datasets/rapidsim_JpsiK.pkl', 'rb') as infile:
        JpsiK_data = pickle.load(infile)

    with open('datasets/rapidsim_Kmumu.pkl', 'rb') as infile:
        Kmumu_data = pickle.load(infile)

    return JpsiK_data, Kmumu_data

# %% CP Asymmetry Calculations

def count_B_mesons(data):
    """
    Count the total number of B mesons in the dataset, and seperate them into B+ and B- mesons.
    returns:
    total_B : int
        Total number of B mesons in the dataset
    total_B_plus : int
        Total number of B+ mesons in the dataset
    total_B_minus : int
        Total number of B- mesons in the dataset
    """
    total_B = data.shape[0]
    total_B_plus = data[data['Kaon assumed particle type'] > 0].shape[0]
    total_B_minus = data[data['Kaon assumed particle type'] < 0].shape[0]

    return total_B, total_B_plus, total_B_minus

def compute_b_asymmetry(data):
    """
    Computes the CP asymmetry for the B mesons in the dataset given.
    parameters:
    data : pd.DataFrame
        The dataset to evaluate.
    returns:
    cp_asy : float
        The CP asymmetry value for the B mesons in the dataset.
    """
    # count the B mesons
    total_B, total_B_plus, total_B_minus = count_B_mesons(data)
    # compute asymmetry
    cp_asy = (total_B_plus - total_B_minus) / (total_B)
    return cp_asy

def compute_asymmetry_uncertainty(data):
    """
    Compute the statistical uncertainty on the CP asymmetry measurement.
    Uses the propagation of Poisson errors.

    parameters:
    data : pd.DataFrame
        The dataset to evaluate.

    returns:
    uncertainty : float
        The statistical uncertainty on the CP asymmetry.
    """
    N_total, N_plus, N_minus = count_B_mesons(data)

    if N_total <= 0:
        return 0.0
    uncertainty = (2 * math.sqrt(N_plus * N_minus)) / (N_total**1.5)
    return uncertainty

def compute_peaks_asymmetry(data):
    peaks_data = data[
        (data['dimuon-system invariant mass'].between(3000, 3150)) | 
        (data['dimuon-system invariant mass'].between(3600, 3750))
    ]

    cpa = compute_b_asymmetry(peaks_data)
    cpa_uncert = compute_asymmetry_uncertainty(peaks_data)

    return cpa, cpa_uncert

def compute_rare_asymmetry(data):
    """
    Computes CP asymmetry for the non-resonant mass regions.
    """
    rare_data = data[~(
        (data['dimuon-system invariant mass'].between(3000, 3150)) | 
        (data['dimuon-system invariant mass'].between(3600, 3750)))
    ]

    cpa = compute_b_asymmetry(rare_data)
    cpa_uncert = compute_asymmetry_uncertainty(rare_data)

    return cpa, cpa_uncert

# %% Main Execution

if __name__ == "__main__":
    signal_data = __load_signal_data()
    cpa_rare, uncertainty_rare = compute_rare_asymmetry(signal_data)
    cpa_peaks, uncertianty_peaks = compute_peaks_asymmetry(signal_data)

    print(f"CP Asymmetry in Rare Decay Regions: {cpa_rare} ± {uncertainty_rare}")
    print(f"CP Asymmetry in Resonant Peaks: {cpa_peaks} ± {uncertianty_peaks}")
