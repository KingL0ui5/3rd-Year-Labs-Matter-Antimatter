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
    total_B_plus = data[data['Kaon assumed particle type']
                        > 0]['event_weight'].sum()
    total_B_minus = data[data['Kaon assumed particle type']
                         < 0]['event_weight'].sum()
    return total_B, total_B_plus, total_B_minus


def compute_b_asymmetry(data):
    # count the B mesons using weights
    _, total_B_plus, total_B_minus = count_B_mesons(data)
    weighted_total = total_B_plus + total_B_minus
    if weighted_total == 0:
        return 0
    cp_asy = (total_B_plus - total_B_minus) / weighted_total
    return cp_asy


def compute_weighted_asymmetry_uncertainty(data):
    """
    Compute statistical uncertainty on CP asymmetry using weighted events.

    Formula: sigma_A = (2 * sqrt(N_minus^2 * var_plus + N_plus^2 * var_minus)) / (N_total^2)
    where var = sum(weights^2).
    """
    # Separate the weights for B+ and B-
    weights_plus = data[data['Kaon assumed particle type'] > 0]['event_weight']
    weights_minus = data[data['Kaon assumed particle type']
                         < 0]['event_weight']

    # Sum of weights (The 'Yield')
    N_plus = weights_plus.sum()
    N_minus = weights_minus.sum()
    N_total = N_plus + N_minus

    if N_total <= 0:
        return 0.0

    # Sum of weights squared (The 'Variance')
    var_plus = (weights_plus**2).sum()
    var_minus = (weights_minus**2).sum()

    # Propagation of error formula
    numerator = 2 * math.sqrt((N_minus**2 * var_plus) +
                              (N_plus**2 * var_minus))
    denominator = N_total**2

    uncertainty = numerator / denominator
    return uncertainty


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
    cpa_uncert = compute_weighted_asymmetry_uncertainty(peaks_data)

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
    cpa_uncert = compute_weighted_asymmetry_uncertainty(rare_data)

    return cpa, cpa_uncert


def compute_penguin_cp_symmetry(cpa_rare, cpa_peaks, cpa_rare_unc, cpa_peaks_unc):
    """
    Calculates the final physical CP asymmetry by subtracting the 
    instrumental bias (measured in resonant peaks) from the raw rare asymmetry.

    Parameters:
    cpa_rare (float): Raw CP asymmetry in the rare decay region.
    cpa_peaks (float): Raw CP asymmetry in the resonant control region.
    cpa_rare_unc (float): Uncertainty of the rare region asymmetry.
    cpa_peaks_unc (float): Uncertainty of the resonant region asymmetry.

    Returns:
    tuple: (final_cp_asymmetry, final_uncertainty)
    """
    final_cp_asymmetry = cpa_rare - cpa_peaks
    final_uncertainty = math.sqrt(cpa_rare_unc**2 + cpa_peaks_unc**2)

    return final_cp_asymmetry, final_uncertainty

# %% Main Execution


if __name__ == "__main__":
    signal_data = __load_signal_data()
    cpa_rare_val, uncertainty_rare_val = compute_rare_asymmetry(signal_data)
    cpa_peaks_val, uncertianty_peaks_val = compute_peaks_asymmetry(signal_data)
    final_cp_asy_val, final_cp_asy_unc_val = compute_penguin_cp_symmetry(cpa_rare_val, cpa_peaks_val,
                                                                         uncertainty_rare_val, uncertianty_peaks_val)

    print('================================================')
    print(
        f"CP Asymmetry in Rare Decay Regions: {cpa_rare_val} ± {uncertainty_rare_val}")
    print(
        f"CP Asymmetry in Resonant Peaks: {cpa_peaks_val} ± {uncertianty_peaks_val}")

    print('================================================')
    print(
        f'The rare case CP-violation value is: {final_cp_asy_val} $\pm$ {final_cp_asy_unc_val}.')

    mag_up_data, mag_down_data = __load_cleaned_mag_data()

    cpa_rare_mag_up, uncert_rare_mag_up = compute_rare_asymmetry(mag_up_data)
    cpa_peaks_mag_up, uncert_peaks_mag_up = compute_peaks_asymmetry(
        mag_up_data)
    final_cp_asy_mag_up, final_unc_mag_up = compute_penguin_cp_symmetry(
        cpa_rare_mag_up, cpa_peaks_mag_up, uncert_rare_mag_up, uncert_peaks_mag_up)
    print('================================================')
    print(
        f'Magnetic Up CP Asymmetry in Rare Decay Regions: {cpa_rare_mag_up} ± {uncert_rare_mag_up}')
    print(
        f'Magnetic Up CP Asymmetry in Resonant Peaks: {cpa_peaks_mag_up} ± {uncert_peaks_mag_up}')
    print(
        f'The Magnetic Up CP-violation value is: {final_cp_asy_mag_up} $\pm$ {final_unc_mag_up}.')

    cpa_rare_mag_down, uncert_rare_mag_down = compute_rare_asymmetry(
        mag_down_data)
    cpa_peaks_mag_down, uncert_peaks_mag_down = compute_peaks_asymmetry(
        mag_down_data)
    final_cp_asy_mag_down, final_unc_mag_down = compute_penguin_cp_symmetry(
        cpa_rare_mag_down, cpa_peaks_mag_down, uncert_rare_mag_down, uncert_peaks_mag_down)
    print('================================================')
    print(
        f'Magnetic Down CP Asymmetry in Rare Decay Regions: {cpa_rare_mag_down} ± {uncert_rare_mag_down}')
    print(
        f'Magnetic Down CP Asymmetry in Resonant Peaks: {cpa_peaks_mag_down} ± {uncert_peaks_mag_down}')
    print(
        f'The Magnetic Down CP-violation value is: {final_cp_asy_mag_down} $\pm$ {final_unc_mag_down}.')
