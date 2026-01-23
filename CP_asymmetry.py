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


def compute_b_asymmetry(B_plus_count, B_minus_count):
    weighted_total = B_plus_count + B_minus_count
    if weighted_total == 0:
        return 0
    cp_asy = (B_plus_count - B_minus_count) / weighted_total
    return cp_asy


# def compute_weighted_asymmetry_uncertainty(data):
#     """
#     Compute statistical uncertainty on CP asymmetry using weighted events.

#     Formula: sigma_A = (2 * sqrt(N_minus^2 * var_plus + N_plus^2 * var_minus)) / (N_total^2)
#     where var = sum(weights^2).
#     """
#     # Separate the weights for B+ and B-
#     weights_plus = data[data['Kaon assumed particle type'] > 0]['event_weight']
#     weights_minus = data[data['Kaon assumed particle type']
#                          < 0]['event_weight']

#     # Sum of weights (The 'Yield')
#     N_plus = weights_plus.sum()
#     N_minus = weights_minus.sum()
#     N_total = N_plus + N_minus

#     if N_total <= 0:
#         return 0.0

#     # Sum of weights squared (The 'Variance')
#     var_plus = (weights_plus**2).sum()
#     var_minus = (weights_minus**2).sum()

#     # Propagation of error formula
#     numerator = 2 * math.sqrt((N_minus**2 * var_plus) +
#                               (N_plus**2 * var_minus))
#     denominator = N_total**2

#     uncertainty = numerator / denominator
#     return uncertainty


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
    # must return uncertainty
    pass
    # N_total, N_plus, N_minus = count_B_mesons(data)

    # if N_total <= 0:
    #     return 0.0
    # uncertainty = (2 * math.sqrt(N_plus * N_minus)) / (N_total**1.5)
    # return uncertainty

# %% Main Execution

def compute_asymmetry(data, plot: bool = False):
    counts, uncertaintes = dimuon_binning.B_counts(data, n_bins=5)

    asy = []
    for bin_counts, uncertainty in zip(counts, uncertaintes):
        N_plus, N_minus = bin_counts
        cp_asy = compute_b_asymmetry(N_plus, N_minus)

        uncertainty = compute_asymmetry_uncertainty(uncertainty)
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
    signal_data = __load_signal_data()
    compute_asymmetry(signal_data, plot=True)
    
    mag_up, mag_down = __load_cleaned_mag_data()
    compute_asymmetry(mag_up, plot=True)
    compute_asymmetry(mag_down, plot=True)  