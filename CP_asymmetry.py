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

# We compute the CP symmetry for the peaks:

def compute_combined_calibration(data, plot: bool=True):
    """
    Concatenates J/psi and psi(2s) data to compute a single 
    high-precision calibration asymmetry.
    """
    # Define resonance windows
    is_jpsi = (data['dimuon-system invariant mass'] >= 3000) & \
              (data['dimuon-system invariant mass'] <= 3200)
    is_psi2s = (data['dimuon-system invariant mass'] >= 3600) & \
               (data['dimuon-system invariant mass'] <= 3800)
    
    combined_peak_data = data[is_jpsi | is_psi2s]
    
    # Get the integrated counts for the calibration
    counts, uncertainties, inv_mass = dimuon_binning.B_counts(combined_peak_data, n_bins=1)
    
    # Calculate the calibration values
    B_plus, B_minus = counts[0]
    B_p_unc, B_m_unc = uncertainties[0]
    delta_A, delta_A_unc = compute_b_asymmetry(B_plus, B_minus, B_p_unc, B_m_unc)

    if plot:
        plt.figure(figsize=(10, 6))

        # 1. Histogram on the main Y-axis
        plt.hist(data['dimuon-system invariant mass'], bins=200, alpha=0.5,
                 color='gray', log=True, label='Mass Distribution')

        # 2. Define the exact splice points for the vlines
        splice_points = [3000, 3200, 3600, 3800]

        # Draw the lines at the splice points
        plt.vlines(splice_points, ymin=0, ymax=10e6, colors='red', 
                   linestyles='dashed', alpha=0.8, label='Resonance Splicing')

        plt.ylabel('Counts (Log Scale)')
        plt.xlabel('Dimuon Invariant Mass [MeV]')
        plt.title('Combined Calibration Yield & Asymmetry')
        plt.show()

    if len(counts) > 0:
        B_plus, B_minus = counts[0]
        B_p_unc, B_m_unc = uncertainties[0]

        # Calculate the single global asymmetry
        delta_A, delta_A_unc = compute_b_asymmetry(B_plus, B_minus, B_p_unc, B_m_unc)

        print("=== Combined Calibration (J/psi + psi(2S)) ===")
        print(f"Total Calibration Mesons: {B_plus + B_minus:.0f}")
        print(f"Measured Detector Bias (delta A): {delta_A:.5f} ± {delta_A_unc:.5f}")

        return delta_A, delta_A_unc
    else:
        print("Error: No peak data found.")
        return 0.0, 0.0

def compute_asymmetry(data, plot: bool = True):
    n_bins = 15
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
        ax[1].errorbar(inv_mass, [v[0] for v in asy], yerr=[v[1] for v in asy], 
                    fmt='o', color='tab:blue', ecolor='tab:blue', capsize=3, label='Resonance measurement')
        ax[1].set_xlabel('Invariant mass')
        ax[1].set_ylabel('CP Asymmetry')
        ax[1].axhline(0, color='red', linestyle='--')
        ax[1].set_ylim(-0.25,0.25)
        plt.title('CP Asymmetry vs Dimuon Mass Bin (Signal Data)')
        plt.show()

    return asy

#%% Main execution.

if __name__ == "__main__":
    signal_data = __load_signal_data()
    #compute_asymmetry(signal_data, plot=True)

    #mag_up, mag_down = __load_cleaned_mag_data()
    #compute_asymmetry(mag_up, plot=True)
    #compute_asymmetry(mag_down, plot=True)

    peak_results = compute_combined_calibration(signal_data)
