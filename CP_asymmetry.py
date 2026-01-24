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


def compute_asymmetry(data, plot: bool = True):
    n_bins = 15
    counts, uncertainties, inv_mass = dimuon_binning.B_counts(data, n_bins)

    asy = []
    for bin_counts, count_uncertainty in zip(counts, uncertainties):
        B_plus, B_minus = bin_counts

        # print(f"Total counts in bin: {B_plus + B_minus}")
        B_plus_unc, B_minus_unc = count_uncertainty

        cp_asy, uncertainty = compute_b_asymmetry(
            B_plus, B_minus, B_plus_unc, B_minus_unc)
        asy.append((cp_asy, uncertainty))

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
        data.hist('dimuon-system invariant mass',
                  bins=100, alpha=0.5, ax=ax[0])
        ax[0].vlines(inv_mass, ymin=0, ymax=10e5, colors='red',
                     linestyles='dashed', label='Bin Edges')
        ax[0].set_yscale('log')
        ax[0].set_ylabel('Counts')
        ax[0].set_title('Dimuon Invariant Mass Distribution (Signal Data)')
        ax[1].errorbar(inv_mass, [v[0] for v in asy], yerr=[v[1] for v in asy],
                       fmt='o', color='tab:blue', ecolor='tab:blue', capsize=3, label='Resonance measurement')
        ax[1].set_xlabel('Invariant mass')
        ax[1].set_ylabel('CP Asymmetry')
        ax[1].axhline(0, color='red', linestyle='--')
        ax[1].set_ylim(-0.25, 0.25)
        plt.title('CP Asymmetry vs Dimuon Mass Bin (Signal Data)')
        plt.show()
    return asy


def asymmetry_calibrated(data, n_bins=10, plot: bool = False):
    """
    Computes binned CP asymmetry, shifts it by the resonance calibration,
    and plots subplots comparing the mass spectrum to corrected asymmetry.
    """
    delta_A, delta_A_unc = compute_combined_calibration(data, plot=plot)
    counts, uncertainties, inv_mass = dimuon_binning.B_counts(
        data, n_bins, plot=True)

    corrected_asy = []
    for bin_counts, bin_unc in zip(counts, uncertainties):
        B_plus, B_minus = bin_counts
        B_p_unc, B_m_unc = bin_unc

        raw_asy, raw_unc = compute_b_asymmetry(
            B_plus, B_minus, B_p_unc, B_m_unc)

        # Shift and propagate error
        val_corr = raw_asy - delta_A
        err_corr = math.sqrt(raw_unc**2 + delta_A_unc**2)
        corrected_asy.append((val_corr, err_corr))

    # 3. Plotting with Subplots
    if plot:
        # sharex=True ensures the mass scales are perfectly aligned
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                                       gridspec_kw={'height_ratios': [1, 1]})

        # --- Top Subplot: Original Dimuon Spectrum ---
        ax1.hist(data['dimuon-system invariant mass'], bins=200, color='gray',
                 alpha=0.6, log=True, label='Original Data')

        # Plot the bin boundaries to show where splicing occurred
        ax1.vlines(inv_mass, ymin=0, ymax=ax1.get_ylim()[1], colors='red',
                   linestyles='--', alpha=0.4, label='Analysis Bins')

        ax1.set_ylabel('Yield (Counts)')
        ax1.set_title('Dimuon Invariant Mass Spectrum (Log Scale)')
        ax1.legend(loc='upper right')

        # --- Bottom Subplot: Corrected CP Asymmetry ---
        y_vals = [v[0] for v in corrected_asy]
        y_errs = [v[1] for v in corrected_asy]

        ax2.errorbar(inv_mass, y_vals, yerr=y_errs,
                     fmt='o', color='black', ecolor='black', capsize=3,
                     label='Calibrated $A_{CP}$')

        # Reference line and calibration uncertainty band
        ax2.axhline(0, color='red', linestyle='--', linewidth=1.5,
                    label='SM Expectation ($A_{CP}=0$)')
        ax2.axhspan(-delta_A_unc, delta_A_unc, color='blue',
                    alpha=0.1, label='Calibration Precision')

        ax2.set_xlabel('Dimuon Invariant Mass [MeV]')
        ax2.set_ylabel('Corrected CP Asymmetry')
        ax2.set_ylim(-0.25, 0.25)
        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

    return corrected_asy, inv_mass


def rare_decay_asymmetry(data, n_bins=10, plot: bool = False):
    """
    Slices peaks out first, bins the remainder, and handles fit failures.
    Plots the spliced mass spectrum and the resulting CP asymmetry.
    """
    is_jpsi = (data['dimuon-system invariant mass'] >= 3000) & \
              (data['dimuon-system invariant mass'] <= 3200)
    is_psi2s = (data['dimuon-system invariant mass'] >= 3600) & \
               (data['dimuon-system invariant mass'] <= 3800)

    rare_data = data[~(is_jpsi | is_psi2s)]

    #  compute calibration shift
    delta_A, delta_A_unc = compute_combined_calibration(data, plot=False)

    #  bins on only rare data
    counts, uncertainties, inv_mass = dimuon_binning.B_counts(
        rare_data, n_bins, plot=True)

    corrected_asy = []
    rare_vals = []
    # weighted values for integration
    rare_weights = []
    valid_masses = []

    print(len(counts), "nbins for rare data.")
    for i in range(len(counts)):
        B_p, B_m = counts[i]
        B_p_u, B_m_u = uncertainties[i]

        # exception handling for failed fits (with unphysical uncertainties)
        # if B_p_u <= 0 or B_m_u <= 0 or math.isnan(B_p_u) or math.isinf(B_p_u):
        #     print(
        #         f"Skipping bin {i} at {inv_mass[i]:.0f} MeV: Fit did not converge.")
        #     continue

        raw_asy, raw_unc = compute_b_asymmetry(B_p, B_m, B_p_u, B_m_u)

        val_corr = raw_asy - delta_A
        err_corr = math.sqrt(raw_unc**2 + delta_A_unc**2)

        corrected_asy.append((val_corr, err_corr))
        valid_masses.append(inv_mass[i])

        w = 1.0 / (err_corr**2)
        rare_vals.append(val_corr * w)
        rare_weights.append(w)

    # 4. Final Integration Check
    if not rare_weights:
        print("Error: No rare bins survived the fit. Try reducing n_bins.")
        return 0.0, 0.0, [], []

    integrated_asy = sum(rare_vals) / sum(rare_weights)
    integrated_unc = math.sqrt(1.0 / sum(rare_weights))

    print(f"\n=== Spliced Rare Decay Results ===")
    print(f"Integrated Rare A_cp: {integrated_asy:.5f} ± {integrated_unc:.5f}")

    if plot:
        # Create a two-panel vertical plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # --- Top Plot: Spliced Histogram ---
        ax1.hist(rare_data['dimuon-system invariant mass'], bins=150,
                 color='steelblue', alpha=0.6, log=True, label='Spliced Data (Rare Only)')

        # Show where the bins are
        ax1.vlines(inv_mass, ymin=0, ymax=10e5, colors='red',
                   linestyles='--', alpha=0.3, label='Rare Bin Edges')

        ax1.set_ylabel('Yield (Log Scale)')
        ax1.set_title('Dimuon Invariant Mass (Resonances Removed)')
        ax1.legend(loc='upper right')

        # --- Bottom Plot: Corrected Asymmetry ---
        print("plotting corrected asymmetry...")
        y_points = [v[0] for v in corrected_asy]
        y_errors = [v[1] for v in corrected_asy]

        ax2.errorbar(valid_masses, y_points, yerr=y_errors,
                     fmt='ko', capsize=3, label='Calibrated $A_{CP}$')

        # Reference lines
        ax2.axhline(0, color='red', linestyle='--', alpha=0.6,
                    label='SM Expectation ($A_{CP}=0$)')
        ax2.axhspan(-delta_A_unc, delta_A_unc, color='blue',
                    alpha=0.1, label='Calibration Precision')

        # Integrated value line
        ax2.axhline(integrated_asy, color='green', linestyle=':', linewidth=2,
                    label=f'Global Rare Average: {integrated_asy:.4f}')

        ax2.set_ylabel('Corrected CP Asymmetry')
        ax2.set_xlabel('Dimuon Invariant Mass [MeV]')
        ax2.set_ylim(-0.25, 0.25)
        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

    return integrated_asy, integrated_unc, corrected_asy, valid_masses


# %% asymmetry calibration helper
def compute_combined_calibration(data, plot: bool = False):
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

    counts, uncertainties, _ = dimuon_binning.B_counts(
        combined_peak_data, n_bins=1)
    #  get counts for single bin

    # Calculate the calibration values
    B_plus, B_minus = counts[0]
    B_p_unc, B_m_unc = uncertainties[0]
    delta_A, delta_A_unc = compute_b_asymmetry(
        B_plus, B_minus, B_p_unc, B_m_unc)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.hist(data['dimuon-system invariant mass'], bins=200, alpha=0.5,
                 color='gray', log=True, label='Mass Distribution')
        splice_points = [3000, 3200, 3600, 3800]
        plt.vlines(splice_points, ymin=0, ymax=10e6, colors='red',
                   linestyles='dashed', alpha=0.8, label='Resonance Splicing')

        plt.ylabel('Counts (Log Scale)')
        plt.xlabel('Dimuon Invariant Mass [MeV]')
        plt.title('Calibration Mass Spectrum')
        plt.show()

    B_plus, B_minus = counts[0]
    B_p_unc, B_m_unc = uncertainties[0]

    delta_A, delta_A_unc = compute_b_asymmetry(
        B_plus, B_minus, B_p_unc, B_m_unc)

    return delta_A, delta_A_unc


# %% Main execution block

def detector_asymmetry():
    """
    Computes the detector asymmetry using magnet polarity reversal data.
    returns: float
        The detector asymmetry value.
    """

    mag_up, mag_down = __load_cleaned_mag_data()

    mag_up_asy = rare_decay_asymmetry(mag_up, plot=False)
    mag_down_asy = rare_decay_asymmetry(mag_down, plot=False)

    detector_asymmetry = 0.5 * (mag_up_asy - mag_down_asy)
    return detector_asymmetry


if __name__ == "__main__":
    signal_data = __load_signal_data()
    # cal_asy, mass_bins = asymmetry_calibrated(
    #     signal_data, n_bins=3, plot=False)

    acp_rare, acp_rare_unc, corrected_asy, mass_bins = rare_decay_asymmetry(
        signal_data, n_bins=3, plot=True)
