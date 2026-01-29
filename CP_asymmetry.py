"""
Count the number of B mesons in the dataset, and seperate them into B+ and B- mesons.
20/01 - created
"""

import pandas as pd
import pickle
import math
import config
import dimuon_binning
import filtered_data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_style('darkgrid')
sns.set_context('talk', font_scale=1.5)
# %% Data Loading Functions


def __load_signal_data(dataset):
    """
    Load the cleaned signal 2011 dataset after background fitting and weighting.
    Returns:
    pd.DataFrame
        The cleaned dataset with event weights applied.
    """
    with open(f'data/cleaned_data_{dataset}.pkl', 'rb') as f:
        cleaned_data = pickle.load(f)

        raw_data = filtered_data.load_dataset(dataset=dataset)

        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        cleaned_data.hist('dimuon-system invariant mass', bins=200, ax=ax[0])
        raw_data.hist('dimuon-system invariant mass',
                      bins=200, alpha=0.5, ax=ax[0])
        plt.legend(['Cleaned Signal Data', 'Raw Data'])
        ax[0].set_yscale('log')
        ax[0].set_title(
            'Dimuon Invariant Mass Distribution: Raw vs Cleaned Signal Data')
        ax[0].set_ylabel('Counts (Log Scale)')
        ax[0].set_xlabel('Dimuon Invariant Mass [MeV]')

        cleaned_data.hist('B invariant mass', bins=200, ax=ax[1])
        raw_data.hist('B invariant mass', bins=200, alpha=0.5, ax=ax[1])
        plt.legend(['Cleaned Signal Data', 'Raw Data'])
        ax[1].set_yscale('log')
        ax[1].set_title(
            'B Invariant Mass Distribution: Raw vs Cleaned Signal Data')
        ax[1].set_ylabel('Counts (Log Scale)')
        ax[1].set_xlabel('B Invariant Mass [MeV]')
        plt.show()

    return cleaned_data


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
    counts, uncertainties, inv_mass = dimuon_binning.B_counts(
        data, n_bins, one_bin=True)

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
        data, one_bin=True, plot=True)

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
        ax2.vlines()

        ax2.set_xlabel('Dimuon Invariant Mass [MeV]')
        ax2.set_ylabel('Corrected CP Asymmetry')
        ax2.set_ylim(-0.25, 0.25)
        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

    return corrected_asy, inv_mass


def rare_decay_asymmetry(plot: bool = False):
    """
    Computes CP asymmetry separately for Magnet Up and Magnet Down data,
    calculates the Detector Bias, averages the results to cancel systematics, 
    and plots the final asymmetry.
    """

    raw_up = __load_signal_data('up')
    raw_down = __load_signal_data('down')

    polarity_results = {}

    integrated_polarity_stats = {}

    all_rare_data = []

    for label, df in [('up', raw_up), ('down', raw_down)]:
        print(f"\n--- Processing Magnet {label.upper()} ---")

        jpsi_low, jpsi_high = 2946, 3176
        psi2s_low, psi2s_high = 3586, 3776

        is_jpsi = (df['dimuon-system invariant mass'] >= jpsi_low) & \
                  (df['dimuon-system invariant mass'] <= jpsi_high)
        is_psi2s = (df['dimuon-system invariant mass'] >= psi2s_low) & \
                   (df['dimuon-system invariant mass'] <= psi2s_high)

        partially_reconstructed = df['B invariant mass'] < 5170

        rare_df = df[~(is_jpsi | is_psi2s | partially_reconstructed)].copy()

        analyze_k_mu_system(rare_df)
        rare_df = filter_misidentified_background(rare_df)
        analyze_k_mu_system(rare_df)

        all_rare_data.append(rare_df)

        delta_A, delta_A_unc = compute_combined_calibration(df, plot=plot)
        print(
            f"Calibration shift for {label}: {delta_A:.5f} ± {delta_A_unc:.5f}")

        counts, uncertainties, (inv_mass, x_widths) = dimuon_binning.B_counts(
            rare_df, one_bin=False, plot=plot
        )

        acp_values = []
        acp_errors = []

        for i in range(len(counts)):
            B_p, B_m = counts[i]
            B_p_u, B_m_u = uncertainties[i]

            raw_asy, raw_unc = compute_b_asymmetry(B_p, B_m, B_p_u, B_m_u)

            val_corr = raw_asy - delta_A

            err_corr = math.sqrt(raw_unc**2 + delta_A_unc**2)

            acp_values.append(val_corr)
            acp_errors.append(err_corr)

        vals_arr = np.array(acp_values)
        errs_arr = np.array(acp_errors)

        polarity_results[label] = {
            'vals': vals_arr,
            'errs': errs_arr,
            'mass': inv_mass,
            'widths': x_widths
        }

        weights_pol = 1.0 / (errs_arr**2)
        weights_pol[np.isinf(weights_pol)] = 0

        if np.sum(weights_pol) > 0:
            int_val_pol = np.sum(vals_arr * weights_pol) / np.sum(weights_pol)
            int_err_pol = math.sqrt(1.0 / np.sum(weights_pol))
        else:
            int_val_pol, int_err_pol = 0.0, 0.0

        integrated_polarity_stats[label] = (int_val_pol, int_err_pol)

    a_up, a_up_err = integrated_polarity_stats['up']
    a_down, a_down_err = integrated_polarity_stats['down']

    detector_bias = 0.5 * (a_up - a_down)
    bias_uncertainty = 0.5 * math.sqrt(a_up_err**2 + a_down_err**2)

    print("\n" + "="*45)
    print("DETECTOR BIAS ANALYSIS (POST-CALIBRATION)")
    print("="*45)
    print(f"Calibrated A_up:   {a_up:+.5f} ± {a_up_err:.5f}")
    print(f"Calibrated A_down: {a_down:+.5f} ± {a_down_err:.5f}")
    print("-" * 45)
    print(
        f"ISOLATED DETECTOR BIAS: {detector_bias:+.5f} ± {bias_uncertainty:.5f}")

    if abs(detector_bias) > 2 * bias_uncertainty:
        print("NOTE: Significant residual detector bias detected (>2 sigma).")
    else:
        print("RESULT: Detector bias is compatible with zero within 2 sigma.")
    print("="*45)

    res_up = polarity_results['up']
    res_down = polarity_results['down']

    if len(res_up['vals']) != len(res_down['vals']):
        raise ValueError("Bin mismatch between Up and Down datasets.")

    avg_asy = (res_up['vals'] + res_down['vals']) / 2.0
    avg_err = 0.5 * np.sqrt(res_up['errs']**2 + res_down['errs']**2)

    valid_masses = res_up['mass']
    valid_widths = res_up['widths']

    weights = 1.0 / (avg_err**2)
    weights[np.isinf(weights)] = 0

    if np.sum(weights) == 0:
        print("Error: Total weight is zero.")
        return 0.0, 0.0, [], []

    integrated_asy = np.sum(avg_asy * weights) / np.sum(weights)
    integrated_unc = math.sqrt(1.0 / np.sum(weights))

    print(f"\n=== Final Combined Results (Up+Down) ===")
    print(f"Integrated Rare A_cp: {integrated_asy:.5f} ± {integrated_unc:.5f}")

    if plot:
        total_rare_data = pd.concat(all_rare_data)

        fig, (ax1) = plt.subplots(figsize=(10, 12))

        ax1.hist(total_rare_data['dimuon-system invariant mass'], bins=150,
                 color='steelblue', alpha=0.6, log=True, label='Total Spliced Data')

        bin_edges = (valid_masses[0] - valid_widths[0],
                     valid_masses[-1] + valid_widths[-1])
        ax1.axvline(bin_edges[0], color='red',
                    linestyle='--', alpha=0.3, label='Bin Edges')
        ax1.axvline(bin_edges[1], color='red', linestyle='--', alpha=0.3)

        ax1.set_ylabel('Yield (Log Scale)')
        ax1.set_title('Dimuon Invariant Mass (Resonances Removed)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.errorbar(valid_masses, avg_asy,
                     xerr=valid_widths,
                     yerr=avg_err,
                     fmt='ko', capsize=3)

        ax2.axhline(0, color='red', linestyle='--', alpha=0.6)
        ax2.axhline(integrated_asy, color='green', linestyle=':', linewidth=2,
                    label=f'Int. Avg: {integrated_asy:.4f}')

        ax2.axvspan(jpsi_low, jpsi_high, color='gray',
                    alpha=0.3)
        ax2.axvspan(psi2s_low, psi2s_high, color='gray', alpha=0.3)

        ax2.set_ylabel(r'Avg $A_{CP}$')
        ax2.set_xlabel(r'Dimuon Invariant Mass [MeV/c$^2$]')
        ax2.set_ylim(-0.25, 0.25)
        ax2.legend(loc='upper left', ncol=2)
        ax2.set_title(r'Magnet-Averaged CP Asymmetry vs $q^2$')
        ax2.grid(True, alpha=0.3)

        plt.subplots_adjust(hspace=0.3)
        plt.show()

    corrected_asy_list = list(zip(avg_asy, avg_err))

    return integrated_asy, integrated_unc, corrected_asy_list, valid_masses, detector_bias, bias_uncertainty

# %% asymmetry calibration helper


def compute_combined_calibration(data, plot: bool = False):
    """
    Concatenates J/psi and psi(2s) data to compute a single 
    high-precision calibration asymmetry.
    """
    # Define resonance windows
    is_jpsi = (data['dimuon-system invariant mass'] >= 2900) & \
              (data['dimuon-system invariant mass'] <= 3200)
    is_psi2s = (data['dimuon-system invariant mass'] >= 3600) & \
               (data['dimuon-system invariant mass'] <= 3800)

    combined_peak_data = data[is_jpsi | is_psi2s]

    counts, uncertainties, _ = dimuon_binning.B_counts(
        combined_peak_data, one_bin=True, plot=plot)
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
        splice_points = [2900, 3200, 3600, 3800]
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

# %% K+mu-


def filter_misidentified_background(data):
    """
    CRITICAL FIX 2: Calculates the Mass under the 'Swapped Hypothesis'.
    Treats the Kaon as a Muon to find misidentified J/psi events.
    """
    M_MU = 105.658   # Muon mass in MeV
    M_JPSI = 3096.9  # J/psi mass in MeV

    K_px = data['Kaon 4-momentum x component']
    K_py = data['Kaon 4-momentum y component']
    K_pz = data['Kaon 4-momentum z component']

    Mu_px = data['Opposite-sign muon 4-momentum x component']
    Mu_py = data['Opposite-sign muon 4-momentum y component']
    Mu_pz = data['Opposite-sign muon 4-momentum z component']

    P2_K = K_px**2 + K_py**2 + K_pz**2
    E_K_swapped = np.sqrt(P2_K + M_MU**2)

    # Calculate Energy of the real Muon (using Muon mass)
    P2_Mu = Mu_px**2 + Mu_py**2 + Mu_pz**2
    E_Mu = np.sqrt(P2_Mu + M_MU**2)

    # Calculate the Invariant Mass of the pair
    E_tot = E_K_swapped + E_Mu
    Px_tot = K_px + Mu_px
    Py_tot = K_py + Mu_py
    Pz_tot = K_pz + Mu_pz
    P2_tot = Px_tot**2 + Py_tot**2 + Pz_tot**2

    # Avoid negative inputs to sqrt due to precision issues
    mass_squared = np.maximum(E_tot**2 - P2_tot, 0)
    swapped_mass = np.sqrt(mass_squared)

    # Define the Veto (Remove events near J/psi mass)
    #: Veto if K-mu mass is consistent with J/psi
    is_ghost_jpsi = (np.abs(swapped_mass - M_JPSI) < 60)

    plt.figure(figsize=(10, 5))
    plt.hist(swapped_mass, bins=100, range=(
        2500, 3500), label='Before Veto', alpha=0.5)
    plt.hist(swapped_mass[~is_ghost_jpsi], bins=100, range=(
        2500, 3500), label='After Veto', alpha=0.5)
    plt.axvline(M_JPSI, color='red', linestyle='--', label='J/psi Mass')
    plt.title("Swapped Mass Hypothesis (Kaon treated as Muon)")
    plt.xlabel(r'$K^+\mu^-$ Invariant Mass [MeV/$c^2$]')
    plt.ylabel('Candidates')
    plt.legend()
    plt.show()

    # Return filtered data
    return data[~is_ghost_jpsi]


def analyze_k_mu_system(data):
    """
    Calculates K+mu- invariant mass and visualizes the spectrum.
    """
    e_sum = data['Kaon 4-momentum energy component'] + \
        data['Opposite-sign muon 4-momentum energy component']

    px_sum = data['Kaon 4-momentum x component'] + \
        data['Opposite-sign muon 4-momentum x component']

    py_sum = data['Kaon 4-momentum y component'] + \
        data['Opposite-sign muon 4-momentum y component']

    pz_sum = data['Kaon 4-momentum z component'] + \
        data['Opposite-sign muon 4-momentum z component']

    p_squared = px_sum**2 + py_sum**2 + pz_sum**2
    k_mu_mass = np.sqrt(np.maximum(e_sum**2 - p_squared, 0))

    data['k_mu_invariant_mass'] = k_mu_mass

    plt.figure(figsize=(10, 6))
    bins = 100
    sns.histplot(data=data[data['signal'] == 1], x='k_mu_invariant_mass',
                 bins=bins, color='blue', label='Signal-like Candidates', kde=True)
    plt.xlabel(r'$K^+\mu^-$ Invariant Mass [MeV/$c^2$]')

    # compute bin width for display in ylabel
    km = data['k_mu_invariant_mass'].dropna()
    if len(km) > 0:
        bin_width = (km.max() - km.min()) / bins
    else:
        bin_width = 0.0

    plt.ylabel(f'Candidates / ({bin_width:.1f} MeV/$c^2$)')
    plt.title('Invariant Mass Spectrum of $K^+\mu^-$ System')
    plt.legend()
    plt.show()

# %%

# %% Main execution block


def compare_simulation_to_data():
    """
    Plots a publication-quality comparison of Real Data vs Simulation.
    - Simulation: plotted as a filled histogram (since it represents the 'shape').
    - Real Data: plotted as black points with error bars.
    - Ratio Plot: attached below with no gap.
    """
    # 1. Load and Count Data
    # ----------------------
    real_data = __load_signal_data(config.dataset)
    # Note: B_counts returns centers in GeV^2 now (after your unit fix)
    real_counts, real_unc, (real_centers, real_widths) = dimuon_binning.B_counts_GeV(
        real_data, plot=False)

    real_yields = np.array([p + m for p, m in real_counts])
    real_errors = np.array([np.sqrt(pu**2 + mu**2) for pu, mu in real_unc])

    with open('datasets/rapidsim_Kmumu.pkl', 'rb')as f:
        sim_data = pickle.load(f)
    sim_counts, sim_unc, (sim_centers, sim_widths) = dimuon_binning.B_counts_simulation(
        sim_data, plot=False)

    sim_yields = np.array([p + m for p, m in sim_counts])
    sim_errors = np.array([np.sqrt(pu**2 + mu**2) for pu, mu in sim_unc])

    # 2. Normalize Simulation
    # -----------------------
    if np.sum(sim_yields) > 0:
        scale_factor = np.sum(real_yields) / np.sum(sim_yields)
    else:
        scale_factor = 1.0

    sim_yields_scaled = sim_yields * scale_factor
    sim_errors_scaled = sim_errors * scale_factor

    # 3. Setup Plotting Layout (No Gap)
    # ---------------------------------
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # 4. Top Panel: Yield Comparison
    # ------------------------------

    # A. Plot Simulation as a "Filled Histogram" (using bar)
    # We use bar width = 2 * half_width
    ax1.bar(sim_centers, sim_yields_scaled, width=2*sim_widths,
            color='none', alpha=0.3, edgecolor='black', linewidth=2,
            label=f'Simulation B+ (Scaled)')

    # Add a "hatched" fill for style (optional)
    ax1.bar(sim_centers, sim_yields_scaled, width=2*sim_widths,
            color='none', edgecolor='black', alpha=0.3, hatch='//')

    # B. Plot Real Data as Points
    ax1.errorbar(real_centers, real_yields, xerr=real_widths, yerr=real_errors,
                 fmt='ko', markersize=6, elinewidth=2, capsize=3,
                 label='Real Data (Total)')

    # Formatting Top Panel
    ax1.set_ylabel('Events / Bin', fontsize=22)
    ax1.set_title('Data vs. Simulation: Efficiency Comparison',
                  fontsize=24, pad=15)

    # Legend with Scientific Notation for Scale Factor
    # We construct a custom legend label to include the scale factor neatly
    handles, labels = ax1.get_legend_handles_labels()
    # Update the simulation label to include the scientific notation number
    labels[0] = f'Simulation B+ (Scale: {scale_factor:.2e})'
    ax1.legend(handles, labels, fontsize=16, frameon=True,
               fancybox=True, loc='upper right')

    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelbottom=False)  # Hide x-labels on top plot

    # 5. Bottom Panel: Ratio Plot
    # ---------------------------
    valid = sim_yields_scaled > 0
    ratio = np.zeros_like(real_yields)
    ratio_err = np.zeros_like(real_yields)

    ratio[valid] = real_yields[valid] / sim_yields_scaled[valid]

    # Error propagation: (dR/R)^2 = (dN_data/N_data)^2 + (dN_sim/N_sim)^2
    rel_err_data = real_errors[valid] / real_yields[valid]
    rel_err_sim = sim_errors_scaled[valid] / sim_yields_scaled[valid]
    ratio_err[valid] = ratio[valid] * np.sqrt(rel_err_data**2 + rel_err_sim**2)

    # Plot Ratio Points
    ax2.errorbar(real_centers[valid], ratio[valid], xerr=real_widths[valid], yerr=ratio_err[valid],
                 fmt='ko', markersize=5, elinewidth=1.5, capsize=0)

    # Reference Line at 1.0
    ax2.axhline(1.0, color='gray', linestyle='--', linewidth=2)

    # Shaded band for 10% deviation (visual guide)
    ax2.axhspan(0.9, 1.1, color='gray', alpha=0.15, label='$\pm$10% agreement')

    ax2.set_ylabel('Ratio\n(Data / Sim)', fontsize=18)
    ax2.set_xlabel(r'$q^2$ [GeV$^2/c^4$]', fontsize=22)

    # Limit ratio view to reasonable deviations
    ax2.set_ylim(0.5, 1.5)
    ax2.set_yticks([0.5, 0.75, 1.0, 1.25, 1.5])
    ax2.grid(True, alpha=0.5)

    plt.tight_layout()
    # Re-apply hspace adjustment after tight_layout
    plt.subplots_adjust(hspace=0.2)

    plt.show()


def plot_psi_2s(data):
    """
    Plots the dimuon invariant mass distribution around the psi(2s) resonance.
    """
    is_psi2s = data['dimuon-system invariant mass'].between(3600, 3750)

    psi2s_data = data[is_psi2s]
    psi2s_data.hist('B invariant mass', bins=100, figsize=(10, 6),
                    color='purple', alpha=0.7)
    plt.title('B Invariant Mass Distribution around $\psi(2S)$ Resonance')
    plt.xlabel('B Invariant Mass [MeV]')
    plt.ylabel('Counts')
    plt.show()


if __name__ == "__main__":

    rare_decay_asymmetry(
        plot=True)

    # compare_simulation_to_data()

    # detector_bias, bias_uncertainty = compute_detector_bias()
