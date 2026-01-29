
# Ensure circular import is handled or this function is defined
import pickle
import pandas as pd
import config
import os
import zfit
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style('darkgrid')
sns.set_context('talk', font_scale=1.5)


def bin_data(data, plot=False, one_bin=False):
    """
    Splits the data into specific q^2 bins, SKIPPING resonance gaps 
    and extending to the kinematic limit.
    """
    if 'q2_GeV' not in data.columns:
        data['q2_GeV'] = (data['dimuon-system invariant mass'] / 1000.0)**2

    mass_col = 'q2_GeV'

    if one_bin:
        # Full range covering the entire signal
        bin_ranges = [(0.05, 25.00)]
    else:
        # Define Discontinuous Ranges (Tuples)
        # We explicitly SKIP the gaps (8.68-10.09) and (12.86-14.18)
        bin_ranges = [
            (0.05, 2.00),
            (2.00, 4.30),
            (4.30, 8.68),
            (10.15, 12.80),
            (14.18, 16.00),
            (16.00, 18.00),
            (18.00, 22.00)
        ]

    binned_data = []
    ranges_MeV = []
    bin_labels = []

    for low, high in bin_ranges:
        mask = (data['q2_GeV'] >= low) & (data['q2_GeV'] < high)
        binned_data.append(data[mask].copy())

        # Convert q2 edges to MeV for plotting
        min_mev = np.sqrt(low) * 1000
        max_mev = np.sqrt(high) * 1000
        ranges_MeV.append((min_mev, max_mev))

        bin_labels.append(f"{low}-{high}")

    if plot:
        n_bins = len(binned_data)
        plt.figure(figsize=(12, 7))
        colors = plt.cm.nipy_spectral(np.linspace(0, 0.95, n_bins))
        plot_data = [df[mass_col] for df in binned_data]

        plt.hist(plot_data, bins=300, stacked=True, color=colors,
                 label=bin_labels, edgecolor='black', linewidth=0.5, alpha=0.8)

        plt.xlabel(r'$q^2$ [GeV$^2/c^4$]', fontsize=12)
        plt.ylabel('Events (Log Scale)', fontsize=12)
        plt.yscale('log')
        plt.title('Data Partitioned into Specific $q^2$ Bins', fontsize=14)
        plt.legend(title="$q^2$ Bins", frameon=True, loc='upper right')
        plt.tight_layout()
        plt.show()

    return binned_data, ranges_MeV


def split_Bs(data):
    """ 
    Splits the dataset into B+ and B- mesons based on the 'Kaon assumed particle type' column.
    Returns:
    total_B_plus : pd.DataFrame
        DataFrame containing all B+ mesons        
    total_B_minus : pd.DataFrame
        DataFrame containing all B- mesons
    """
    B_plus = data[data['Kaon assumed particle type']
                  > 0]
    B_minus = data[data['Kaon assumed particle type']
                   < 0]
    return B_plus, B_minus


def B_counts_simulation(data, plot=False, one_bin=False):
    """
    For SIMULATION (B+ Only): Splits into B+/B- and counts events.
    Since the dataset is B+ only, we expect B- counts to be zero.
    """
    B_plus, B_minus = split_Bs(data)

    # Use same binning logic as real data
    binned_B_plus, ranges_MeV = bin_data(B_plus, plot=False, one_bin=one_bin)

    # We still bin B_minus to maintain the parallel list structure,
    # even though we expect it to be empty.
    binned_B_minus, _ = bin_data(B_minus, plot=False, one_bin=one_bin)

    counts = []
    uncertainties = []
    bin_centers = []
    bin_half_widths = []

    print("--- Counting Simulation Events (B+ Only) ---")
    for i, (bin_p, bin_m, (m_min, m_max)) in enumerate(zip(binned_B_plus, binned_B_minus, ranges_MeV)):

        # Count events (B+ should have counts, B- should be 0)
        n_plus = len(bin_p)
        n_minus = len(bin_m)

        # Poisson error (Standard Deviation = sqrt(N))
        n_plus_unc = np.sqrt(n_plus) if n_plus > 0 else 0
        n_minus_unc = np.sqrt(n_minus) if n_minus > 0 else 0

        range_str = f"{m_min:.0f}-{m_max:.0f} MeV"
        # Print info, but don't warn about zero B- since that is expected
        print(f"Sim Bin {i} ({range_str}): B+={n_plus}, B-={n_minus}")

        counts.append((n_plus, n_minus))
        uncertainties.append((n_plus_unc, n_minus_unc))

        # Geometric center for plotting
        low_gev2 = (m_min / 1000)**2
        high_gev2 = (m_max / 1000)**2
        bin_centers.append((low_gev2 + high_gev2) / 2.0)
        bin_half_widths.append((high_gev2 - low_gev2) / 2.0)

    return counts, uncertainties, (np.array(bin_centers), np.array(bin_half_widths))


def B_counts(data, plot=False, one_bin=False):
    """
    Returns counts, uncertainties, and (centers, half_widths) for plotting.
    """
    B_plus, B_minus = split_Bs(data)

    # Get dataframes and ranges (tuples)
    binned_B_plus, ranges_MeV = bin_data(B_plus, plot=plot, one_bin=one_bin)
    binned_B_minus, _ = bin_data(B_minus, plot=plot, one_bin=one_bin)

    counts = []
    uncertainties = []

    bin_centers = []
    bin_half_widths = []

    # ZIP safely iterates over the Valid Bins only
    for i, (bin_p, bin_m, (m_min, m_max)) in enumerate(zip(binned_B_plus, binned_B_minus, ranges_MeV)):

        # Calculate center/width from the tuple directly
        center = (m_min + m_max) / 2.0
        half_width = (m_max - m_min) / 2.0

        range_str = f"{m_min:.0f}-{m_max:.0f} MeV"

        # Note: Ensure background_fit_cleaning is available here
        _, n_plus, n_plus_unc = background_fit_cleaning(
            bin_p,
            plotting=plot,
            plot_title=f'$B^+$ Fit (Bin {i}: {range_str})',
            fold=f'Bplus_bin{i}'
        )

        _, n_minus, n_minus_unc = background_fit_cleaning(
            bin_m,
            plotting=plot,
            plot_title=f'$B^-$ Fit (Bin {i}: {range_str})',
            fold=f'Bminus_bin{i}'
        )

        print(f"Bin {i} ({range_str}): B+={n_plus:.1f}, B-={n_minus:.1f}")

        # Basic safety check
        if np.isclose(n_plus, 0, atol=0.01) or np.isclose(n_minus, 0, atol=0.01):
            print(f"Skipping Bin {i} due to insufficient signal.")
            continue

        counts.append((n_plus, n_minus))
        uncertainties.append((n_plus_unc, n_minus_unc))

        bin_centers.append(center)
        bin_half_widths.append(half_width)

    return counts, uncertainties, (np.array(bin_centers), np.array(bin_half_widths))


def B_counts_GeV(data, plot=False, one_bin=False):
    """
    For REAL DATA: Splits into B+/B- and uses ZFIT to remove background.
    Returns calibrated signal yields.

    UPDATED: Returns bin_centers in q^2 (GeV^2), not Invariant Mass (MeV).
    """
    B_plus, B_minus = split_Bs(data)

    # Bin the dataframes
    binned_B_plus, ranges_MeV = bin_data(B_plus, plot=plot, one_bin=one_bin)
    binned_B_minus, _ = bin_data(B_minus, plot=plot, one_bin=one_bin)

    counts = []
    uncertainties = []
    bin_centers = []
    bin_half_widths = []

    print("--- Starting Fits for Real Data ---")
    for i, (bin_p, bin_m, (m_min, m_max)) in enumerate(zip(binned_B_plus, binned_B_minus, ranges_MeV)):

        range_str = f"{m_min:.0f}-{m_max:.0f} MeV"

        # Fit B+
        _, n_plus, n_plus_unc = background_fit_cleaning(
            bin_p, plotting=plot, plot_title=f'$B^+$ Fit (Bin {i}: {range_str})', fold=f'Bplus_bin{i}'
        )

        # Fit B-
        _, n_minus, n_minus_unc = background_fit_cleaning(
            bin_m, plotting=plot, plot_title=f'$B^-$ Fit (Bin {i}: {range_str})', fold=f'Bminus_bin{i}'
        )

        # Skip bin if fits failed or returned zero signal
        if np.isclose(n_plus, 0) or np.isclose(n_minus, 0):
            print(f"Skipping Bin {i} ({range_str}): Insufficient signal.")
            continue

        counts.append((n_plus, n_minus))
        uncertainties.append((n_plus_unc, n_minus_unc))

        # --- UNIT FIX IS HERE ---
        # Convert MeV Mass ranges back to q^2 in GeV^2
        # q (GeV) = m_min / 1000
        # q^2 (GeV^2) = (m_min / 1000)^2
        low_gev2 = (m_min / 1000.0)**2
        high_gev2 = (m_max / 1000.0)**2

        # Calculate center and width in GeV^2
        q2_center = (low_gev2 + high_gev2) / 2.0
        q2_width = (high_gev2 - low_gev2) / 2.0

        bin_centers.append(q2_center)
        bin_half_widths.append(q2_width)

    return counts, uncertainties, (np.array(bin_centers), np.array(bin_half_widths))


def crystal_ball(x, x0, sigma, alpha, n, N):
    """
    x0: mean, sigma: width, alpha: transition point, n: tail parameter, N: normalization
    """
    A = (n / np.abs(alpha))**n * np.exp(-np.abs(alpha)**2 / 2)
    B = n / np.abs(alpha) - np.abs(alpha)

    t = (x - x0) / sigma

    if alpha < 0:
        t = -t

    # Piecewise implementation
    condition = t > -np.abs(alpha)
    main_peak = N * np.exp(-t**2 / 2)
    tail = N * A * (B - t)**(-n)

    return np.where(condition, main_peak, tail)


def total_fit_func(x, x0, sigma, alpha, n, N, a, b, c):
    # Signal + Background
    return crystal_ball(x, x0, sigma, alpha, n, N) + (a * np.exp(b * (x - 5400)) + c)


def background_fit_cleaning(data, plotting=True, plot_title='Fit Result: B Invariant Mass Distribution', fold='all'):
    data = data[data['signal'] == 1].copy()
    lower_obs, upper_obs = 5200, 6500
    data = data[(data['B invariant mass'] >= lower_obs) &
                (data['B invariant mass'] <= upper_obs)].reset_index(drop=True)

    if data.empty or len(data) < 10:  # Added minimum entry check
        return data, 0.0, 0.0

    # 2. Define Space
    obs = zfit.Space('B invariant mass', limits=(lower_obs, upper_obs))

    # 3. Define Yield Parameters with unique names
    u_id = f"{id(data)}_{np.random.randint(1000)}"
    sig_yield = zfit.Parameter(
        f'sig_yield_{u_id}', len(data)*0.5, 0, len(data)*1.5)
    bkg_yield = zfit.Parameter(
        f'bkg_yield_{u_id}', len(data)*0.5, 0, len(data)*1.5)

    # 4. Define Signal (Crystal Ball)
    # CORRECTION: In edge bins, these often need to be fixed to global values.
    # If the fit is unstable, consider setting floating=False for these.
    mu = zfit.Parameter(f'mu_{u_id}', 5280, 5260, 5300)
    sigma = zfit.Parameter(f'sigma_{u_id}', 20, 10, 40)
    alpha = zfit.Parameter(f'alpha_{u_id}', 1.5, 0.5, 3.0)
    n = zfit.Parameter(f'n_{u_id}', 2, 1, 5)

    signal_pdf = zfit.pdf.CrystalBall(obs=obs, mu=mu, sigma=sigma,
                                      alpha=alpha, n=n, extended=sig_yield)

    # 5. Define Background (Exponential)
    lam = zfit.Parameter(f'lam_{u_id}', -0.001, -0.01, 0.001)
    background_pdf = zfit.pdf.Exponential(obs=obs, lam=lam, extended=bkg_yield)

    # 6. Build and Minimize
    model = zfit.pdf.SumPDF([signal_pdf, background_pdf])
    z_data = zfit.Data.from_pandas(df=data, obs=obs)

    nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=z_data)
    minimizer = zfit.minimize.Minuit()

    try:
        result = minimizer.minimize(nll)
        result.hesse()
        sig_val = float(sig_yield.numpy())

        # Capture the actual error from the fit!
        if sig_yield in result.params:
            sig_err = result.params[sig_yield].get('error', np.sqrt(sig_val))
        else:
            sig_err = np.sqrt(max(sig_val, 1.0))

    except Exception as e:
        print(f"Fit failed: {e}")
        sig_val = 0.0
        sig_err = 0.0

    if plotting:
        # We only plot if the data isn't empty and the model exists
        plot_zfit_results(data, model, obs, plot_title=plot_title,
                          fold=fold, b_counts=sig_val)

    # 8. Calculate Event Weights
    probs_sig = signal_pdf.ext_pdf(z_data).numpy()
    probs_tot = model.ext_pdf(z_data).numpy()
    data['event_weight'] = np.clip(probs_sig / (probs_tot + 1e-10), 0, 1)

    return data, sig_val, sig_err


def plot_zfit_results(data, model, obs, b_counts, log_scale=False, plot_title='Fit Result: B Invariant Mass Distribution', fold='all'):
    lower, upper = obs.limit1d
    n_bins = 100
    x_plot = np.linspace(lower, upper, 1000)
    bin_width = (upper - lower) / n_bins

    # Create two subplots: Main plot (ratio 4) and Pull plot (ratio 1)
    fig, (ax_main, ax_pull) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                           gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.05})

    # Calculate Histogram data
    counts, bin_edges = np.histogram(data['B invariant mass'], bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    y_err = np.sqrt(counts)

    # --- MAIN PLOT ---
    # Background & Signal
    y_bkg = model.pdfs[1].pdf(x_plot).numpy() * \
        model.pdfs[1].get_yield().numpy() * bin_width
    ax_main.plot(x_plot, y_bkg, color='forestgreen',
                 lw=2, label='Background (Exp)', zorder=1)

    y_sig = model.pdfs[0].pdf(x_plot).numpy() * \
        model.pdfs[0].get_yield().numpy() * bin_width
    ax_main.plot(x_plot, y_sig, color='royalblue', lw=2,
                 ls='--', label='Signal (CB)', zorder=2)

    # Total Fit
    total_yield = model.get_yield().numpy()
    y_plot_tot = model.pdf(x_plot).numpy() * total_yield * bin_width
    ax_main.plot(x_plot, y_plot_tot, color='crimson',
                 lw=3.5, label='Total Fit', zorder=3)

    # Data Points (Zero-suppressed)
    mask = counts > 0
    ax_main.errorbar(bin_centers[mask], counts[mask], yerr=y_err[mask], xerr=bin_width/2, fmt='ko',
                     markersize=6.5, capsize=0, elinewidth=1.5, label='Data', zorder=10)

    if b_counts > 0:
        textstr = f'Yield: {b_counts:.0f}'
        props = dict(boxstyle='round', facecolor='lightgray',
                     alpha=0.5, edgecolor='gray')
        ax_main.text(0.98, 0.88, textstr,
                     transform=ax_main.transAxes,
                     verticalalignment='top', horizontalalignment='right', bbox=props)

    # --- PULL PLOT ---
    # 1. Calculate the fit value at each bin center
    fit_counts = model.pdf(bin_centers).numpy() * total_yield * bin_width

    # 2. Calculate Pulls: (Data - Fit) / Error
    # We only calculate pulls where we have data to avoid dividing by zero
    pulls = (counts[mask] - fit_counts[mask]) / y_err[mask]

    ax_pull.errorbar(bin_centers[mask], pulls, yerr=np.ones_like(
        pulls), fmt='ko', markersize=4)
    ax_pull.axhline(0, color='crimson', lw=1)
    ax_pull.fill_between(bin_centers[mask], -1,
                         1, color='gray', alpha=0.2)  # 1-sigma band
    ax_pull.fill_between(bin_centers[mask], -2,
                         2, color='gray', alpha=0.1)  # 2-sigma band

    # Formatting
    ax_main.set_title(plot_title, pad=15)
    ax_main.set_ylabel(f'Events / ({bin_width:.1f} MeV/$c^2$)')
    ax_main.legend(frameon=True)

    ax_pull.set_xlabel(r'B candidate mass [MeV/$c^2$]')
    ax_pull.set_ylabel('Pull')
    ax_pull.set_ylim(-5, 5)  # Pulls usually live between -3 and 3
    ax_pull.set_yticks([-3, 0, 3])

    for ax in [ax_main, ax_pull]:
        ax.tick_params(axis='both', which='major', labelsize=14)

    if log_scale:
        ax_main.set_yscale('log')
        ax_main.set_ylim(0.1, counts.max() * 5)

    plt.tight_layout()
    plt.savefig(f'figs/fit_result_{fold}.png', dpi=300)
    plt.show()


def clean_signal(cleaned_data, plotting=True):
    final_signal_data = []
    yields = []
    yields_errors = []
    cleaned_data = __load_data()
    binned_data = bin_data(cleaned_data, n_bins=1)

    for n, data_bin in enumerate(binned_data):
        cleaned_df, sig_count_val, err = background_fit_cleaning(
            data_bin, plotting=plotting)
        final_val = float(sig_count_val)

        yields.append(final_val)
        yields_errors.append(err)
        final_signal_data.append(cleaned_df)

    print("Yields:", yields)
    print("Errors:", yields_errors)

    full_cleaned_df = pd.concat(final_signal_data, ignore_index=True)
    print(f"Total signal-enhanced events: {len(full_cleaned_df)}")
    # 1. Define the correct columns based on your list
    mass_col = 'B invariant mass'
    id_col = 'Kaon assumed particle type'

    # 2. Separate B+ and B- using the Kaon ID
    # K+ (321) -> B+ | K- (-321) -> B-
    b_plus = full_cleaned_df[full_cleaned_df[id_col] == 321]
    b_minus = full_cleaned_df[full_cleaned_df[id_col] == -321]

    # 3. Plotting
    plt.figure(figsize=(10, 6))

    # We use the 'event_weight' column from your list to get the actual signal counts
    plt.hist(b_plus[mass_col], bins=80, range=(5100, 5600), weights=b_plus['event_weight'],
             histtype='step', label=r'$B^+$ ($K^+$ candidate)', color='blue', linewidth=1.5)

    plt.hist(b_minus[mass_col], bins=80, range=(5100, 5600), weights=b_minus['event_weight'],
             histtype='step', label=r'$B^-$ ($K^-$ candidate)', color='red', linewidth=1.5)

    plt.xlabel(r'$B$ Invariant Mass [MeV/$c^2$]')
    plt.ylabel('Weighted Candidates / Bin')
    plt.title('Separated $B^+$ and $B^-$ Invariant Mass (Background Subtracted)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    # 4. Final Counts
    # Note: Use sum of weights for the "true" signal count if background was fitted
    yield_plus = b_plus['event_weight'].sum()
    yield_minus = b_minus['event_weight'].sum()

    print(f"--- Summary ---")
    print(f"B+ Yield (weighted): {yield_plus:.2f}")
    print(f"B- Yield (weighted): {yield_minus:.2f}")
    print(f"Total Yield:         {yield_plus + yield_minus:.2f}")

    return full_cleaned_df, b_plus, b_minus


def overlay_and_calculate_residuals(new_data, params_path='data/popt_crystal_ball.npy'):
    """
    Loads saved Crystal Ball parameters, scales the amplitude to the new data,
    overlays them, and calculates residuals.
    """
    popt_saved = np.load(params_path)

    hist, bin_edges = np.histogram(
        new_data['B invariant mass'], bins=200, range=(5175, 5400))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    data_integral = np.sum(hist)
    model_vals = crystal_ball(bin_centers, *popt_saved)
    model_integral = np.sum(model_vals) * bin_width
    print(
        f"Integral (total yield) in range: Data = {data_integral}, Model = {model_integral:.2f}")

    scale_factor = np.max(hist) / popt_saved[4]
    popt_scaled = popt_saved.copy()
    popt_scaled[4] = popt_saved[4] * scale_factor

    fit_on_new_data = crystal_ball(bin_centers, *popt_scaled)

    residuals = hist - fit_on_new_data

    mask = hist > 0
    chi_sq = np.sum(((hist[mask] - fit_on_new_data[mask])**2) / hist[mask])
    degrees_of_freedom = len(hist[mask]) - len(popt_saved)
    reduced_chi_sq = chi_sq / degrees_of_freedom

    print(f"Reduced Chi-Squared: {reduced_chi_sq:.4f}")

    # --- Normalized shape comparison plot (top), overlay (middle), residuals (bottom) ---
    hist_norm = hist / np.max(hist) if np.max(hist) > 0 else hist
    model_norm = model_vals / \
        np.max(model_vals) if np.max(model_vals) > 0 else model_vals

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 14), sharex=True)

    # Top: Normalized shape comparison
    ax0.step(bin_centers, hist_norm, where='mid',
             label='Data (normalized)', color='black')
    ax0.plot(bin_centers, model_norm, label='Model (normalized)',
             color='red', linestyle='--')
    ax0.set_ylabel('Normalized to Max')
    ax0.set_title('Normalized Shape Comparison: Data vs Model')
    ax0.legend()
    ax0.set_xlim(5200, 5400)
    ax0.grid(True, alpha=0.3)

    # Bottom: Residuals
    ax1.errorbar(bin_centers, residuals, yerr=np.sqrt(
        hist), fmt='ko', markersize=2)
    ax1.axhline(0, color='red', linestyle='--')
    ax1.set_ylabel("Data - Model")
    ax1.set_xlabel(r'B candidate mass / MeV/$c^2$')
    ax1.set_title('Residuals between the two datasets')
    ax1.set_xlim(5200, 5400)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return residuals, reduced_chi_sq

# %% should not be used in main code


def __load_data():
    with open(f'data/cleaned_data_{config.dataset}.pkl', 'rb') as infile:
        data = pickle.load(infile)
    return data


if __name__ == "__main__":
    test_data = __load_data()
    signal, _, _ = clean_signal(test_data, plotting=True)
    overlay_and_calculate_residuals(signal)

# %%
