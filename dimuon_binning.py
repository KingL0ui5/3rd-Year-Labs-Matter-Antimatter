
import numpy as np
import matplotlib.pyplot as plt
import zfit

import pickle
import pandas as pd
import config

def bin_data(data, n_bins):
    sorted_data = data.sort_values('dimuon-system invariant mass', axis=0, ascending=True, ignore_index=True)
    binned_data = np.array_split(sorted_data, n_bins)
    return binned_data

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

def B_counts(data, n_bins):
    """
    Returns the number of B+ and B- mesons in the dataset.

    Parameters:
    data : pd.DataFrame
        The dataset in question.
    n_bins : int
        The number of bins to divide the data into based on dimuon-system invariant mass.
        
    Returns:
    tuple
        An array containing the counts of B+ and B- mesons for each bin [(count_B_plus, count_B_minus)].
    """
    B_plus, B_minus = split_Bs(data)
    binned_B_plus = bin_data(B_plus, n_bins=n_bins)
    binned_B_minus = bin_data(B_minus, n_bins=n_bins)

    boundary_inv_mass = np.linspace(0, max(data['dimuon-system invariant mass']), n_bins)


    counts = []
    uncertaintes = []
    for bin_p, bin_m in zip(binned_B_plus, binned_B_minus):
        _, B_plus, B_plus_uncertainty = background_fit_cleaning(bin_p)
        _, B_minus, B_minus_uncertainty = background_fit_cleaning(bin_m)
        counts.append((B_plus, B_minus))
        uncertaintes.append((B_plus_uncertainty, B_minus_uncertainty))
    return counts, uncertaintes, boundary_inv_mass
    
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

def background_fit_cleaning(data, plotting=False):
    data = data[data['signal'] == 1].copy()
    lower_obs, upper_obs = 5200, 6500
    data = data[(data['B invariant mass'] >= lower_obs) & 
                (data['B invariant mass'] <= upper_obs)].reset_index(drop=True)
    


    if data.empty or len(data) < 10: # Added minimum entry check
        return data, 0.0, 0.0

    # 2. Define Space
    obs = zfit.Space('B invariant mass', limits=(lower_obs, upper_obs))

    # 3. Define Yield Parameters with unique names
    u_id = f"{id(data)}_{np.random.randint(1000)}"
    sig_yield = zfit.Parameter(f'sig_yield_{u_id}', len(data)*0.5, 0, len(data)*1.5)
    bkg_yield = zfit.Parameter(f'bkg_yield_{u_id}', len(data)*0.5, 0, len(data)*1.5)

    # 4. Define Signal (Crystal Ball)
    # CORRECTION: In edge bins, these often need to be fixed to global values.
    # If the fit is unstable, consider setting floating=False for these.
    mu    = zfit.Parameter(f'mu_{u_id}', 5280, 5260, 5300)
    sigma = zfit.Parameter(f'sigma_{u_id}', 20, 10, 40)
    alpha = zfit.Parameter(f'alpha_{u_id}', 1.5, 0.5, 3.0)
    n     = zfit.Parameter(f'n_{u_id}', 2, 1, 5)

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
        result.hesse() # Try to get the Hesse errors
        
        # Extract yield value
        sig_val = float(sig_yield.numpy())
        
        # 7. Error Fallback Logic
        # Try Hesse first, then Minuit approx error, then Poisson
        if 'error' in result.params[sig_yield]:
            sig_err = result.params[sig_yield]['error']
        else:
            sig_err = np.sqrt(max(sig_val, 1.0))
            
        # If Hesse returned a "blown up" error (e.g., > yield or very large)
        if sig_err > sig_val or sig_err > 500: 
            sig_err = np.sqrt(max(sig_val, 1.0))
            
    except Exception:
        sig_val = float(sig_yield.numpy())
        # sig_err = np.sqrt(max(sig_val, 1.0))
    
    if plotting:
        plot_zfit_results(data, model, obs)

    sig_err = 0
    # 8. Calculate Event Weights
    probs_sig = signal_pdf.ext_pdf(z_data).numpy()
    probs_tot = model.ext_pdf(z_data).numpy()
    data['event_weight'] = np.clip(probs_sig / (probs_tot + 1e-10), 0, 1)

    return data, sig_val, sig_err

def plot_zfit_results(data, model, obs):
    lower, upper = obs.limit1d
    n_bins = 100
    x_plot = np.linspace(lower, upper, 1000)
    bin_width = (upper - lower) / n_bins

    plt.figure(figsize=(10, 6))

    # 1. Calculate Histogram data for the scatter plot
    counts, bin_edges = np.histogram(data['B invariant mass'], bins=n_bins, range=(lower, upper))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 2. Plot Data as Scatter (with Poisson errors)
    # We use errorbar with fmt='ko' (black circles)
    errors = np.sqrt(counts) # Poisson uncertainty
    plt.errorbar(bin_centers, counts, yerr=errors, fmt='ko', markersize=3, 
                 label='Data', capsize=0, elinewidth=1)

    # 3. Total Model Curve
    total_yield = model.get_yield().numpy()
    y_plot_tot = model.pdf(x_plot).numpy() * total_yield * bin_width
    plt.plot(x_plot, y_plot_tot, 'r-', lw=2.5, label='Total Fit (CB + Exp)')

    # 4. Individual Components
    y_sig = model.pdfs[0].pdf(x_plot).numpy() * model.pdfs[0].get_yield().numpy() * bin_width
    y_bkg = model.pdfs[1].pdf(x_plot).numpy() * model.pdfs[1].get_yield().numpy() * bin_width

    plt.plot(x_plot, y_sig, '--', color='tab:blue', label='Crystal Ball (Signal)')
    plt.plot(x_plot, y_bkg, '--', color='tab:orange', label='Exponential (Background)')

    # Formatting
    plt.yscale('log')
    plt.ylim(0.5, counts.max() * 5) # Adjusted for log scale visibility
    plt.xlabel(r'B candidate mass [MeV/$c^2$]')
    plt.ylabel(f'Events / ({bin_width:.1f} MeV)')
    plt.legend()
    plt.show()

def clean_signal(cleaned_data, plotting=True):
    final_signal_data = []
    yields = []
    yields_errors = []
    cleaned_data = __load_data()
    binned_data = bin_data(cleaned_data, n_bins=5)

    for n, data_bin in enumerate(binned_data):
        cleaned_df, sig_count_val, err = background_fit_cleaning(data_bin, plotting=plotting)
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
    
    hist, bin_edges = np.histogram(new_data['B invariant mass'], bins=200, range=(5175, 5400))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, 
                                gridspec_kw={'height_ratios': [3, 1]})

    ax1.hist(new_data['B invariant mass'], bins=200, range=(5175, 5400), 
            alpha=0.3, label='Experimental Data', color='black')
    ax1.plot(bin_centers, fit_on_new_data, 'r-', lw=2, 
            label=f'Simulated Signal (Scale: {scale_factor:.2f})')
    ax1.set_ylabel("Candidates")
    ax1.set_title('Overlay of the simulated signal with the experimental data')
    ax1.legend()

    ax2.errorbar(bin_centers, residuals, yerr=np.sqrt(hist), fmt='ko', markersize=2)
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_ylabel("Data - Model")
    ax2.set_xlabel(r'B candidate mass / MeV/$c^2$')
    ax2.set_title('Residuals between the two datasets')

    plt.tight_layout()
    plt.show()

    return residuals, reduced_chi_sq

#%% should not be used in main code

def __load_data():
    with open(f'data/cleaned_data_{config.dataset}.pkl', 'rb') as infile:
        data = pickle.load(infile)
    return data

if __name__ == "__main__":
    test_data = __load_data()
    signal, _, _ = clean_signal(test_data, plotting=True)
    overlay_and_calculate_residuals(signal)

# %%
