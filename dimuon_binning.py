
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


def return_B_counts(data, n_bins):
    """
    Returns the number of B+ and B- mesons in the dataset.
    
    Parameters:
    data : pd.DataFrame
        The dataset in question.
        
    Returns:
    tuple
        An array containing the counts of B+ and B- mesons for each bin [(count_B_plus, count_B_minus)].
    """
    B_plus, B_minus = split_Bs(data)
    binned_B_plus = bin_data(B_plus, n_bins=n_bins)
    binned_B_minus = bin_data(B_minus, n_bins=n_bins)

    counts = []
    for n, data in enumerate(binned_B_plus, binned_B_minus):
        _, B_plus = background_fit_cleaning(data)
        _, B_minus = background_fit_cleaning(data)
        counts.append((B_plus, B_minus))
    return counts


    # should run fitting functions on both B+ and B- datasets seperately
    pass
    
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

def background_fit_cleaning(data):
    # 1. Data Cleaning
    data = data[data['signal'] == 1].copy()
    lower_obs, upper_obs = 5200, 6500
    data = data[(data['B invariant mass'] >= lower_obs) & 
                (data['B invariant mass'] <= upper_obs)].reset_index(drop=True)

    if data.empty:
        return data, 0.0, 0.0

    # 2. Define Space
    obs = zfit.Space('B invariant mass', limits=(lower_obs, upper_obs))

    # 3. Define Yield Parameters
    sig_yield = zfit.Parameter(f'sig_yield_{id(data)}', len(data)*0.5, 0, len(data)*1.2)
    bkg_yield = zfit.Parameter(f'bkg_yield_{id(data)}', len(data)*0.5, 0, len(data)*1.2)

    # 4. Define Signal (Crystal Ball)
    mu    = zfit.Parameter(f'mu_{id(data)}', 5280, 5250, 5350)
    sigma = zfit.Parameter(f'sigma_{id(data)}', 20, 5, 50)
    alpha = zfit.Parameter(f'alpha_{id(data)}', 1.5, 0.1, 5.0)
    n     = zfit.Parameter(f'n_{id(data)}', 2, 0.5, 10)

    signal_pdf = zfit.pdf.CrystalBall(obs=obs, mu=mu, sigma=sigma, 
                                      alpha=alpha, n=n, extended=sig_yield)

    # 5. Define Background (Exponential)
    lam = zfit.Parameter(f'lam_{id(data)}', -0.001, -0.01, 0.0)
    background_pdf = zfit.pdf.Exponential(obs=obs, lam=lam, extended=bkg_yield)

    # 6. Build and Minimize Model
    model = zfit.pdf.SumPDF([signal_pdf, background_pdf])
    z_data = zfit.Data.from_pandas(df=data, obs=obs)

    nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=z_data)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll)

    # 7. Calculate HESSE Errors
    # Hesse assumes a parabolic likelihood and returns a single symmetric error
    result.hesse() 

    # 8. Extract Results
    sig_count = sig_yield
    # Hesse provides a single value, unlike the lower/upper pair in Minos
    sig_err = result.params[sig_yield]['hesse']['error']

    # 9. Calculate Event Weights & Plot
    probs_sig = signal_pdf.ext_pdf(z_data).numpy()
    probs_tot = model.ext_pdf(z_data).numpy()
    data['event_weight'] = np.clip(probs_sig / (probs_tot + 1e-10), 0, 1)

    plot_zfit_results(data, model, obs)

    return data, sig_count, sig_err

def plot_zfit_results(data, model, obs):
    lower, upper = obs.limit1d
    x_plot = np.linspace(lower, upper, 1000)
    bin_width = (upper - lower) / 100

    plt.figure(figsize=(10, 6))

    # Plot Data
    plt.hist(data['B invariant mass'], bins=100, alpha=0.3, label='Data', color='gray')

    # Total Model Curve (The sum of CB and Exponential)
    total_yield = model.get_yield().numpy()
    y_plot_tot = model.pdf(x_plot).numpy() * total_yield * bin_width
    plt.plot(x_plot, y_plot_tot, 'r-', lw=2.5, label='Total Fit (CB + Exp)')

    # Individual Components
    # pdfs[0] is CrystalBall, pdfs[1] is Exponential
    y_sig = model.pdfs[0].pdf(x_plot).numpy() * model.pdfs[0].get_yield().numpy() * bin_width
    y_bkg = model.pdfs[1].pdf(x_plot).numpy() * model.pdfs[1].get_yield().numpy() * bin_width

    plt.plot(x_plot, y_sig, '--', color='tab:blue', label='Crystal Ball (Signal)')
    plt.plot(x_plot, y_bkg, '--', color='tab:orange', label='Exponential (Background)')

    plt.yscale('log')
    plt.ylim(0.1, len(data) * 2) 
    plt.xlabel(r'B candidate mass [MeV/$c^2$]')
    plt.ylabel(f'Events / ({bin_width:.1f} MeV)')
    plt.legend()
    plt.show()

#%% should not be used in main code
def __load_data():
    with open(f'data/cleaned_data_{config.dataset}.pkl', 'rb') as infile:
        data = pickle.load(infile)
    return data

if __name__ == "__main__":
    yields = []
    yields_errors = []
    cleaned_data = __load_data()
    binned_data = bin_data(cleaned_data, n_bins=5)
    
    for n, data_bin in enumerate(binned_data):
        cleaned_df, sig_count_val, err = background_fit_cleaning(data_bin)
        final_val = float(sig_count_val)
        
        yields.append(final_val)
        yields_errors.append(err)

    print("Yields:", yields)
    print("Minos Errors:", yields_errors)

# %%
