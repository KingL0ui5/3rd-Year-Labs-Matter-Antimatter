"""
Runs predictions for k folded BDT models on data and visualises the results. 
Also finds the optimal cutoff probability to classify signal and background events, and filters partially 
reconstructed, peaking and misidentified backgrounds (by sideband subtraction). 

15/01 - created
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import glob
from scipy.optimize import curve_fit
sns.set_style('darkgrid')
sns.set_context('paper')

with open('data/filtered_data.pkl', 'rb') as f:
    seperation = pickle.load(f)

dataset_name = None


def predict_all():
    global dataset_name
    """
    Run predictions for all k folded models in 'models/' directory, using the seperation object stored in 'data/filtered_data.pkl'.
    Returns a DataFrame of the entire dataset combined with the predicted signal probabilities in column 'signal_probability'.

    Parameters
    ----------
    None
    -------
    Returns
    -------
    pd.DataFrame
        DataFrame containing the entire dataset with an additional column 'signal_probability' for predicted signal probabilities
    """
    dataset = []
    for file in glob.glob('models/xgboost_model_*.pkl'):
        filename = os.path.basename(file)
        name_no_ext = os.path.splitext(filename)[0]
        parts = name_no_ext.split('_')
        k = int(parts[-2])
        dataset_name = int(parts[-1])

        data_k = seperation.dataset_k(
            k+1, drop_cols=['B invariant mass', 'dimuon-system invariant mass'])

        with open(file, 'rb') as f:
            model = pickle.load(f)

        predictions = model.predict_proba(data_k)[:, 1]
        plt.hist(predictions, bins=50, alpha=0.5, label=f'Fold {k}')

        indexed_data_k = seperation.dataset_k(k+1)

        full_dataset = pd.merge(indexed_data_k, pd.DataFrame(predictions, columns=['signal_probability']),
                                left_index=True, right_index=True)

        df_fold = pd.DataFrame(full_dataset)
        dataset.append(df_fold)

    plt.legend()
    plt.show()

    all_data = pd.concat(dataset, ignore_index=True)
    return all_data


def determine_signal(data, threshold):
    """
    Determine signal events based on a probability threshold.
    Assigns 1 if it is signal, and 0 if it is background in the new 'signal column'

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing 'signal_probability' column.
    threshold : float
        Probability threshold to classify signal events.    
    """
    data['signal'] = (data['signal_probability'] >= threshold).astype(int)
    data.drop(columns=['signal_probability'], inplace=True)
    return data


def cutoff_ratio(data_series, signal_range):
    # Filter data_series to values within signal_range
    filtered = data_series[(data_series >= signal_range[0])
                           & (data_series <= signal_range[1])]
    num_sig = len(filtered)  # Count of events in signal range
    num_sigbck = len(data_series)  # Total events
    weight = num_sig / np.sqrt(num_sigbck)
    return weight


def find_optimal_cutoff(data_series, signal_range):
    cutoffs = np.linspace(0, 1, 100)
    weights = []

    for cutoff in cutoffs:
        filtered_probs = data_series[data_series >= cutoff]
        weight = cutoff_ratio(filtered_probs, signal_range)
        weights.append(weight)

    optimal_cutoff = cutoffs[np.argmax(weights)]
    plot_limit = 91

    plt.figure(figsize=(8, 5))
    plt.plot(cutoffs[:plot_limit], weights[:plot_limit],
             label='Significance Curve')
    if optimal_cutoff <= 0.9:
        plt.axvline(optimal_cutoff, color='red', linestyle='--',
                    label=f'Optimum: {optimal_cutoff:.2f}')
    plt.xlabel('Cutoff Probability')
    plt.ylabel(r'$S/\sqrt{S+B}$')
    plt.yscale('log')
    plt.title('Finding Optimal Cutoff Probability (up to 0.9)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

    return optimal_cutoff


def separate_data(feature='B invariant mass'):
    all_data = predict_all()
    optimal_cutoff = find_optimal_cutoff(
        all_data['signal_probability'], signal_range=(0.6, 1.0))
    print(f'Optimal Cutoff Probability: {optimal_cutoff}')
    final_data = determine_signal(all_data, optimal_cutoff)

    # We histogram the final classified data
    plt.hist(final_data[final_data['signal'] == 1][feature],
             bins=100, alpha=0.5, label='Classified Signal')
    plt.hist(final_data[final_data['signal'] == 0][feature],
             bins=100, alpha=0.5, label='Classified Background')
    plt.xlabel(r'B candidate mass / MeV/$c^2$')
    plt.ylabel(r'Candidates / (23 MeV/$c^2$)')
    plt.yscale('log')
    plt.legend()
    plt.show()

    # Save final classified data
    # final_data.to_csv('data/final_classified_data.csv', index=False)
    # print('Final classified data saved to data/final_classified_data.csv')
    return final_data


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
    sns.histplot(data=data[data['signal'] == 1], x='k_mu_invariant_mass',
                 bins=100, color='blue', label='Signal-like Candidates', kde=True)
    plt.xlabel(r'$K^+\mu^-$ Invariant Mass [MeV/$c^2$]')
    plt.ylabel('Candidates')
    plt.title('Invariant Mass Spectrum of $K^+\mu^-$ System')
    plt.legend()
    plt.show()


def background_fit_cleaning(data):
    data = data[data['signal'] == 1].copy()
    data = data[data['B invariant mass'] >= 5200].reset_index(drop=True)

    bg_mask = (data['B invariant mass'] > 5400) & (
        data['B invariant mass'] < 6500)
    background_data = data[bg_mask]

    hist_bg, bin_edges_bg = np.histogram(
        background_data['B invariant mass'], bins=50)
    bin_centers_bg = (bin_edges_bg[:-1] + bin_edges_bg[1:]) / 2

    x_offset = 5400

    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c

    popt, _ = curve_fit(exp_func, bin_centers_bg - x_offset,
                        hist_bg, p0=[hist_bg[0], -0.005, 0.5])

    ref_bins = 100
    ref_range = (5200, 6500)
    counts, bin_edges = np.histogram(
        data['B invariant mass'], bins=ref_bins, range=ref_range)
    bin_width = bin_edges[1] - bin_edges[0]
    scale_factor = bin_width / (bin_edges_bg[1] - bin_edges_bg[0])

    def get_bg_weight(mass):
        bg_level = exp_func(
            mass - x_offset, popt[0]*scale_factor, popt[1], popt[2]*scale_factor)
        bin_idx = np.digitize(mass, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, ref_bins - 1)
        total_in_bin = counts[bin_idx]

        if total_in_bin <= 0:
            return 0
        weight = (total_in_bin - bg_level) / total_in_bin
        return max(0, weight)

    data['event_weight'] = data['B invariant mass'].apply(get_bg_weight)

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    ax1.hist(data['B invariant mass'], bins=ref_bins,
             range=ref_range, alpha=0.5, label='Raw Data')
    x_plot = np.linspace(5200, 6500, 500)
    ax1.plot(x_plot, exp_func(x_plot - x_offset, popt[0]*scale_factor,
                              popt[1], popt[2]*scale_factor),
             color='red', label='Background Model')
    ax1.set_yscale('log')
    ax1.legend()

    ax2.hist(data['B invariant mass'], bins=ref_bins, range=ref_range,
             weights=data['event_weight'], alpha=0.7, color='tab:green',
             label='Weighted (Cleaned) Data')
    ax2.set_title("Full Event Data (Weighted)")
    ax2.set_xlabel(r'B candidate mass / MeV/$c^2$')
    plt.tight_layout()
    plt.show()

    return data


def plot_resulting_dimuon_masses(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='dimuon-system invariant mass',
                 bins=100, color='purple')
    plt.xlabel(r'Dimuon System Invariant Mass [MeV/$c^2$]')
    plt.ylabel('Candidates')
    plt.title('Dimuon System Invariant Mass Spectrum After Background Cleaning')
    plt.show()

def main():
    data = separate_data()
    cleaned_data = background_fit_cleaning(data)
    return cleaned_data


def main():
    final_data = separate_data()
    cleaned = background_fit_cleaning(final_data)
    return cleaned


def save_cleaned_data(dataset_name):
    """
    Saves the cleaned data to a Pickle file.
    """
    cleaned_data = main()
    filename = f'data/cleaned_data_{dataset_name}.pkl'
    cleaned_data.to_pickle(filename)
    print(f'Cleaned data saved to {filename}')


if __name__ == "__main__":
    data = separate_data()
    cleaned_data = background_fit_cleaning(data)
    analyze_k_mu_system(cleaned_data)
    plot_resulting_dimuon_masses(cleaned_data)
