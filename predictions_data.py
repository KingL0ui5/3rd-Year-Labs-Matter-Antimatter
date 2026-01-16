"""
Runs predictions for k folded BDT models on data and visualises the results.
Louis Liu 15/01
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import filtered_data
import glob
sns.set_style('darkgrid')
sns.set_context('paper')

with open('data/filtered_data.pkl', 'rb') as f:
    seperation = pickle.load(f)


def predict_all():
    predictions_all = pd.DataFrame()
    for file in glob.glob('models/xgboost_model_*.pkl'):
        k = int(file.split('_')[-1].split('.')[0])
        data_k = seperation.dataset_k(k+1)
        with open(file, 'rb') as f:
            model = pickle.load(f)

        predictions = model.predict_proba(data_k)[:, 1]
        plt.hist(predictions, bins=50, alpha=0.5, label=f'Fold {k}')
        plt.yscale('log')
        predictions_all = pd.concat(predictions_all)
    plt.legend()
    plt.show()

    return predictions


def cutoff_ratio(data_series, signal_range):
    num_sig = np.trapezoid(
        data_series[signal_range[0]:signal_range[1]], dx=0.01)
    num_sigbck = len(data_series)
    weight = num_sig / np.sqrt(num_sigbck)
    return weight


def find_optimal_cutoff(data):
    data_series = data['dimuon-system invariant mass']
    peaks, hist = filtered_data.get_dimuon_peaks(data)

    if len(peaks) == 0:
        search_range = range(0, len(data_series))
    else:
        peak_bin_width = 10
        min_start = max(0, peaks[0] - peak_bin_width)
        max_end = min(len(data_series), peaks[-1] + peak_bin_width)
        search_range = range(min_start, max_end)

    best_weight = 0
    best_range = (0, 0)
    for start in search_range:
        for end in range(start + 1, min(start + 50, len(data_series))):
            weight = cutoff_ratio(data_series, (start, end))
            if weight > best_weight:
                best_weight = weight
                best_range = (start, end)
    
    data_series.plot(kind='hist', bins=100, alpha=0.5)
    plt.axvspan(data_series.iloc[best_range[0]], data_series.iloc[best_range[1]], color='red', alpha=0.3,
                label='Optimal Cutoff Range')
    plt.xlabel('dimuon-system invariant mass / MeV/c^2')
    plt.ylabel('Candidates')
    plt.legend()
    plt.show()

    return best_range


data_2011 = pickle.load(open('datasets/dataset_2011.pkl', 'rb'))

if __name__ == "__main__":
    optimal_cutoff = find_optimal_cutoff(data_2011)
    print(f'Optimal cutoff range indices: {optimal_cutoff}')
