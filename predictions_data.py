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

def predict_all():
    dataset = []
    for file in glob.glob('models/xgboost_model_*.pkl'):
        k = int(file.split('_')[-1].split('.')[0])

        data_k = seperation.dataset_k(k+1)

        with open(file, 'rb') as f:
            model = pickle.load(f)
        predictions = model.predict_proba(data_k)[:, 1]
        plt.hist(predictions, bins=50, alpha=0.5, label=f'Fold {k}')

        pd.merge(data_k, pd.DataFrame(predictions, columns=['signal_probability']),
                 left_index=True, right_index=True)

        df_fold = pd.DataFrame(data_k)
        dataset.append(df_fold)

    plt.legend()
    plt.show()

    all_data = pd.concat(dataset, ignore_index=True)

    return all_data


def cutoff_ratio(data_series, signal_range):
    num_sig = np.trapezoid(
        data_series[signal_range[0]:signal_range[1]], dx=0.01)
    num_sigbck = len(data_series)
    weight = num_sig / np.sqrt(num_sigbck)
    return weight


def find_optimal_cutoff(data):
    data_series = data['dimuon-system invariant mass']
    
    # 1. Create the histogram first to get counts and bin edges
    counts, bin_edges = np.histogram(data_series, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 2. Find local maxima in the counts
    peaks = []
    window = 3 
    for i in range(window, len(counts) - window):
        if counts[i] > max(counts[i-window:i]) and counts[i] > max(counts[i+1:i+window+1]):
            peaks.append((i, counts[i]))
    
    # Sort by height and take top 2
    peaks.sort(key=lambda x: x[1], reverse=True)
    top_peak_indices = sorted([p[0] for p in peaks[:2]])

    optimal_ranges_values = []
    
    # 3. Optimize range based on histogram counts
    # Significance = Signal / sqrt(Total) -> approx Counts / sqrt(Total Bins)
    for peak_idx in top_peak_indices:
        best_weight = -1
        best_bins = (peak_idx, peak_idx)
        
        # Search window in bins
        search_min = max(0, peak_idx - 10)
        search_max = min(len(counts), peak_idx + 10)
        
        for start in range(search_min, peak_idx + 1):
            for end in range(peak_idx + 1, search_max + 1):
                # Sum the counts in this bin range
                signal = np.sum(counts[start:end])
                # Simple significance metric (S/sqrt(S+B))
                # Note: For a real physics analysis, you'd estimate B from sidebands
                weight = signal / np.sqrt(signal + 1) 
                
                if weight > best_weight:
                    best_weight = weight
                    best_bins = (start, end)
        
        # Convert bin indices back to mass values (MeV/c^2)
        range_val = (bin_edges[best_bins[0]], bin_edges[best_bins[1]])
        optimal_ranges_values.append(range_val)
    
    # 4. Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(data_series, bins=100, alpha=0.5, label='dimuon-system invariant mass')
    
    colors = ['red', 'blue']
    for i, (val_start, val_end) in enumerate(optimal_ranges_values):
        plt.axvspan(val_start, val_end, color=colors[i], alpha=0.3, 
                    label=f'Peak {i+1} Range ({val_start:.0f}-{val_end:.0f})')
    
    plt.xlabel('dimuon-system invariant mass / MeV/c^2')
    plt.ylabel('Candidates')
    plt.legend()
    plt.show()

    return optimal_ranges_values


data_2011 = pickle.load(open('datasets/dataset_2011.pkl', 'rb'))

if __name__ == "__main__":
    optimal_cutoff = find_optimal_cutoff(data_2011)
    print(f'Optimal cutoff range indices: {optimal_cutoff}')
