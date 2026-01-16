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

def find_optimal_cutoff(data_series, signal_range):
    # Now each event has a probability [0,1] of being signal. 
    # We want to find the cutoff that maximises S/sqrt(S+B)
    cutoffs = np.linspace(0, 1, 100)
    weights = []
    for cutoff in cutoffs:
        filtered = data_series[data_series >= cutoff]
        weight = cutoff_ratio(filtered, signal_range)
        weights.append(weight)
    optimal_cutoff = cutoffs[np.argmax(weights)]
    plt.plot(cutoffs, weights)
    plt.xlabel('Cutoff Probability')
    plt.ylabel('S/sqrt(S+B)')
    plt.title('Finding Optimal Cutoff Probability')
    plt.show()

data_2011 = pickle.load(open('datasets/dataset_2011.pkl', 'rb'))

if __name__ == "__main__":
    predict_all()
