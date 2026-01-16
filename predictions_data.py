"""
Runs predictions for k folded BDT models on data and visualises the results.
Louis Liu 15/01
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import glob
sns.set_style('darkgrid')
sns.set_context('paper')

with open('data/filtered_data.pkl', 'rb') as f:
    seperation = pickle.load(f)


def predict_all():
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
        k = int(file.split('_')[-1].split('.')[0])

        data_k = seperation.dataset_k(k+1, drop_cols=['B invariant mass',
                                                      'dimuon-system invariant mass'])

        with open(file, 'rb') as f:
            model = pickle.load(f)

        predictions = model.predict_proba(data_k)[:, 1]
        plt.hist(predictions, bins=50, alpha=0.5, label=f'Fold {k}')

        data_k = pd.merge(data_k, pd.DataFrame(predictions, columns=['signal_probability']),
                          left_index=True, right_index=True)

        df_fold = pd.DataFrame(data_k)
        dataset.append(df_fold)

    plt.legend()
    plt.show()

    all_data = pd.concat(dataset, ignore_index=True)
    print(all_data.info())

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
    filtered = data_series[(data_series >= signal_range[0]) & (data_series <= signal_range[1])]
    num_sig = len(filtered)  # Count of events in signal range
    num_sigbck = len(data_series)  # Total events
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
    plt.yscale('log')
    plt.title('Finding Optimal Cutoff Probability')
    plt.show()
    
    return optimal_cutoff

def separate_data():
    all_data = predict_all()
    optimal_cutoff = find_optimal_cutoff(all_data['signal_probability'], signal_range=(0.6, 1.0))
    print(f'Optimal Cutoff Probability: {optimal_cutoff}')
    final_data = determine_signal(all_data, optimal_cutoff)
    # We histogram the final classified data
    plt.hist(final_data[final_data['signal'] == 1]['B invariant mass'], bins=100, alpha=0.5, label='Classified Signal')
    plt.hist(final_data[final_data['signal'] == 0]['B invariant mass'], bins=100, alpha=0.5, label='Classified Background')
    plt.xlabel(r'B candidate mass / MeV/$c^2$')
    plt.ylabel(r'Candidates / (23 MeV/$c^2$)')
    plt.yscale('log')
    plt.legend()
    plt.show()
    # Save final classified data
    final_data.to_csv('data/final_classified_data.csv', index=False)
    print('Final classified data saved to data/final_classified_data.csv')

if __name__ == "__main__":
    separate_data()
