"""
A module to load simulation data and evaluate pre-trained models on it to show misidentification bias.
22/01

NOTE: this code will need to be revisited. The simulation data does not contian the same features as the training data.
- this will probably need new models.
"""
import pickle
import glob
import config
import numpy as np
import os
import seaborn as sns
sns.set_style('darkgrid')
sns.set_context('paper')

dataset = '2011'
drop_cols = ['B invariant mass', 'dimuon-system invariant mass']


def __load_JpsiK_data():
    """
    Load the rapidsim JpsiK dataset.
    Returns:
    data : pd.DataFrame
        The simulation data
    """

    with open('datasets/rapidsim_JpsiK.pkl', 'rb') as f:
        data = pickle.load(f)

    # data.hist(column='B invariant mass', bins=100)
    # data.hist(column='dimuon-system invariant mass', bins=100)
    return data


def __load_Kmumu_data():
    """
    Load the rapidsim Kmumu dataset.
    Returns:
    data : pd.DataFrame
        The simulation data
    """

    with open('datasets/rapidsim_Kmumu.pkl', 'rb') as f:
        data = pickle.load(f)
    print('\n'.join(data.keys()))

    # data.hist(column='B invariant mass', bins=100)
    # data.hist(column='dimuon-system invariant mass', bins=100)
    return data


def run_model():
    data = __load_Kmumu_data()
    data = data.drop(columns=drop_cols)
    misclassified = []

    for model_file in glob.glob(f'models_{dataset}/*.pkl'):
        with open(model_file, 'rb') as f:
            model = pickle.load(f)

        predictions = model.predict_proba(data)[:, 1]

        avg_prob = np.mean(predictions)
        threshold = 0.5
        misidentified_count = np.sum(predictions > threshold)
        mis_rate = (misidentified_count / len(predictions)) * 100

        print(f"{os.path.basename(model_file):<20} | {avg_prob:.4f} | {mis_rate:.2f}%")

    print("-" * 65)
    print(
        f"Average Misidentification Rate: {np.mean(misclassified):.2f}% (+/- {np.std(misclassified):.2f})")
    return misclassified


if __name__ == "__main__":
    run_model()
