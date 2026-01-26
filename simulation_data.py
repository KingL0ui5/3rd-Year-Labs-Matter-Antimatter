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
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
sns.set_style('darkgrid')
sns.set_context('talk')

dataset = '2011'
drop_cols = ['B invariant mass', 'dimuon-system invariant mass']

# %% check which features are important against those that exist in the simulation data.


def list_feature_importance_check(model, sim_df, top_n=None, plot=True):
    """
    Lists features by importance and checks if they exist in the simulation dataframe.

    Args:
        model: The trained XGBoost model.
        sim_df: The simulation dataframe to check against.
        top_n: Number of top features to return/plot. If None, returns all.
        plot: Boolean to generate a bar chart.

    Returns:
        pd.DataFrame: Sorted dataframe with columns ['Feature', 'Importance', 'In_Sim']
    """

    importance = model.get_booster().get_score(importance_type='gain')

    if not importance:
        print("No importance scores found. Ensure the model is trained.")
        return pd.DataFrame()

    # 2. Create DataFrame
    df_imp = pd.DataFrame(list(importance.items()),
                          columns=['Feature', 'Importance'])

    # 3. Check existence in Simulation Dataframe (1 = Yes, 0 = No)
    sim_columns = set(sim_df.columns)
    df_imp['In_Sim'] = df_imp['Feature'].apply(
        lambda x: 1 if x in sim_columns else 0)

    # 4. Sort by Importance (Descending)
    df_imp = df_imp.sort_values(
        by='Importance', ascending=False).reset_index(drop=True)

    # Filter for top N if requested
    if top_n:
        df_display = df_imp.head(top_n)
    else:
        df_display = df_imp

    print(f"--- Top {len(df_display)} Features by Gain ---")
    print(df_display.to_string(index=False))

    if plot:
        plt.figure(figsize=(10, len(df_display) * 0.4 + 2))  # Dynamic height

        colors = df_display['In_Sim'].map({1: 'forestgreen', 0: 'firebrick'})

        sns.barplot(x='Importance', y='Feature',
                    data=df_display, palette=colors.values)

        # Legend hack
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='forestgreen', label='Exists in Sim'),
                           Patch(facecolor='firebrick', label='Missing in Sim')]
        plt.legend(handles=legend_elements, loc='lower right')

        plt.title('Feature Importance & Availability Check', fontsize=15)
        plt.xlabel('Gain', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.show()

    return df_imp


def check_models(dataset):
    for glob_file in glob.glob(f'models_{dataset}/*.pkl'):
        with open(glob_file, 'rb') as f:
            model = pickle.load(f)

        print(f"\nEvaluating Model: {os.path.basename(glob_file)}")
        sim_data = __load_Kmumu_data()

        list_feature_importance_check(
            model, sim_data, top_n=20, plot=True)

# %% Load Simulation Data


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

# %% Run Models on Simulation Data


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
    check_models(dataset)
