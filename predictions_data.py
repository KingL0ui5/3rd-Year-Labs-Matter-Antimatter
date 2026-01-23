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
import config
import glob
from scipy.optimize import curve_fit
sns.set_style('darkgrid')
sns.set_context('paper')


def __load_sim_data():
    """
    Load the rapidsim dataset for Kmumu.
    Returns:
    data : pd.DataFrame
        The rapidsim Kmumu dataset.
    """
    with open('datasets/rapidsim_Kmumu.pkl', 'rb') as infile:
        data = pickle.load(infile)

    data.hist(column='B invariant mass', bins=500)
    plt.xlim(5100, 5500)
    return data

# %% prediction class


class BDT_Analysis:
    def __init__(self, dataset=config.dataset):
        k = config.k
        with open(f'data/filtered_data_{dataset}.pkl', 'rb') as f:
            self._seperation = pickle.load(f)

        models = [None]*k
        for file in glob.glob(f'models_{dataset}/xgboost_model_*.pkl'):
            model_k = int(os.path.basename(file).split('_')[-1].split('.')[0])
            with open(file, 'rb') as infile:
                model = pickle.load(infile)
                models[model_k] = model

        self._models = models
        self._dataset_name = dataset

        data = self.__classify_data(hist=True)
        self._cleaned_data = self.__background_fit_cleaning(data)

    # - - - - - - - - - - - - - - - - - - run predictions and classify - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def __run_predictions(self):
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
        for k, model in enumerate(self._models):
            data_k = self._seperation.dataset_k(
                k+1, drop_cols=config.drop_cols)
            predictions = model.predict_proba(data_k)[:, 1]
            plt.hist(predictions, bins=50, alpha=0.5, label=f'Fold {k}')

            indexed_data_k = self._seperation.dataset_k(k+1)

            df_fold = pd.merge(indexed_data_k, pd.DataFrame(predictions, columns=['signal_probability']),
                               left_index=True, right_index=True)

            dataset.append(df_fold)

        plt.legend()
        plt.show()

        all_data = pd.concat(dataset, ignore_index=True)
        return all_data

    def __classify_data(self, feature='B invariant mass', hist: bool = False):
        data = self.__run_predictions()

        optimal_cutoff = self.__find_optimal_cutoff(
            data['signal_probability'], signal_range=(0.6, 1.0))
        print(f'Optimal Cutoff Probability: {optimal_cutoff}')
        classified_data = self.__determine_signal(data, optimal_cutoff)

        # We histogram the final classified data
        if hist:
            plt.hist(classified_data[classified_data['signal'] == 1][feature],
                     bins=100, alpha=0.5, label='Classified Signal')
            plt.hist(classified_data[classified_data['signal'] == 0][feature],
                     bins=100, alpha=0.5, label='Classified Background')
            plt.xlabel(r'B candidate mass / MeV/$c^2$')
            plt.ylabel(r'Candidates / (23 MeV/$c^2$)')
            plt.yscale('log')
            plt.legend()
            plt.show()

        return classified_data

    @staticmethod
    def __crystal_ball(x, x0, sigma, alpha, n, N):
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

    @staticmethod
    def __total_fit_func(x, x0, sigma, alpha, n, N, a, b, c):
        # Signal + Background
        return BDT_Analysis.__crystal_ball(x, x0, sigma, alpha, n, N) + (a * np.exp(b * (x - 5400)) + c)

    def __background_fit_cleaning(self, data):
        data = data[data['signal'] == 1].copy()
        data = data[(data['B invariant mass'] >= 5200) & (
            data['B invariant mass'] <= 6500)].reset_index(drop=True)

        data = data[(data['B invariant mass'] >= 5200) & (
            data['B invariant mass'] <= 6500)].reset_index(drop=True)
        hist, bin_edges = np.histogram(
            data['B invariant mass'], bins=200, range=(5200, 6500))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        p0 = [5280, 20, 1.5, 2, np.max(hist), np.max(hist)/20, -0.0005, 1]
        try:
            popt, _ = curve_fit(BDT_Analysis.__total_fit_func, bin_centers, hist, p0=p0,
                                bounds=([5250, 5, 0.1, 1, 0, 0, -0.01, 0],
                                        [5350, 50, 5.0, 10, np.inf, np.inf, 0, np.inf]))
        except RuntimeError:
            print("Fit failed; check initial parameters.")
            return data

        def get_bg_weight(mass):
            sig = BDT_Analysis.__crystal_ball(mass, *popt[0:5])
            bg = popt[5] * np.exp(popt[6] * (mass - 5400)) + popt[7]

            total = sig + bg
            if total <= 0:
                return 0

            weight = sig / total
            return np.clip(weight, 0, 1)

        data['event_weight'] = data['B invariant mass'].apply(get_bg_weight)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        x_plot = np.linspace(5200, 6500, 2000)

        ax1.hist(data['B invariant mass'], bins=100,
                 alpha=0.3, label='Data', color='gray')
        ax1.plot(x_plot, BDT_Analysis.__total_fit_func(x_plot, *popt),
                 'r-', label='Total Fit (CB + Exp)')
        ax1.plot(x_plot, popt[5] * np.exp(popt[6] * (x_plot - 5400)) + popt[7],
                 'b--', label='Background Component')
        ax1.set_title("Crystal Ball + Exponential Decay Fit")
        ax1.legend()
        ax1.set_yscale('log')

        ax2.hist(data['B invariant mass'], bins=100, weights=data['event_weight'],
                 alpha=0.7, color='tab:green', label='Signal-Weighted Data')
        ax2.set_title("Cleaned Data (Background Subtracted via Weights)")
        ax2.set_xlabel(r'B candidate mass / MeV/$c^2$')

        plt.tight_layout()
        plt.show()

        return data

    #  - - - - - - - - - - - - - - - - - - - - - - - - - signal cutoff methods - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @staticmethod
    def __determine_signal(data, threshold):
        """
        Determine signal events based on a probability threshold.
        Assigns 1 if it is signal, and 0 if it is background in the new 'signal column'

        Parameters
        ----------
        data: pd.DataFrame
            DataFrame containing 'signal_probability' column.
        threshold: float
            Probability threshold to classify signal events.
        """
        data['signal'] = (data['signal_probability'] >= threshold).astype(int)
        data.drop(columns=['signal_probability'], inplace=True)
        return data

    @staticmethod
    def __cutoff_ratio(data_series, signal_range):
        # Filter data_series to values within signal_range
        filtered = data_series[(data_series >= signal_range[0])
                               & (data_series <= signal_range[1])]
        num_sig = len(filtered)  # Count of events in signal range
        num_sigbck = len(data_series)  # Total events
        weight = num_sig / np.sqrt(num_sigbck)
        return weight

    @staticmethod
    def __find_optimal_cutoff(data_series, signal_range):
        cutoffs = np.linspace(0, 1, 100)
        weights = []

        for cutoff in cutoffs:
            filtered_probs = data_series[data_series >= cutoff]
            weight = BDT_Analysis.__cutoff_ratio(filtered_probs, signal_range)
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

 #  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - getters and saver - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def cleaned_data(self):
        return self._cleaned_data

    def save_cleaned_data(self):
        """
        Saves the cleaned data to a Pickle file.
        """
        filename = f'data/cleaned_data_{self._dataset_name}.pkl'
        self._cleaned_data.to_pickle(filename)
        print(f'Cleaned data saved to {filename}')


# %% k muon system analysis

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


def plot_resulting_dimuon_masses(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='dimuon-system invariant mass',
                 bins=100, color='purple')
    plt.axvline(3040, color='red', linestyle='--')
    plt.axvline(3200, color='red', linestyle='--', label='J/Psi')
    plt.axvline(3600, color='orange', linestyle='--')
    plt.axvline(3780, color='orange', linestyle='--', label='Psi(2S)')
    plt.xlabel(r'Dimuon System Invariant Mass [MeV/$c^2$]')
    plt.ylabel('Candidates')
    plt.yscale('log')
    plt.title('Dimuon System Invariant Mass Spectrum After Background Cleaning')
    plt.legend()
    plt.show()


def rare_decay_analysis(data):
    # We take out the peaks from the dimuon masses:
    signal_region_1 = (data['dimuon-system invariant mass'] < 3040) | (
        data['dimuon-system invariant mass'] > 3200)
    signal_region_2 = (data['dimuon-system invariant mass'] < 3600) | (
        data['dimuon-system invariant mass'] > 3780)
    rare_decay_data = data[signal_region_1 & signal_region_2]

    plt.figure(figsize=(10, 6))
    sns.histplot(data=rare_decay_data, x='dimuon-system invariant mass',
                 bins=100, color='green')
    plt.xlabel(r'Dimuon System Invariant Mass [MeV/$c^2$]')
    plt.ylabel('Candidates')
    plt.title('Dimuon System Invariant Mass Spectrum (Rare Decay Regions)')
    plt.show()
    return rare_decay_data


if __name__ == "__main__":
    __load_sim_data()
    analyse = BDT_Analysis()
    analyse.save_cleaned_data()
    cleaned_data = analyse.cleaned_data()

    analyze_k_mu_system(cleaned_data)
    plot_resulting_dimuon_masses(cleaned_data)
    rare_decay_data = rare_decay_analysis(cleaned_data)
    rare_decay_data.to_pickle('data/rare_decay_data.pkl')
