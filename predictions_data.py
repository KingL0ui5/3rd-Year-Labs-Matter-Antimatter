"""
Runs predictions for k folded BDT models on data and visualises the results.
Also finds the optimal cutoff probability to classify signal and background events, and filters partially
reconstructed, peaking and misidentified backgrounds (by sideband subtraction).

15/01 - created
"""
import filtered_data
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import config
import glob
sns.set_style('darkgrid')
sns.set_context('talk', font_scale=1.5)
dataset = config.dataset

# %% prediction class


class BDT_Analysis:
    def __init__(self, plot: bool = False):
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

        data = self.__classify_data(hist=plot)
        self._cleaned_data = data

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
                k, drop_cols=config.drop_cols)

            predictions = model.predict_proba(data_k)[:, 1]
            plt.hist(predictions, bins=50, alpha=0.5, label=f'Fold {k}')

            indexed_data_k = self._seperation.dataset_k(k)

            preds = pd.DataFrame(predictions, columns=[
                                 'signal_probability'], index=indexed_data_k.index)

            df_fold = pd.merge(indexed_data_k, preds,
                               left_index=True, right_index=True)

            dataset.append(df_fold)

        plt.legend()
        plt.show()

        all_data = pd.concat(dataset, ignore_index=True)
        return all_data

    def __classify_data(self, feature='B invariant mass', hist: bool = False):
        data = self.__run_predictions()

        # Define your signal range here (example: (0.6, 1.0) or use config if available)
        signal_range = (0.6, 1.0)
        optimal_cutoff = self.__find_optimal_cutoff(data['signal_probability'], signal_range)
        print(f'Optimal Cutoff Probability: {optimal_cutoff}')
        classified_data = self.__determine_signal(data, optimal_cutoff)

        # We histogram the final classified data
        if hist:
            # Plot original (raw) data filled at the back and the classified signal on top
            sig_col = '#4C72B0'  # muted blue for signal
            raw_col = '#F0AD4E'  # softer orange for raw data (filled)

            # Use consistent bin edges for both histograms
            bins = np.histogram_bin_edges(data[feature].dropna(), bins=80)

            # Raw data (filled, lower alpha) plotted first so it stays behind
            plt.hist(data[feature], bins=bins, label='Raw Data',
                    color=raw_col, edgecolor=raw_col, zorder=1)

            # Classified signal (filled) on top for contrast
            plt.hist(classified_data[classified_data['signal'] == 1][feature],
                    bins=bins, label='Classified Signal',
                    color=sig_col, edgecolor=sig_col, zorder=2)
            plt.xlabel(r'B candidate mass [MeV/$c^2$]')
            plt.ylabel(r'Candidates')
            plt.yscale('log')
            plt.title('BDT Ensemble Data Classified by Optimal Cutoff')
            plt.legend()
            plt.show()

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
        cutoffs = np.linspace(0.1, 0.9, 100)
        weights = []

        for cutoff in cutoffs:
            filtered_probs = data_series[data_series >= cutoff]
            weight = BDT_Analysis.__cutoff_ratio(filtered_probs, signal_range)
            weights.append(weight)

        optimal_cutoff = cutoffs[np.argmax(weights)]
        plot_limit = 89

        plt.figure(figsize=(8, 5))
        plt.plot(cutoffs[:plot_limit], weights[:plot_limit],
                 label='Figure of Merit Curve', color='black')
        plt.axvline(optimal_cutoff, color='red', linestyle='--',
                    label=f'Optimum: {optimal_cutoff:.2f}')
        plt.xlabel('Cutoff Probability')
        plt.ylabel(r'$S/\sqrt{S+B}$')
        plt.yscale('log')
        plt.title('FoM Optimal Cutoff Probability')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.show()

        return optimal_cutoff

 #  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - getters and saver - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def cleaned_data(self):
        return self._cleaned_data

    def save_cleaned_data(self, override=None):
        """
        Saves the cleaned data to a Pickle file.
        """
        if override is not None:
            data = override
        else:
            data = self._cleaned_data
        filename = f'data/cleaned_data_{dataset}.pkl'
        data.to_pickle(filename)
        print(f'Cleaned data saved to {filename}')


# %% testing

def run_preds_samesign():
    print('Running predictions on samesign data...')
    dataset = config.dataset

    raw_samesign = filtered_data.load_samesign()
    samesign = raw_samesign.copy()
    models = []
    preds = []

    samesign = samesign.drop(columns=config.drop_cols)
    for file in glob.glob(f'models_{dataset}/xgboost_model_*.pkl'):
        with open(file, 'rb') as infile:
            model = pickle.load(infile)
            models.append(model)

    for model in models:
        predictions = model.predict_proba(samesign)[:, 1]
        plt.hist(predictions, bins=50, alpha=0.5,
                 label=f'Model {models.index(model)}')
        preds.append(predictions)
        plt.yscale('log')

    plt.legend()
    plt.show()

    avg_signal_prob = np.mean(preds, axis=0)
    samesign['avg_signal_probability'] = avg_signal_prob

    samesign['signal'] = (
        samesign['avg_signal_probability'] >= 0.6).astype(int)

    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    # Plot histograms and get bin edges to calculate bin width
    n_signal, bins_signal, _ = ax[0].hist(
        raw_samesign[samesign['signal'] == 1]['B invariant mass'],
        bins=100, alpha=0.5, label='Classified Signal', color='green')
    n_bkg, bins_bkg, _ = ax[1].hist(
        raw_samesign[samesign['signal'] == 0]['B invariant mass'],
        bins=100, alpha=0.5, label='Classified Background', color='red')

    # Calculate bin width (should be the same for both)
    bin_width = bins_signal[1] - bins_signal[0]

    ax[1].set_xlabel(r'B candidate mass [MeV/$c^2$]')
    ax[0].set_ylim(
        1, 3*max(raw_samesign[samesign['signal'] == 0]['B invariant mass']))
    ax[1].set_ylim(
        1, 3*max(raw_samesign[samesign['signal'] == 0]['B invariant mass']))
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[0].legend()
    ax[1].legend()
    ax[0].set_title('Classified Same-Sign (Background) Data')
    # Communal y-label with bin width
    fig.text(0.04, 0.5, fr'Candidates / {bin_width:.0f} MeV/$c^2$',
             va='center', rotation='vertical', ha='center')
    plt.tight_layout(rect=[0.06, 0, 1, 1])
    plt.show()

    background_to_signal_ratio = samesign['signal'].value_counts(
    )[0] / (samesign['signal'].value_counts()[1] + samesign['signal'].value_counts()[0])

    print(
        f'Background to Signal Ratio in Same-Sign Data: {background_to_signal_ratio:.2f}')


# %% main

if __name__ == "__main__":
    # analyse = BDT_Analysis(plot=True)
    # analyse.save_cleaned_data()

    run_preds_samesign()


# %%
