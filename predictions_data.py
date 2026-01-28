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
sns.set_context('talk')
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

        optimal_cutoff = self.__find_optimal_cutoff(
            data['signal_probability'], signal_range=(0.6, 1.0))
        optimal_cutoff = 0.6
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
    def __cutoff_ratio(data_with_mass, cutoff_val, signal_range):
        survivors = data_with_mass[data_with_mass['BDT_prob'] >= cutoff_val]
        
        s_filter = (survivors['B invariant mass'] >= signal_range[0]) & \
                   (survivors['B invariant mass'] <= signal_range[1])
        S = len(survivors[s_filter])
        
        Stot = len(survivors[s_filter]) 
        
        if Stot <= 0: return 0
        return S / np.sqrt(Stot)

    @staticmethod
    @staticmethod
    def __find_optimal_cutoff(full_df, signal_range):
        # Assumes full_df has 'BDT_prob' and 'B invariant mass'
        cutoffs = np.linspace(0, 0.99, 100)
        weights = []

        for cutoff in cutoffs:
            # Pass the dataframe so we can check both BDT prob and Mass
            weight = BDT_Analysis.__cutoff_ratio(full_df, cutoff, signal_range)
            weights.append(weight)

        optimal_cutoff = cutoffs[np.argmax(weights)]
        
        # Plotting logic...
        plt.plot(cutoffs, weights, label=f'Peak significance: {max(weights):.2f}')
        plt.axvline(optimal_cutoff, color='red', label=f'Optimum: {optimal_cutoff:.2f}')
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
    ax[0].hist(raw_samesign[samesign['signal'] == 1]['B invariant mass'],
               bins=100, alpha=0.5, label='Classified Signal', color='green')
    ax[1].hist(raw_samesign[samesign['signal'] == 0]['B invariant mass'],
               bins=100, alpha=0.5, label='Classified Background', color='red')
    ax[0].set_ylabel(r'Candidates')

    ax[1].set_xlabel(r'B candidate mass / MeV/$c^2$')
    ax[1].set_ylabel(r'Candidates')
    ax[0].set_ylim(
        1, 3*max(raw_samesign[samesign['signal'] == 0]['B invariant mass']))
    ax[1].set_ylim(
        1, 3*max(raw_samesign[samesign['signal'] == 0]['B invariant mass']))
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[0].legend()
    ax[1].legend()
    ax[0].set_title('Same-Sign (Background) Data Classified by BDT Ensemble')
    plt.show()

    background_to_signal_ratio = samesign['signal'].value_counts(
    )[0] / (samesign['signal'].value_counts()[1] + samesign['signal'].value_counts()[0])

    print(
        f'Background to Signal Ratio in Same-Sign Data: {background_to_signal_ratio:.2f}')


# %% main

if __name__ == "__main__":
    analyse = BDT_Analysis(plot=True)
    analyse.save_cleaned_data()

    run_preds_samesign()


# %%
