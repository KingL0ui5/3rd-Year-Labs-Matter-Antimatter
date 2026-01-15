import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#  rethink: the predictions are on the training dataset not the whole dataset


def __load_data():
    with open('datasets/dataset_2012.pkl', 'rb') as infile:
        __data_2012 = pickle.load(infile)
    return __data_2012


def __load_model():
    import xgboost
    model = xgboost.XGBClassifier()
    model.load_model('data/xgboost_model.json')
    return


def binarize_predictions(threshold=0.5):
    predictions = __load_data()
    predictions['Predicted Class'] = (
        predictions['Signal Probability'] >= threshold).astype(int)
    return predictions


def plot_histogram():
    import filtered_data
    classes = binarize_predictions(0.5)['Predicted Class']
    data = filtered_data.raw_data()

    signal_data = data[(classes == 1).values]
    background_data = data[(classes == 0).values]
    plt.hist(signal_data['B invariant mass'], bins=100,
             alpha=0.5, label='Predicted Signal')
    plt.show()

    plt.hist(background_data['B invariant mass'], bins=100,
             alpha=0.5, label='Predicted Background')
    plt.xlabel(r'B candidate mass / MeV/$c^2$')
    plt.ylabel(r'Candidates / (23 MeV/$c^2$)')
    plt.legend()
    plt.show()


# %% Predict
model = __load_model()
