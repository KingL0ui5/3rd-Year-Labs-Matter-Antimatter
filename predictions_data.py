"""
Runs predictions for k folded BDT models on data and visualises the results.
Louis Liu 15/01
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import filtered_data
import glob


def predict_all():
    for file in glob.glob('models/xgboost_model_*.pkl'):
        k = int(file.split('_')[-1].split('.')[0])
        data_k = filtered_data.dataset_k(k+1)
        with open(file, 'rb') as f:
            model = pickle.load(f)

        predictions = model.predict_proba(data_k)[:, 1]
        plt.hist(predictions, bins=50, alpha=0.5, label=f'Fold {k}')
        plt.yscale('log')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    predict_all()
