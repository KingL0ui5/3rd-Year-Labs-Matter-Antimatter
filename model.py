"""
Using XGBoost to classify signal and background events.
12/01
"""
import os
import config
import filtered_data
from sklearn.metrics import roc_curve
import pandas as pd
import xgboost
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
sns.set_style('darkgrid')
sns.set_context('talk')

detail = True
models = []
k = config.k
dataset = config.dataset

# %% Load Data
seperation = filtered_data.seperate(k=k, dataset=dataset)

sig, bkg = seperation.data(drop_cols=config.drop_cols)

print('\n'.join(sig[0].keys()))
#  save the filtered object
with open(f'data/filtered_data_{dataset}.pkl', 'wb') as f:
    pickle.dump(seperation, f)

# %% Model Loop

for i in range(k):
    sig_k = sig[i]
    bkg_k = bkg[i]
    nSignalTotal_k = sig_k.shape[0]
    nBackgroundTotal_k = bkg_k.shape[0]

    model_k = xgboost.XGBClassifier(
        eval_metric='auc', early_stopping_rounds=50, n_estimators=1000, learning_rate=0.3, max_depth=10)

    # split data
    nSignalTrain_k = int(0.9*nSignalTotal_k)
    nSignalTest_k = nSignalTotal_k - nSignalTrain_k
    nBackgroundTrain_k = int(0.9*nBackgroundTotal_k)
    nBackgroundTest_k = nBackgroundTotal_k - nBackgroundTrain_k

    # x are the features and y are the classes
    x_train_k = pd.concat(
        [sig_k[:nSignalTrain_k], bkg_k[:nBackgroundTrain_k]], axis=0, ignore_index=True)
    y_train_k = [1]*nSignalTrain_k + [0]*nBackgroundTrain_k

    x_test_k = pd.concat(
        [sig_k[nSignalTrain_k:], bkg_k[nBackgroundTrain_k:]], axis=0, ignore_index=True)
    y_test_k = [1]*nSignalTest_k + [0]*nBackgroundTest_k

    #  train the model
    model_k.fit(x_train_k, y_train_k, eval_set=[(x_test_k, y_test_k)])

    #  Evaluate Model
    prediction_k = model_k.predict_proba(x_test_k)

    if detail:
        y_true = np.array(y_test_k)
        probs = np.array(prediction_k)
        if probs.ndim == 2:
            probs = probs[:, 1]

        # Create Figure 1: 1 Row, 2 Columns
        fig1, axes = plt.subplots(1, 2, figsize=(16, 6))

        # --- Left Plot: Signal vs Background Histogram ---
        ax_hist = axes[0]
        ax_hist.hist(probs[y_true == 1], bins=50, label='Signal', alpha=0.6)
        ax_hist.hist(probs[y_true == 0], bins=50,
                     label='Background', alpha=0.6)

        ax_hist.set_title(f'Signal vs Background (Model {i})')
        ax_hist.set_xlabel('Signal Probability')
        ax_hist.set_ylabel('log(Candidates)')
        ax_hist.set_yscale('log')
        ax_hist.legend()

        # --- Right Plot: ROC Curve ---
        ax_roc = axes[1]
        fpr, tpr, _ = roc_curve(y_true, probs)

        ax_roc.plot(fpr, tpr, label='ROC curve', linewidth=2)
        ax_roc.plot([0, 1], [0, 1], 'r--', label='Random')

        ax_roc.set_title(f'ROC Curve (Model {i})')
        ax_roc.set_xlabel('Background Contamination')
        ax_roc.set_ylabel('Signal Efficiency')
        ax_roc.legend()
        ax_roc.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        importance = model_k.get_booster().get_score(importance_type='gain')

        df = pd.DataFrame(list(importance.items()),
                          columns=['Feature', 'Importance'])
        df = df.sort_values(by='Importance', ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature',
                    hue='Feature', data=df, palette='viridis')

        plt.title(f'Top 10 Features by Gain model {i}', fontsize=15)
        plt.xlabel('Gain', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        plt.show()

    models.append(model_k)

# %% Save Models
os.makedirs(f'models_{dataset}', exist_ok=True)

for i, model in enumerate(models):
    with open(f'models_{dataset}/xgboost_model_{i}.pkl', 'wb') as f:
        pickle.dump(model, f)
