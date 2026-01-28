"""
Using XGBoost to classify signal and background events.
12/01
"""
import os
import config
import filtered_data
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, log_loss
from scipy.stats import ks_2samp
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

cv_metrics = {
    'auc': [],
    'accuracy': [],
    'log_loss': [],
    'ks_stat_sig': [],  # KS score for Signal
    'ks_pval_sig': [],
    'ks_stat_bkg': [],  # KS score for Background
    'ks_pval_bkg': []
}

# %% Load Data
seperation = filtered_data.seperate(k=k, dataset=dataset)
sig_indexed, bkg_indexed = seperation.data()

sig_indexed[0].hist('dimuon-system invariant mass', bins=100)
plt.yscale('log')
plt.title('Signal: Dimuon System Invariant Mass - Model 0')
plt.xlabel(r'Dimuon System Invariant Mass [MeV/c^2]')
plt.ylabel('log(Candidates)')
plt.show()

bkg_indexed[0].hist('B invariant mass', bins=100)
plt.yscale('log')
plt.title('Background: B Invariant Mass - Model 0')
plt.xlabel(r'B Invariant Mass [MeV/c^2]')
plt.ylabel('log(Candidates)')
plt.show()

sig, bkg = seperation.data(drop_cols=config.drop_cols)
print(len(sig[0]) * k)

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
        eval_metric='auc', early_stopping_rounds=50, n_estimators=1000, learning_rate=0.05, max_depth=4)

    # split data
    nSignalTrain_k = int(0.9*nSignalTotal_k)
    nSignalTest_k = nSignalTotal_k - nSignalTrain_k
    nBackgroundTrain_k = int(0.9*nBackgroundTotal_k)
    nBackgroundTest_k = nBackgroundTotal_k - nBackgroundTrain_k

    # x are the features and y are the classes
    x_train_k = pd.concat(
        [sig_k[:nSignalTrain_k], bkg_k[:nBackgroundTrain_k]], axis=0, ignore_index=True)
    y_train_k = np.array([1]*nSignalTrain_k + [0]*nBackgroundTrain_k)

    x_test_k = pd.concat(
        [sig_k[nSignalTrain_k:], bkg_k[nBackgroundTrain_k:]], axis=0, ignore_index=True)
    y_test_k = np.array([1]*nSignalTest_k + [0]*nBackgroundTest_k)

    #  train the model
    model_k.fit(x_train_k, y_train_k, eval_set=[(x_test_k, y_test_k)])

    #  Evaluate Model
    pred_test = model_k.predict_proba(x_test_k)[:, 1].ravel()
    pred_train = model_k.predict_proba(x_train_k)[:, 1].ravel()

    auc = roc_auc_score(y_test_k, pred_test)
    acc = accuracy_score(y_test_k, np.round(pred_test))
    ll = log_loss(y_test_k, pred_test)

    # 2. Kolmogorov-Smirnov (KS) Test ("K Score")
    # We compare the distribution of predictions for Signal-Train vs Signal-Test
    # If they are different (High KS statistic, Low p-value), the model is overtrained.

    # Signal KS
    sig_train_preds = pred_train[y_train_k == 1]
    sig_test_preds = pred_test[y_test_k == 1]
    ks_stat_sig, ks_pval_sig = ks_2samp(sig_train_preds, sig_test_preds)

    # ks_pval determines similarity between train and test sets
    # ks_stat determines the degree of overtraining (higher = more overtrained)

    # Background KS
    bkg_train_preds = pred_train[y_train_k == 0]
    bkg_test_preds = pred_test[y_test_k == 0]
    ks_stat_bkg, ks_pval_bkg = ks_2samp(bkg_train_preds, bkg_test_preds)

    # Store metrics
    cv_metrics['auc'].append(auc)
    cv_metrics['accuracy'].append(acc)
    cv_metrics['log_loss'].append(ll)
    cv_metrics['ks_stat_sig'].append(ks_stat_sig)
    cv_metrics['ks_pval_sig'].append(ks_pval_sig)
    cv_metrics['ks_stat_bkg'].append(ks_stat_bkg)
    cv_metrics['ks_pval_bkg'].append(ks_pval_bkg)

    print(
        f"  AUC: {auc:.4f} | KS Score (Sig): {ks_stat_sig:.3f} (p={ks_pval_sig:.3f})")

    if detail:
        fig1, axes = plt.subplots(1, 2, figsize=(16, 6))

        ax_hist = axes[0]

        # Plot Train (Solid/Step) vs Test (Errorbar/Points)

        # Signal
        ax_hist.hist(sig_train_preds, bins=50, density=True,
                     alpha=0.3, color='blue', label='Sig (Train)')
        ax_hist.hist(sig_test_preds, bins=50, density=True,
                     histtype='step', linewidth=2, color='blue', label='Sig (Test)')

        # Background
        ax_hist.hist(bkg_train_preds, bins=50, density=True,
                     alpha=0.3, color='red', label='Bkg (Train)')
        ax_hist.hist(bkg_test_preds, bins=50, density=True,
                     histtype='step', linewidth=2, color='red', label='Bkg (Test)')

        ax_hist.set_title(f'Overtraining Check (KS p-val: {ks_pval_sig:.2f})')
        ax_hist.set_xlabel('Signal Probability')
        ax_hist.set_ylabel('Normalized Density')
        ax_hist.legend()

        # ROC Curve
        ax_roc = axes[1]
        fpr, tpr, _ = roc_curve(y_test_k, pred_test)

        ax_roc.plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
        ax_roc.plot([0, 1], [0, 1], 'r--', label='Random')

        ax_roc.set_title(f'ROC Curve (Model {i})')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.legend()
        ax_roc.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Feature Importance
        importance = model_k.get_booster().get_score(importance_type='gain')
        df = pd.DataFrame(list(importance.items()),
                          columns=['Feature', 'Importance'])
        df = df.sort_values(by='Importance', ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature',
                    hue='Feature', data=df, palette='viridis')
        plt.title(f'Top 10 Features by Gain model {i}')
        plt.show()

    models.append(model_k)

# %% Save Models
os.makedirs(f'models_{dataset}', exist_ok=True)

for i, model in enumerate(models):
    with open(f'models_{dataset}/xgboost_model_{i}.pkl', 'wb') as f:
        pickle.dump(model, f)
