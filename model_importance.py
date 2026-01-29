"""
Plot feature importances and ROC stability from trained XGBoost models across k folds.
Updated to match Correlation Plot style.
"""
import glob
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import textwrap
from sklearn.metrics import roc_curve, auc
import config
import filtered_data

# --- STYLE UPDATE: Match Correlation Plot ---
sns.set_style('darkgrid')
sns.set_context('talk', font_scale=2)  # Increased scale

dataset = config.dataset
k = config.k

seperation = filtered_data.seperate(k=k, dataset=dataset)
sig, bkg = seperation.data(drop_cols=config.drop_cols)

print("Loading Models...")
models = [None]*k
for file in glob.glob(f'models_{dataset}/xgboost_model_*.pkl'):
    model_k = int(os.path.basename(file).split('_')[-1].split('.')[0])
    with open(file, 'rb') as infile:
        model = pickle.load(infile)
        models[model_k] = model

all_importances = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

print("Processing Folds...")
for i in range(k):
    model = models[i]
    if model is None:
        continue

    importance = model.get_booster().get_score(importance_type='gain')
    for feat, score in importance.items():
        all_importances.append(
            {'Feature': feat, 'Importance': score, 'Fold': i})

    sig_k = sig[i]
    bkg_k = bkg[i]

    nSigTrain = int(0.9 * sig_k.shape[0])
    nBkgTrain = int(0.9 * bkg_k.shape[0])

    x_test_sig = sig_k[nSigTrain:]
    x_test_bkg = bkg_k[nBkgTrain:]

    x_test = pd.concat([x_test_sig, x_test_bkg], ignore_index=True)
    y_test = np.array([1]*len(x_test_sig) + [0]*len(x_test_bkg))

    probas_ = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probas_)

    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(auc(fpr, tpr))

df_imp = pd.DataFrame(all_importances)

# Wrap text to approx 40 characters (same as correlation plot)
df_imp['Feature'] = df_imp['Feature'].apply(
    lambda x: '\n'.join(textwrap.wrap(x, width=40)))

order = df_imp.groupby('Feature')['Importance'].mean(
).sort_values(ascending=False).index[:10]

# --- PLOT 1: FEATURE IMPORTANCE ---
# Increased size to handle large fonts
plt.figure(figsize=(14, 12))

sns.barplot(
    data=df_imp,
    x='Importance',
    y='Feature',
    order=order,
    errorbar='sd',
    palette='viridis',
    estimator=np.mean,
    edgecolor='white',  # Clean edges like correlation plot
    # Thicker error bars for visibility
    err_kws={'color': 'crimson', 'linewidth': 3}
)

plt.title(f'Feature Importance Stability (k={k})',
          fontsize=25, pad=20)
plt.xlabel('Average Gain (Separation Power)', fontsize=25)
plt.ylabel('')  # Remove y-label

# Explicitly set tick size
plt.tick_params(axis='y', labelsize=20)
plt.tick_params(axis='x', labelsize=20)

plt.tight_layout()
plt.show()


# --- PLOT 2: ROC CURVE ---
plt.figure(figsize=(12, 10))

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
std_tpr = np.std(tprs, axis=0)

plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' %
         (np.mean(aucs), np.std(aucs)), lw=3, alpha=.8)

tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.plot([0, 1], [0, 1], linestyle='--', lw=3,
         color='r', label='Random', alpha=.8)

plt.xlabel('False Positive Rate', fontsize=25)
plt.ylabel('True Positive Rate', fontsize=25)
plt.title(f'ROC Stability ({k}-Fold CV)', fontsize=30, pad=20)
plt.legend(loc="lower right", fontsize=18)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
