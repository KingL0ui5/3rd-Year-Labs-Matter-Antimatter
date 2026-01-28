"""
Plot feature importances from trained XGBoost models across k folds.
28/01
"""

import glob
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import config
sns.set_style('darkgrid')
sns.set_context('talk')

dataset = config.dataset
k = config.k


models = [None]*k
for file in glob.glob(f'models_{dataset}/xgboost_model_*.pkl'):
    model_k = int(os.path.basename(file).split('_')[-1].split('.')[0])
    with open(file, 'rb') as infile:
        model = pickle.load(infile)
        models[model_k] = model

all_importances = []
for i, model in enumerate(models):
    importance = model.get_booster().get_score(importance_type='gain')
    for feat, score in importance.items():
        all_importances.append(
            {'Feature': feat, 'Importance': score, 'Fold': i})

df_imp = pd.DataFrame(all_importances)

order = df_imp.groupby('Feature')['Importance'].mean(
).sort_values(ascending=False).index[:10]  # Top 15

plt.figure(figsize=(12, 8))
sns.barplot(
    data=df_imp,
    x='Importance',
    y='Feature',
    order=order,
    errorbar='sd',       # Show standard deviation error bars
    palette='viridis',
    estimator=np.mean,    # mean length of bars
    err_kws={'color': 'crimson', 'linewidth': 2.5}
)

plt.title(
    f'Feature Importance Stability (Mean $\pm$ Std Dev across {k} folds)')
plt.xlabel('Average Gain (Separation Power)')
plt.ylabel('Feature')
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
