"""
Using XGBoost to classify signal and background events.
Louis Liu 12/01
"""
import numpy as np
import filtered_data
from sklearn.metrics import roc_curve
import pandas as pd
import xgboost
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
sns.set_context('paper')

# %% Load Data

sig, bkg = filtered_data.seperated_data(drop_cols=['B invariant mass',
                                        'dimuon-system invariant mass'])
nSignalTotal = sig.shape[0]
nBackgroundTotal = bkg.shape[0]

# %% Define Model
model = xgboost.XGBClassifier(eval_metric='auc', early_stopping_rounds=50,
                              n_estimators=1000, learning_rate=0.3, max_depth=6, base_score=0.5)

# %% Train Model
nSignalTrain = int(0.9*nSignalTotal)
nSignalTest = nSignalTotal - nSignalTrain
nBackgroundTrain = int(0.9*nBackgroundTotal)
nBackgroundTest = nBackgroundTotal - nBackgroundTrain

# x are the features and y are the classes
x_train = pd.concat(
    [sig[:nSignalTrain], bkg[:nBackgroundTrain]], axis=0, ignore_index=True)
y_train = [1]*nSignalTrain + [0]*nBackgroundTrain

x_test = pd.concat(
    [sig[nSignalTrain:], bkg[nBackgroundTrain:]], axis=0, ignore_index=True)
y_test = [1]*nSignalTest + [0]*nBackgroundTest

model.fit(x_train, y_train, eval_set=[(x_test, y_test)])

# %% Evaluate Model
prediction = model.predict_proba(x_test)

np.savetxt("data/output.txt", prediction, delimiter='\t')

plt.hist([p[1] for p, cls in zip(prediction, y_test)
         if cls == 1], bins=50, label='Signal')
plt.hist([p[1] for p, cls in zip(prediction, y_test)
         if cls == 0], bins=50, label='Background')
plt.xlabel(r'Signal Probability')
plt.ylabel(r'Candidates$')
plt.yscale('log')
plt.legend()
plt.show()

# %% ROC Curve and Feature Importance
background_accepted, signal_accepted, probabilities_tested = roc_curve(
    y_test, prediction[:, 1])
plt.plot(background_accepted, signal_accepted, label='ROC curve')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random')
plt.xlabel(r'Background Contamination')
plt.ylabel(r'Signal Efficiency')
plt.legend()
plt.show()

importance = model.get_booster().get_score(importance_type='gain')

df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
df = df.sort_values(by='Importance', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature',
            hue='Feature', data=df, palette='viridis')

plt.title('Top 10 Features by Gain', fontsize=15)
plt.xlabel('Gain', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.show()

# %% Save data

df_pred = pd.DataFrame(prediction, columns=[
                       'Background Probability', 'Signal Probability'])
df_pred.to_pickle('data/predictions.pkl')
