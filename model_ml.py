"""
Using XGBoost to classify signal and background events.
Louis Liu 12/01
"""

from matplotlib import pyplot as plt
import pandas
import xgboost
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve
import filtered_data


sig, bkg = filtered_data.seperated_data(drop_cols=['B invariant mass'])
nSignalTotal = sig.shape[0]
nBackgroundTotal = bkg.shape[0]

# %% Define Model
model = xgboost.XGBClassifier(eval_metric='auc', early_stopping_rounds=50,
                              n_estimators=1000, learning_rate=0.3, max_depth=6)

# %% Train Model
nSignalTrain = int(0.9*nSignalTotal)
nSignalTest = nSignalTotal - nSignalTrain
nBackgroundTrain = int(0.9*nBackgroundTotal)
nBackgroundTest = nBackgroundTotal - nBackgroundTrain

# x are the features and y are the classes
x_train = pandas.concat(
    [sig[:nSignalTrain], bkg[:nBackgroundTrain]], axis=0, ignore_index=True)
y_train = [1]*nSignalTrain + [0]*nBackgroundTrain

x_test = pandas.concat(
    [sig[nSignalTrain:], bkg[nBackgroundTrain:]], axis=0, ignore_index=True)
y_test = [1]*nSignalTest + [0]*nBackgroundTest

model.fit(x_train, y_train, eval_set=[(x_test, y_test)])

# %% Evaluate Model
prediction = model.predict_proba(x_test)

plt.hist([p[1] for p, cls in zip(prediction, y_test)
         if cls == 1], bins=50, label='Signal')
plt.hist([p[1] for p, cls in zip(prediction, y_test)
         if cls == 0], bins=50, label='Background')
plt.xlabel(r'Signal Probability')
plt.ylabel(r'Candidates$')
plt.legend()
plt.show()

# %% ROC Curve
background_accepted, signal_accepted, probabilities_tested = roc_curve(
    y_test, prediction[:, 1])
plt.plot(background_accepted, signal_accepted, label='ROC curve')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random')
plt.xlabel(r'Background Contamination')
plt.ylabel(r'Signal Efficiency')
plt.legend()
plt.show()

a = xgboost.plot_importance(model)
