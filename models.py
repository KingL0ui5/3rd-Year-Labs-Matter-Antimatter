"""
Using XGBoost to classify signal and background events.
Louis Liu 12/01
"""
import filtered_data
from sklearn.metrics import roc_curve
import pandas as pd
import xgboost
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
sns.set_style('darkgrid')
sns.set_context('paper')

detail = True
models = []
k = 5

# %% Load Data
filtered = filtered_data.seperate(drop_cols=['B invariant mass',
                                             'dimuon-system invariant mass', 'index'], k=k)

sig, bkg = filtered.data()

#  save the filtered object
with open('data/filtered_data.pkl', 'wb') as f:
    pickle.dump(filtered, f)

# %% Model Loop

for i in range(k):
    sig_k = sig[i]
    bkg_k = bkg[i]
    nSignalTotal_k = sig_k.shape[0]
    nBackgroundTotal_k = bkg_k.shape[0]

    model_k = xgboost.XGBClassifier(eval_metric='auc', early_stopping_rounds=50,
                                    n_estimators=1000, learning_rate=0.3, max_depth=6, base_score=0.5)

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

    plt.hist([p[1] for p, cls in zip(prediction_k, y_test_k)
              if cls == 1], bins=50, label='Signal')
    plt.hist([p[1] for p, cls in zip(prediction_k, y_test_k)
              if cls == 0], bins=50, label='Background')
    plt.xlabel(r'Signal Probability')
    plt.ylabel(r'Candidates$')
    plt.yscale('log')
    plt.legend()
    plt.show()

    if detail:
        # ROC Curve and Feature Importance
        background_accepted_k, signal_accepted_k, probabilities_tested_k = roc_curve(
            y_test_k, prediction_k[:, 1])
        plt.plot(background_accepted_k, signal_accepted_k, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random')
        plt.title(f'ROC Curve model {i}')
        plt.xlabel(r'Background Contamination')
        plt.ylabel(r'Signal Efficiency')
        plt.legend()
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
for i, model in enumerate(models):
    with open(f'models/xgboost_model_{i}.pkl', 'wb') as f:
        pickle.dump(model, f)
