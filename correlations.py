# Looking at how two features correlate with each other

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_correlation_matrix(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, fmt=".2f", cmap='coolwarm', square=True, xticklabels=False, yticklabels=False)
    plt.title('Feature Correlation Heatmap')
    plt.savefig('figs/correlation_matrix.png')
    plt.show()

# plot the correlation matrix for B invariant mass
def plot_Binv_correlation(data):
    correlation_matrix = data.corrwith(data['B invariant mass']).to_frame()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, fmt=".2f", cmap='coolwarm', square=True, xticklabels=False, yticklabels=False)
    plt.title('B invariant mass correlations')
    plt.savefig('figs/B_inv_correlation_matrix.png')
    plt.show()

# drop all columns that correlate highly with target: "B invariant mass"
def drop_correlated(target, data, threshold=0.9):
    correlations = data.corr().abs()[target]
    to_drop = correlations[(correlations > threshold) & (correlations.index != target)].index
    print(f"Dropping {len(to_drop)} columns: {list(to_drop)}")
    return data.drop(columns=to_drop)

if __name__ == "__main__":
    data = pd.read_pickle('data/dataset_2011.pkl')
    drop_correlated('B invariant mass', data)