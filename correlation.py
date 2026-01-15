# Looking at how two features correlate with each other

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import seaborn as sns


def plot_correlation_matrix(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, fmt=".2f", cmap='coolwarm',
                square=True, xticklabels=False, yticklabels=False)
    plt.title('Feature Correlation Heatmap')
    # plt.savefig('figs/correlation_matrix.png')
    plt.show()

# plot the correlation matrix for B invariant mass


def plot_Binv_correlation(data):
    correlation_matrix = data.corrwith(data['B invariant mass']).to_frame()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, fmt=".2f", cmap='coolwarm',
                square=True, xticklabels=False, yticklabels=False)
    plt.title('B invariant mass correlations')
    # plt.savefig('figs/B_inv_correlation_matrix.png')
    plt.show()
    return correlation_matrix.sort_values(by=0, ascending=False)

#  drop all columns that correlate highly with target: "B invariant mass"


def drop_correlated(target, data, threshold=0.5):
    correlations = data.corr().abs()[target]
    to_drop = correlations[((correlations > threshold) &
                           (correlations.index != target)) | correlations.isna()].index
    print(f"Dropping {len(to_drop)} columns: {list(to_drop)}")
    return data.drop(columns=to_drop)


def calculate_all_correlations(target, data, sample_size=10000):
    # 1. Representative Sampling for speed
    if len(data) > sample_size:
        print(f"Sampling {sample_size} events for speed...")
        calc_data = data.sample(n=sample_size, random_state=42)
    else:
        calc_data = data

    # Ensure we only work with numeric, non-NaN data
    numeric_df = calc_data.select_dtypes(include=[np.number]).dropna()
    X = numeric_df.drop(columns=[target])
    y = numeric_df[target]

    print("Calculating Pearson, Spearman, and Kendall...")
    pearson = numeric_df.corr(method='pearson')[target]
    spearman = numeric_df.corr(method='spearman')[target]
    kendall = numeric_df.corr(method='kendall')[target]

    mi_scores = mutual_info_regression(X, y, n_jobs=-1, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns)

    results = pd.DataFrame({
        'Pearson': pearson,
        'Spearman': spearman,
        'Kendall': kendall,
        'Mutual_Info': mi_series
    }).drop(index=target, errors='ignore')

    return results.sort_values(by='Mutual_Info', ascending=False)


if __name__ == "__main__":
    # Load full dataset
    data = pd.read_pickle('data/dataset_2011.pkl')

    # Run calculation on sample
    correlations = calculate_all_correlations('B invariant mass', data)

    # Output results
    correlations.to_csv("output.txt", sep='\t', index=True)
    plot_correlations()
    print("Done! Results saved to output.txt")
