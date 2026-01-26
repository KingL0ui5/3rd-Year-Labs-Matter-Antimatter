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

    # Keeping MI calculation for the return dataframe, but skipping for plot
    mi_scores = mutual_info_regression(X, y, n_jobs=-1, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns)

    results = pd.DataFrame({
        'Pearson': pearson,
        'Spearman': spearman,
        'Kendall': kendall,
        'Mutual_Info': mi_series
    }).drop(index=target, errors='ignore')

    methods = ['Pearson', 'Spearman', 'Kendall']
    plot_data = results[methods]

    fig, axes = plt.subplots(1, 3, figsize=(12, 8), sharey=True,
                             gridspec_kw={'wspace': 0.5})

    for i, method in enumerate(methods):
        sns.heatmap(
            plot_data[[method]],
            ax=axes[i],
            annot=False,
            cmap='coolwarm',
            yticklabels=False,
            xticklabels=[method],  # Label the bar at the bottom
            center=0,
            cbar=False  # Remove individual colorbars for a cleaner look
        )
        for _, spine in axes[i].spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.5)

    plt.suptitle(f'Statistical Correlation Methods with {target}', fontsize=16)

    mappable = axes[0].get_children()[0]
    fig.colorbar(mappable, ax=axes, orientation='vertical',
                 fraction=0.02, pad=0.04)

    plt.show()

    return results.sort_values(by='Mutual_Info', ascending=False)


if __name__ == "__main__":
    # Load full dataset
    data = pd.read_pickle('datasets/dataset_2011.pkl')

    # Run calculation on sample
    correlations_B = calculate_all_correlations('B invariant mass', data)
    correlations_dimuon = calculate_all_correlations(
        'dimuon-system invariant mass', data)

    # Output results
    correlations_B.to_csv("output_B.txt", sep='\t', index=True)
    correlations_dimuon.to_csv("output_dimuon.txt", sep='\t', index=True)

    print("Done! Results saved to output_B.txt and output_dimuon.txt")
