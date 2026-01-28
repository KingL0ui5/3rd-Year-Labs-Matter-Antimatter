import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import seaborn as sns
import textwrap

sns.set_style('whitegrid')
sns.set_context('talk')


def calculate_and_plot_correlations(target, data, metric='Spearman', top_n=10, sample_size=20000):
    """
    Calculates a SINGLE correlation metric with 'target' and plots it.

    Parameters:
    -----------
    target : str
        The target column name.
    data : pd.DataFrame
        The dataframe.
    metric : str
        One of ['Pearson', 'Spearman', 'Mutual Info'].
    """

    if len(data) > sample_size:
        print(f"Sampling {sample_size} events for speed...")
        calc_data = data.sample(n=sample_size, random_state=42)
    else:
        calc_data = data

    numeric_df = calc_data.select_dtypes(include=[np.number]).dropna()

    if target not in numeric_df.columns:
        raise ValueError(f"Target '{target}' not found in numeric columns.")

    X = numeric_df.drop(columns=[target])
    y = numeric_df[target]

    print(f"Calculating {metric} correlation with {target}...")

    if metric == 'Pearson':
        scores = X.corrwith(y, method='pearson').abs()
    elif metric == 'Spearman':
        scores = X.corrwith(y, method='spearman').abs()
    elif metric == 'Mutual Info':
        mi_scores = mutual_info_regression(X, y, random_state=42)
        scores = pd.Series(mi_scores, index=X.columns)
        scores = scores / scores.max()
    else:
        raise ValueError(
            "Metric must be 'Pearson', 'Spearman', or 'Mutual Info'")

    results = pd.DataFrame({metric: scores})
    top_results = results.sort_values(by=metric, ascending=False).head(top_n)

    top_results['Label'] = [
        '\n'.join(textwrap.wrap(name, width=40)) for name in top_results.index
    ]

    plt.figure(figsize=(10, len(top_results) * 0.8 + 2))

    barplot = sns.barplot(
        data=top_results,
        x=metric,
        y='Label',
        palette='viridis',
        hue=metric,      # Color by intensity
        legend=False     # Hide legend (color bar is self-explanatory)
    )

    plt.title(
        f'Top {top_n} Features by {metric} Correlation\n(Target: {target})', fontsize=16, pad=20)
    plt.xlabel('Correlation Magnitude', fontsize=14)
    plt.ylabel('')

    for i, container in enumerate(barplot.containers):
        barplot.bar_label(container, fmt='%.2f', padding=5)

    plt.xlim(0, 1.15)  # Add room for labels
    plt.tight_layout()
    plt.show()

    return top_results


if __name__ == "__main__":

    import filtered_data
    import config
    data = filtered_data.load_dataset(dataset=config.dataset)
    calculate_and_plot_correlations('B invariant mass', data, top_n=10)
    calculate_and_plot_correlations(
        'dimuon-system invariant mass', data, top_n=10)
