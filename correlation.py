"""
Plot the top ten features correlated with each target variable on the same axes.
Fixed: Legend blocking data and title cutoff.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import seaborn as sns
import textwrap

sns.set_style('darkgrid')
sns.set_context('talk', font_scale=2)


def calculate_and_plot_correlations(targets, data, metric='Spearman', top_n=7, sample_size=20000):

    if isinstance(targets, str):
        targets = [targets]

    if len(data) > sample_size:
        print(f"Sampling {sample_size} events for speed...")
        calc_data = data.sample(n=sample_size, random_state=42)
    else:
        calc_data = data

    numeric_df = calc_data.select_dtypes(include=[np.number]).dropna()
    X = numeric_df.drop(columns=targets)

    all_results = []

    for target in targets:
        print(f"Calculating {metric} correlation with {target}...")
        y = numeric_df[target]

        if metric == 'Pearson':
            scores = X.corrwith(y, method='pearson').abs()
        elif metric == 'Spearman':
            scores = X.corrwith(y, method='spearman').abs()
        elif metric == 'Mutual Info':
            mi_scores = mutual_info_regression(X, y, random_state=42)
            scores = pd.Series(mi_scores, index=X.columns)
            if scores.max() > 0:
                scores = scores / scores.max()

        temp_df = pd.DataFrame({
            'Feature': scores.index,
            'Score': scores.values,
            'Target': target
        })
        all_results.append(temp_df)

    combined_df = pd.concat(all_results, ignore_index=True)

    # --- UNION SELECTION LOGIC ---
    features_to_keep = set()
    for target in targets:
        target_top = combined_df[combined_df['Target']
                                 == target].nlargest(top_n, 'Score')
        features_to_keep.update(target_top['Feature'].tolist())

    plot_df = combined_df[combined_df['Feature'].isin(features_to_keep)].copy()

    print(f"Plotting {len(features_to_keep)} unique features...")

    # Wrap labels
    plot_df['Label'] = plot_df['Feature'].apply(
        lambda x: '\n'.join(textwrap.wrap(x, width=40))
    )

    feature_order = plot_df.groupby(
        'Feature')['Score'].max().sort_values(ascending=False).index
    plot_df['Feature_Rank'] = pd.Categorical(
        plot_df['Feature'], categories=feature_order, ordered=True)
    plot_df = plot_df.sort_values('Feature_Rank')
    y_order = plot_df.drop_duplicates(
        'Label').sort_values('Feature_Rank')['Label']

    # --- PLOTTING ---
    # Increased width to 14 to help labels fit
    plt.figure(figsize=(14, len(features_to_keep) * 1.0 + 3))

    sns.barplot(
        data=plot_df,
        x='Score',
        y='Label',
        hue='Target',
        order=y_order,
        palette='viridis',
        edgecolor='white'
    )

    plt.title(
        f'Union of Top {top_n} Feature Correlations', fontsize=25, pad=20)
    plt.xlabel(f"{metric} Score", fontsize=25)
    plt.ylabel('')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tick_params(axis='y', labelsize=25)

    # --- FIX 1: EXTEND X-AXIS FOR LEGEND SPACE ---
    # Get current x-limit and extend it by 40% to make room on the right
    current_xmax = plot_df['Score'].max()
    plt.xlim(0, current_xmax * 1.4)

    # --- FIX 2: LEGEND PLACEMENT ---
    plt.legend(
        title='Target',
        loc='lower right',    # Bottom right inside the plot
        fontsize=18,
        title_fontsize=20,
        frameon=True,
        framealpha=0.95,      # Opaque background
        edgecolor='gray'
    )

    # --- FIX 3: PREVENT TITLE CUTOFF ---
    plt.tight_layout()
    # Add a little extra margin at the top AFTER tight_layout
    plt.subplots_adjust(top=0.93)

    plt.show()

    return plot_df


if __name__ == "__main__":
    import filtered_data
    import config

    data = filtered_data.load_dataset(dataset=config.dataset)

    calculate_and_plot_correlations(
        ['B invariant mass', 'dimuon-system invariant mass'], data, top_n=5)
