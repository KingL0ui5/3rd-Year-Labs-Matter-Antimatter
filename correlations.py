# Looking at how two features correlate with each other

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import histograms

data = pd.read_pickle('data/dataset_2011.pkl')

# def plot_correlations(feature_x, feature_y):
#     feature_1 = data[feature_x]
#     feature_2 = data[feature_y]
#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(x=feature_1, y=feature_2, alpha=0.5)
#     plt.title(f'Scatter plot of {feature_x} vs {feature_y}')
#     plt.xlabel(feature_x)
#     plt.ylabel(feature_y)
#     plt.grid(True)
#     plt.show()

correlation_matrix = data.corr()

def plot_heatmap(corr_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Feature Correlation Heatmap')
    plt.show()

plot_heatmap(correlation_matrix)
