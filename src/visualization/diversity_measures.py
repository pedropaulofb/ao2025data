import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from src.color_legend import color_text


def execute_visualization_diversity_measures(in_dir_path, out_dir_path, file_path):
    # Read CSV file
    df = pd.read_csv(os.path.join(in_dir_path, file_path))

    # Prepare the data
    x = np.arange(len(df['Construct']))  # The label locations
    bar_width = 0.25  # Width of each bar

    # Create the figure
    fig, ax1 = plt.subplots(figsize=(16, 9), tight_layout=True)  # Increased figure width for more space

    # Plot Shannon Entropy on the first y-axis (left)
    ax1.bar(x - bar_width, df['Shannon Entropy'], width=bar_width, color='#1f77b4', label='Shannon Entropy')
    ax1.set_ylabel('Shannon Entropy')
    ax1.set_xlabel('Construct')

    # Adjust x-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Construct'], rotation=45, ha='right')  # Rotate labels and align to the right

    # Apply color to specific labels
    color_text(ax1.get_xticklabels())

    # Create a second y-axis (right) that shares the same x-axis
    ax2 = ax1.twinx()

    # Plot Gini Coefficient and Simpson Index on the second y-axis (right)
    ax2.bar(x, df['Gini Coefficient'], width=bar_width, color='#ff7f0e', label='Gini Coefficient')
    # ax2.bar(x + bar_width, df['Simpson Index'], width=bar_width, color='#2ca02c', label='Simpson Index')
    ax2.set_ylabel('Gini Coefficient')

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Increase spacing between the labels by setting x-axis limits and tight layout
    plt.xlim(-0.5, len(df['Construct']) - 0.5)  # Adjust limits to give more space around bars
    plt.title('Comparison of Diversity Measures Across Constructs', fontweight='bold')
    plt.tight_layout()

    fig_name = 'diversity_measures_comparison_across_constructs.png'
    fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
    plt.close(fig)

    # 2. Calculate Correlation Matrix
    corr = df[['Shannon Entropy', 'Gini Coefficient', 'Simpson Index']].corr()

    # Plot Heatmap
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Heatmap of Correlations Among Diversity Measures', fontweight='bold')

    fig_name = 'correlation_heatmap_diversity_measures.png'
    fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
    plt.close(fig)

    # 3. Box Plot

    # Melt the data to a long format
    df_melted = df.melt(id_vars='Construct', value_vars=['Shannon Entropy', 'Gini Coefficient', 'Simpson Index'],
                        var_name='Measure', value_name='Value')

    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), tight_layout=True)

    # Plot for Shannon Entropy
    sns.boxplot(x='Measure', y='Value', data=df_melted[df_melted['Measure'] == 'Shannon Entropy'], ax=ax1)
    ax1.set_title('Variation of Shannon Entropy Across Constructs', fontweight='bold')
    ax1.set_xlabel('Measure')
    ax1.set_ylabel('Value')

    # Plot for Gini Coefficient and Simpson Index
    sns.boxplot(x='Measure', y='Value',
                data=df_melted[df_melted['Measure'].isin(['Gini Coefficient', 'Simpson Index'])], ax=ax2)
    ax2.set_title('Variation of Gini Coefficient and Simpson Index Across Constructs', fontweight='bold')
    ax2.set_xlabel('Measure')
    ax2.set_ylabel('Value')

    plt.tight_layout()

    fig_name = 'box_plot_variation_of_diversity_measures.png'
    fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
    plt.close(fig)

    # 4. Density Plot

    # Plotting Histograms with Seaborn
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), tight_layout=True)

    # Histogram for Shannon Entropy
    sns.histplot(df['Shannon Entropy'], bins=15, ax=ax1, kde=True, color='blue')
    ax1.set_title('Density Plot of Shannon Entropy', fontweight='bold')
    ax1.set_xlabel('Metric Value')
    ax1.set_xlim(0, None)  # Start at 0 and go up to the maximum observed

    # Histogram for Gini Coefficient and Simpson Index
    sns.histplot(df['Gini Coefficient'], bins=15, ax=ax2, kde=True, color='orange', label='Gini Coefficient', alpha=0.6)
    sns.histplot(df['Simpson Index'], bins=15, ax=ax2, kde=True, color='green', label='Simpson Index', alpha=0.6)
    ax2.set_title('Density Plot of Gini Coefficient and Simpson Index', fontweight='bold')
    ax2.set_xlabel('Metric Value')
    ax2.set_xlim(0, 1)  # Restrict to [0, 1] range
    ax2.legend()

    plt.tight_layout()

    fig_name = 'density_plot_of_diversity_measures.png'
    fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
    plt.close(fig)
