import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from src.create_figure_subdir import create_figures_subdir


def execute_visualization_group3(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)
    save_dir = create_figures_subdir(file_path)

    # Prepare the data
    x = np.arange(len(df['Construct']))  # The label locations
    bar_width = 0.25  # Width of each bar

    # Create the figure
    fig, ax1 = plt.subplots(figsize=(12, 8), tight_layout=True)  # Increased figure width for more space

    # Plot Shannon Entropy on the first y-axis (left)
    ax1.bar(x - bar_width, df['Shannon Entropy'], width=bar_width, color='#1f77b4', label='Shannon Entropy')
    ax1.set_ylabel('Shannon Entropy')
    ax1.set_xlabel('Construct')

    # Adjust x-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Construct'], rotation=45, ha='right')  # Rotate labels and align to the right

    # Apply color to specific labels
    for label in ax1.get_xticklabels():
        if label.get_text() == 'none':
            label.set_color('blue')
        elif label.get_text() == 'other':
            label.set_color('red')

    # Create a second y-axis (right) that shares the same x-axis
    ax2 = ax1.twinx()

    # Plot Gini Coefficient and Simpson Index on the second y-axis (right)
    ax2.bar(x, df['Gini Coefficient'], width=bar_width, color='#ff7f0e', label='Gini Coefficient')
    ax2.bar(x + bar_width, df['Simpson Index'], width=bar_width, color='#2ca02c', label='Simpson Index')
    ax2.set_ylabel('Gini Coefficient and Simpson Index')

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Increase spacing between the labels by setting x-axis limits and tight layout
    plt.xlim(-0.5, len(df['Construct']) - 0.5)  # Adjust limits to give more space around bars
    plt.title('Combined Bar Chart with Two Y-Axes for Shannon Entropy, Gini Coefficient, and Simpson Index')
    plt.tight_layout()

    fig_name = 'group3_fig1.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")

    # 2. Calculate Correlation Matrix
    corr = df[['Shannon Entropy', 'Gini Coefficient', 'Simpson Index']].corr()

    # Plot Heatmap
    plt.figure(figsize=(12, 8), tight_layout=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Metrics')

    fig_name = 'group3_fig2.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")

    # 3. Box Plot

    # Melt the data to a long format
    df_melted = df.melt(id_vars='Construct', value_vars=['Shannon Entropy', 'Gini Coefficient', 'Simpson Index'],
                        var_name='Measure', value_name='Value')

    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), tight_layout=True)

    # Plot for Shannon Entropy
    sns.boxplot(x='Measure', y='Value', data=df_melted[df_melted['Measure'] == 'Shannon Entropy'], ax=ax1)
    ax1.set_title('Box Plot for Shannon Entropy')
    ax1.set_xlabel('Measure')
    ax1.set_ylabel('Value')

    # Plot for Gini Coefficient and Simpson Index
    sns.boxplot(x='Measure', y='Value',
                data=df_melted[df_melted['Measure'].isin(['Gini Coefficient', 'Simpson Index'])], ax=ax2)
    ax2.set_title('Box Plot for Gini Coefficient and Simpson Index')
    ax2.set_xlabel('Measure')
    ax2.set_ylabel('Value')

    plt.tight_layout()

    fig_name = 'group3_fig3.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")

    # 4. Density Plot

    # Plotting Histograms with Seaborn
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), tight_layout=True)

    # Histogram for Shannon Entropy
    sns.histplot(df['Shannon Entropy'], bins=15, ax=ax1, kde=True, color='blue')
    ax1.set_title('Histogram of Shannon Entropy')
    ax1.set_xlabel('Value')
    ax1.set_xlim(0, None)  # Start at 0 and go up to the maximum observed

    # Histogram for Gini Coefficient and Simpson Index
    sns.histplot(df['Gini Coefficient'], bins=15, ax=ax2, kde=True, color='orange', label='Gini Coefficient', alpha=0.6)
    sns.histplot(df['Simpson Index'], bins=15, ax=ax2, kde=True, color='green', label='Simpson Index', alpha=0.6)
    ax2.set_title('Histogram of Gini Coefficient and Simpson Index')
    ax2.set_xlabel('Value')
    ax2.set_xlim(0, 1)  # Restrict to [0, 1] range
    ax2.legend()

    plt.tight_layout()

    fig_name = 'group3_fig4.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")


execute_visualization_group3('../outputs/analyses/cs_analyses/diversity_measures.csv')
