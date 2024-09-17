import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from src.color_legend import color_text
from src.create_figure_subdir import create_figures_subdir


def execute_visualization_spearman_correlation(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, index_col=0)
    save_dir = create_figures_subdir(file_path)

    # 1. Heatmap

    # Plotting the heatmap
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    ax = sns.heatmap(df, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=0.5)
    plt.title('Spearman Correlation Heatmap of Constructs', fontweight='bold')

    # Customize tick label colors
    color_text(ax.get_xticklabels())
    color_text(ax.get_yticklabels())

    plt.xticks(rotation=45, ha='right')  # Adjust the rotation and alignment for x-axis labels
    plt.yticks(rotation=0)  # Adjust the rotation for y-axis labels
    fig_name = 'spearman_correlation_heatmap.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")
    plt.close(fig)

    # 2. Network Graph of Correlations

    # Convert the DataFrame to a long format
    df_long = df.stack().reset_index()
    df_long.columns = ['Construct 1', 'Construct 2', 'Spearman Correlation']

    # Remove self-correlations (correlation of a construct with itself)
    df_long = df_long[df_long['Construct 1'] != df_long['Construct 2']]

    # Select the N most positive correlations
    N = 10
    top_positive = df_long.nlargest(N, 'Spearman Correlation')

    # Select the N most negative correlations
    top_negative = df_long.nsmallest(N, 'Spearman Correlation')

    # Select the N correlations closest to zero
    near_zero = df_long.iloc[(df_long['Spearman Correlation'].abs().argsort()[:N])]

    # Function to create a network graph with a legend
    def plot_network(data, title, metric_label, palette,fig_name):
        # Create an empty graph
        G = nx.Graph()

        # Add edges to the graph
        for _, row in data.iterrows():
            G.add_edge(row['Construct 1'], row['Construct 2'], weight=row[metric_label])

        # Draw the graph
        pos = nx.circular_layout(G)
        weights = [G[u][v]['weight'] for u, v in G.edges()]

        # Create a figure and axis for the plot
        fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)

        # Draw nodes and edges
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color=weights,
                edge_cmap=palette, width=3, edge_vmin=min(weights), edge_vmax=max(weights), ax=ax)

        # Add a color bar
        sm = plt.cm.ScalarMappable(cmap=palette, norm=plt.Normalize(vmin=min(weights), vmax=max(weights)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)  # Associate the color bar with the current axis
        cbar.set_label(metric_label)

        plt.title(title, fontweight='bold')
        fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
        logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")
        plt.close(fig)

    # Plot network for the N most positive Spearman Correlations
    plot_network(top_positive, f'Top {N} Positive Spearman Correlations Network', 'Spearman Correlation',
                 palette=plt.cm.autumn_r,fig_name='network_top_positive_spearman_correlations.png')

    # Plot network for the N most negative Spearman Correlations
    plot_network(top_negative, f'Top {N} Negative Spearman Correlations Network', 'Spearman Correlation',
                 palette=plt.cm.winter,fig_name='network_top_negative_spearman_correlations.png')

    # Plot network for the N correlations closest to zero
    plot_network(near_zero, f'Top {N} Neutral Spearman Correlations Network', 'Spearman Correlation',
                 palette=plt.cm.summer,fig_name='network_top_neutral_spearman_correlations.png')

    # 3. Ranked Chart

    # Calculate the sum of the absolute correlation values for each construct
    construct_importance = df.abs().sum(axis=1)

    # Sort the constructs by their importance in descending order
    construct_importance_sorted = construct_importance.sort_values(ascending=False)

    # Create a bar chart to visualize construct importance
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    ay = sns.barplot(x=construct_importance_sorted.values, y=construct_importance_sorted.index,
                     hue=construct_importance_sorted.index, palette='viridis', dodge=False, legend=False)

    color_text(ay.get_yticklabels())

    plt.title('Ranking of Constructs by Total Absolute Correlation', fontweight='bold')
    plt.xlabel('Total Absolute Correlation')
    plt.ylabel('Construct')
    # Customize grid lines: Remove horizontal, keep vertical
    plt.grid(axis='y', linestyle='')  # Remove horizontal grid lines
    plt.grid(axis='x', linestyle='-', color='gray')  # Keep vertical grid lines
    # Set x-axis ticks to be every 0.5 units
    plt.xticks(np.arange(0, construct_importance_sorted.max() + 0.5, 0.5))
    fig_name = 'construct_ranking_by_absolute_correlation.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")
    plt.close(fig)

    # 4. Box Plot

    # Create a box plot to show the distribution of correlations for each construct
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    ax = sns.boxplot(x='Construct 1', y='Spearman Correlation', data=df_long, hue='Construct 1', palette='viridis',
                     dodge=False, legend=False)
    plt.title('Box Plot of Spearman Correlations by Construct', fontweight='bold')
    plt.xlabel('Construct')
    plt.ylabel('Spearman Correlation Coefficient')
    plt.xticks(rotation=90)
    plt.grid(axis='y')

    # Customize x-axis label colors
    color_text(ax.get_xticklabels())

    fig_name = 'boxplot_spearman_correlations_by_construct.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")
    plt.close(fig)

