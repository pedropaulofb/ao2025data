import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from src.color_legend import color_text
from src.create_figure_subdir import create_figures_subdir


def execute_visualization_group72(file_path):
    # Load the data from CSV file
    df = pd.read_csv(file_path, index_col=0)
    save_dir = create_figures_subdir(file_path)

    # 1. Heatmap

    # Create a heatmap using Seaborn
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    ax = sns.heatmap(df, annot=True, fmt=".2f", cmap='viridis', linewidths=.5)

    # Customize tick label colors
    color_text(ax.get_xticklabels())
    color_text(ax.get_yticklabels())

    # Customize the plot
    plt.title('Heatmap of Mutual Information Between Constructs', fontweight='bold')
    plt.xlabel('Construct')
    plt.ylabel('Construct')

    fig_name = 'mutual_information_heatmap.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")
    plt.close(fig)

    # 2. Network Graph

    # Convert the DataFrame to a long format for the network graph
    df_long_network = df.stack().reset_index()
    df_long_network.columns = ['Construct 1', 'Construct 2', 'Mutual Information']

    # Remove self-correlations (mutual-information of a construct with itself)
    df_long_network = df_long_network[df_long_network['Construct 1'] != df_long_network['Construct 2']]

    # Drop duplicate pairs (e.g., both (A, B) and (B, A))
    df_long_network['Construct Pair'] = df_long_network.apply(lambda row: tuple(sorted([row['Construct 1'], row['Construct 2']])),
                                              axis=1)
    df_long_network = df_long_network.drop_duplicates(subset='Construct Pair')

    # Select the N most mutual information values
    N = 10
    top_mutual = df_long_network.nlargest(N, 'Mutual Information')

    # Select the N least mutual information values
    least_mutual = df_long_network.nsmallest(N, 'Mutual Information')

    # Function to create a network graph with a legend
    def plot_network(data, title, metric_label, palette, fig_name):
        # Create an empty graph
        G = nx.Graph()

        # Add edges to the graph
        for _, row in data.iterrows():
            G.add_edge(row['Construct 1'], row['Construct 2'], weight=row[metric_label])

        # Draw the graph
        pos = nx.circular_layout(G)  # Use spring layout for better visualization
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

    # Plot network for the N most mutual information values
    plot_network(top_mutual, f'Top {N} Pairs with Highest Mutual Information', 'Mutual Information',
                 palette=plt.cm.autumn_r, fig_name='network_top_high_mutual_information.png')

    # Plot network for the N least mutual information values
    plot_network(least_mutual, f'Top {N} Pairs with Lowest Mutual Information', 'Mutual Information',
                 palette=plt.cm.winter, fig_name='network_top_low_mutual_information.png')

    # 3. Rank Chart

    # Calculate the sum of the mutual information values for each construct
    construct_importance_mi = df.sum(axis=1)

    # Sort the constructs by their importance in descending order
    construct_importance_mi_sorted = construct_importance_mi.sort_values(ascending=False)

    # Create a bar chart to visualize construct importance
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    ax = sns.barplot(x=construct_importance_mi_sorted.values, y=construct_importance_mi_sorted.index,
                     hue=construct_importance_mi_sorted.index, palette='viridis', dodge=False, legend=False)

    # Highlight specific constructs if needed
    color_text(ax.get_yticklabels())

    plt.title('Ranking of Constructs by Total Mutual Information', fontweight='bold')
    plt.xlabel('"Total Mutual Information')
    plt.ylabel('Construct')
    # Customize grid lines: Remove horizontal, keep vertical
    plt.grid(axis='y', linestyle='')  # Remove horizontal grid lines
    plt.grid(axis='x', linestyle='-', color='gray')  # Keep vertical grid lines
    # Set x-axis ticks dynamically based on data range
    plt.xticks(np.arange(0, construct_importance_mi_sorted.max() + 0.5, 0.5))
    fig_name = 'construct_ranking_mutual_information.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")
    plt.close(fig)

    # 4. Box Plot

    # Convert the DataFrame to a long format for the box plot
    df_long_boxplot = df.stack().reset_index()
    df_long_boxplot.columns = ['Construct 1', 'Construct 2', 'Mutual Information']

    # Remove self-correlations (mutual information of a construct with itself)
    df_long_boxplot = df_long_boxplot[df_long_boxplot['Construct 1'] != df_long_boxplot['Construct 2']]

    # Create a box plot to show the distribution of mutual information for each construct
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    ax = sns.boxplot(x='Construct 1', y='Mutual Information', data=df_long_boxplot, hue='Construct 1', palette='viridis',
                     legend=False)
    plt.title('Box Plot of Mutual Information by Construct', fontweight='bold')
    plt.xlabel('Construct')
    plt.ylabel('Mutual Information')
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    # Customize x-axis label colors
    color_text(ax.get_xticklabels())

    fig_name = 'mutual_information_boxplot.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")
    plt.close(fig)


execute_visualization_group72('../outputs/analyses/cs_analyses/mutual_information.csv')
