import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from src.utils import color_text


def execute_visualization_mutual_information(in_dir_path, out_dir_path, file_path):
    # Load the data from CSV file
    df = pd.read_csv(os.path.join(in_dir_path, file_path), index_col=0)

    # 1. Heatmap

    # Create a heatmap using Seaborn
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    ax = sns.heatmap(df, annot=True, fmt=".2f", cmap='viridis', linewidths=.5)

    # Customize tick label colors
    color_text(ax.get_xticklabels())
    color_text(ax.get_yticklabels())

    # Customize the plot
    plt.title('Heatmap of Mutual Information Between Stereotypes', fontweight='bold')
    plt.xlabel('Stereotype')
    plt.ylabel('Stereotype')

    fig_name = 'mutual_information_heatmap.png'
    fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
    plt.close(fig)

    # 2. Network Graph

    # Convert the DataFrame to a long format for the network graph
    df_long_network = df.stack().reset_index()
    df_long_network.columns = ['Stereotype 1', 'Stereotype 2', 'Mutual Information']

    # Remove self-correlations (mutual-information of a stereotype with itself)
    df_long_network = df_long_network[df_long_network['Stereotype 1'] != df_long_network['Stereotype 2']]

    # Drop duplicate pairs (e.g., both (A, B) and (B, A))
    df_long_network['Stereotype Pair'] = df_long_network.apply(
        lambda row: tuple(sorted([row['Stereotype 1'], row['Stereotype 2']])), axis=1
    )
    df_long_network = df_long_network.drop_duplicates(subset='Stereotype Pair')

    # Select the N most mutual information values
    N = 10
    top_mutual = df_long_network.nlargest(N, 'Mutual Information')

    # Select the N least mutual information values
    least_mutual = df_long_network.nsmallest(N, 'Mutual Information')

    # Function to create a network graph with a legend
    def plot_network(data, title, metric_label, palette, fig_name):
        # Create an empty graph
        G = nx.Graph()

        # Add edges to the graph, making sure we only add each pair once
        added_edges = set()  # A set to track which edges have already been added
        for _, row in data.iterrows():
            edge = tuple(sorted([row['Stereotype 1'], row['Stereotype 2']]))
            if edge not in added_edges:
                G.add_edge(edge[0], edge[1], weight=row[metric_label])
                added_edges.add(edge)

        # Draw the graph
        pos = nx.circular_layout(G)  # Use circular layout for visualization
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
        fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
        logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
        plt.close(fig)

    # Plot network for the N most mutual information values
    plot_network(top_mutual, f'Top {N} Pairs with Highest Mutual Information', 'Mutual Information',
                 palette=plt.cm.autumn_r, fig_name='network_top_high_mutual_information.png')

    # Plot network for the N least mutual information values
    plot_network(least_mutual, f'Top {N} Pairs with Lowest Mutual Information', 'Mutual Information',
                 palette=plt.cm.winter, fig_name='network_top_low_mutual_information.png')

    # 3. Rank Chart

    # Calculate the sum of the mutual information values for each stereotype
    stereotype_importance_mi = df.sum(axis=1)

    # Sort the stereotypes by their importance in descending order
    stereotype_importance_mi_sorted = stereotype_importance_mi.sort_values(ascending=False)

    # Create a bar chart to visualize stereotype importance
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    ax = sns.barplot(x=stereotype_importance_mi_sorted.values, y=stereotype_importance_mi_sorted.index,
                     hue=stereotype_importance_mi_sorted.index, palette='viridis', dodge=False, legend=False)

    # Highlight specific stereotypes if needed
    color_text(ax.get_yticklabels())

    plt.title('Ranking of Stereotypes by Total Mutual Information', fontweight='bold')
    plt.xlabel('"Total Mutual Information')
    plt.ylabel('Stereotype')
    # Customize grid lines: Remove horizontal, keep vertical
    plt.grid(axis='y', linestyle='')  # Remove horizontal grid lines
    plt.grid(axis='x', linestyle='-', color='gray')  # Keep vertical grid lines
    # Set x-axis ticks dynamically based on data range
    plt.xticks(np.arange(0, stereotype_importance_mi_sorted.max() + 0.5, 0.5))
    fig_name = 'stereotype_ranking_mutual_information.png'
    fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
    plt.close(fig)

    # 4. Box Plot

    # Convert the DataFrame to a long format for the box plot
    df_long_boxplot = df.stack().reset_index()
    df_long_boxplot.columns = ['Stereotype 1', 'Stereotype 2', 'Mutual Information']

    # Remove self-correlations (mutual information of a stereotype with itself)
    df_long_boxplot = df_long_boxplot[df_long_boxplot['Stereotype 1'] != df_long_boxplot['Stereotype 2']]

    # Create a box plot to show the distribution of mutual information for each stereotype
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    ax = sns.boxplot(x='Stereotype 1', y='Mutual Information', data=df_long_boxplot, hue='Stereotype 1',
                     palette='viridis',
                     legend=False)
    plt.title('Box Plot of Mutual Information by Stereotype', fontweight='bold')
    plt.xlabel('Stereotype')
    plt.ylabel('Mutual Information')
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    # Customize x-axis label colors
    color_text(ax.get_xticklabels())

    fig_name = 'mutual_information_boxplot.png'
    fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
    plt.close(fig)
