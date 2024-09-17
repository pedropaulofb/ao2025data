import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from src.create_figure_subdir import create_figures_subdir


def execute_visualization_group72(file_path):
    # Load the data from CSV file
    df = pd.read_csv(file_path, index_col=0)
    save_dir = create_figures_subdir(file_path)

    # 1. Heatmap

    # Create a heatmap using Seaborn
    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax = sns.heatmap(df, annot=True, fmt=".2f", cmap='viridis', linewidths=.5)

    # Customize tick label colors
    for label in ax.get_xticklabels():
        if label.get_text() == 'none':
            label.set_color('blue')
        elif label.get_text() == 'other':
            label.set_color('red')

    for label in ax.get_yticklabels():
        if label.get_text() == 'none':
            label.set_color('blue')
        elif label.get_text() == 'other':
            label.set_color('red')

    # Customize the plot
    plt.title('Heatmap of Construct Interrelations')
    plt.xlabel('Constructs')
    plt.ylabel('Constructs')

    fig_name = 'group72_fig1.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")

    # 2. Network Graph

    # Convert the DataFrame to a long format
    df_long = df.stack().reset_index()
    df_long.columns = ['Construct 1', 'Construct 2', 'Mutual Information']

    # Remove self-correlations (correlation of a construct with itself)
    df_long = df_long[df_long['Construct 1'] != df_long['Construct 2']]

    # Drop duplicate pairs (e.g., both (A, B) and (B, A))
    df_long['Construct Pair'] = df_long.apply(lambda row: tuple(sorted([row['Construct 1'], row['Construct 2']])),
                                              axis=1)
    df_long = df_long.drop_duplicates(subset='Construct Pair')

    # Select the N most mutual information values
    N = 10
    top_mutual = df_long.nlargest(N, 'Mutual Information')

    # Select the N least mutual information values
    least_mutual = df_long.nsmallest(N, 'Mutual Information')

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
        fig, ax = plt.subplots(figsize=(12, 8), tight_layout=True)

        # Draw nodes and edges
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color=weights,
                edge_cmap=palette, width=3, edge_vmin=min(weights), edge_vmax=max(weights), ax=ax)

        # Add a color bar
        sm = plt.cm.ScalarMappable(cmap=palette, norm=plt.Normalize(vmin=min(weights), vmax=(max(weights))))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)  # Associate the color bar with the current axis
        cbar.set_label(metric_label)

        plt.title(title)
        fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
        logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")

    # Plot network for the N most mutual information values
    plot_network(top_mutual, f'Network Graph for Top {N} Most Mutual Information', 'Mutual Information',
                 palette=plt.cm.autumn_r, fig_name='group72_fig2.png')

    # Plot network for the N least mutual information values
    plot_network(least_mutual, f'Network Graph for Top {N} Least Mutual Information', 'Mutual Information',
                 palette=plt.cm.winter, fig_name='group72_fig3.png')

    # 3. Rank Chart

    # Calculate the sum of the mutual information values for each construct
    construct_importance_mi = df.sum(axis=1)

    # Sort the constructs by their importance in descending order
    construct_importance_mi_sorted = construct_importance_mi.sort_values(ascending=False)

    # Create a bar chart to visualize construct importance
    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax = sns.barplot(x=construct_importance_mi_sorted.values, y=construct_importance_mi_sorted.index,
                     hue=construct_importance_mi_sorted.index, palette='viridis', dodge=False, legend=False)

    # Highlight specific constructs if needed
    for label in ax.get_yticklabels():
        if label.get_text() == 'none':
            label.set_color('blue')
        elif label.get_text() == 'other':
            label.set_color('red')

    plt.title('Construct Importance Ranking Based on Total Mutual Information')
    plt.xlabel('Sum of Mutual Information Values')
    plt.ylabel('Construct')
    # Customize grid lines: Remove horizontal, keep vertical
    plt.grid(axis='y', linestyle='')  # Remove horizontal grid lines
    plt.grid(axis='x', linestyle='-', color='gray')  # Keep vertical grid lines
    # Set x-axis ticks dynamically based on data range
    plt.xticks(np.arange(0, construct_importance_mi_sorted.max() + 0.5, 0.5))
    fig_name = 'group72_fig4.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")

    # 4. Box Plot

    # Convert the DataFrame to a long format
    df_long = df.stack().reset_index()
    df_long.columns = ['Construct 1', 'Construct 2', 'Mutual Information']

    # Remove self-correlations (mutual information of a construct with itself)
    df_long = df_long[df_long['Construct 1'] != df_long['Construct 2']]

    # Create a box plot to show the distribution of mutual information for each construct
    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax = sns.boxplot(x='Construct 1', y='Mutual Information', data=df_long, hue='Construct 1', palette='viridis',
                     legend=False)
    plt.title('Distribution of Mutual Information for Each Construct')
    plt.xlabel('Construct')
    plt.ylabel('Mutual Information')
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    # Customize x-axis label colors
    for label in ax.get_xticklabels():
        if label.get_text() == 'none':
            label.set_color('blue')
        elif label.get_text() == 'other':
            label.set_color('red')

    fig_name = 'group72_fig5.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")


execute_visualization_group72('../outputs/analyses/cs_analyses/mutual_information.csv')
