import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from icecream import ic
from loguru import logger
from matplotlib.colors import Normalize


# Step 2: Find the highest value based on the absolute value (module) of the items
def find_highest_absolute_value(df):
    max_value = df.to_numpy().max()  # This is your current code that finds the max value
    # To find the highest value based on the absolute value (module):
    max_value_by_module = np.abs(df.to_numpy()).max()  # Find the maximum of the absolute values
    return max_value_by_module


# Step 3: Find the element with the highest sum of all Ns
def find_highest_sum_element(df):
    row_sums = df.sum(axis=1)  # Sum across rows
    highest_sum_element = row_sums.idxmax()  # Get index of the highest sum
    return highest_sum_element


# Step 4: Recursively find highest values and create graph edges
def create_graph(df):
    G = nx.Graph()  # Create a new graph

    start_element = find_highest_sum_element(df)
    if pd.isna(start_element):
        logger.error("Starting element is NaN. Cannot create graph.")
        return None

    visited = set()  # To track visited elements
    visited.add(start_element)

    def add_edges_from_element(current_element):
        row_values = df.loc[current_element]
        unvisited_neighbors = row_values[~row_values.index.isin(visited)]

        if unvisited_neighbors.empty or unvisited_neighbors.isna().all():
            return

        next_element = unvisited_neighbors.idxmax()

        if pd.isna(next_element):
            return

        visited.add(next_element)
        G.add_edge(current_element, next_element, weight=row_values[next_element])
        add_edges_from_element(next_element)

    add_edges_from_element(start_element)

    if len(G.nodes) == 0:
        logger.error("Graph has no nodes or edges.")
        return None

    return G



# Step 6: Plot the graph with colored edges using seaborn
def create_learning_line(graph, file_path, out_dir_path,metric_name):
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)  # Increased figure width for more space

    pos = nx.circular_layout(graph)

    # Get the edge weights to determine the coloring
    edge_weights = nx.get_edge_attributes(graph, 'weight')
    edges = graph.edges()

    # Normalize the weights to [0,1] for color mapping
    weights = list(edge_weights.values())
    norm = Normalize(vmin=min(weights), vmax=max(weights))

    # Use seaborn's color palette
    palette = sns.color_palette("viridis_r", as_cmap=True)

    # Map the edge weights to colors
    edge_colors = [palette(norm(weight)) for weight in weights]

    # Draw the graph with seaborn-styled colored edges
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold',
            ax=ax)

    # Draw edges with the corresponding colors
    nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color=edge_colors, width=2, ax=ax)

    # Create a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    sm.set_array([])  # Set empty array for the ScalarMappable

    # Add a colorbar to represent the weight values
    plt.colorbar(sm, ax=ax, label="Edge Weights")

    formatted_metric_name = metric_name.replace('_', ' ').title()

    # Add the title to the plot
    plt.title(f"{formatted_metric_name} Network of Key Elements", fontweight='bold')

    plt.tight_layout()
    fig_name = f'learning_line_{metric_name}.png'
    fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
    plt.close(fig)


# Main function to run the script
def execute_learning_line(in_dir_path, out_dir_path, file_name):

    file_path = os.path.join(in_dir_path, file_name)
    df = pd.read_csv(file_path, index_col=0)

    # Create the graph
    graph = create_graph(df)

    # Check if the graph is None or empty
    if graph is None or len(graph.nodes) == 0:
        logger.error(f"Graph creation failed or graph is empty for {file_path}. Skipping visualization.")
        return

    # Create the visualization
    create_learning_line(graph, file_path, out_dir_path, file_name)
