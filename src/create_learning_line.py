import os

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.colors import Normalize
import seaborn as sns


# Step 1: Read the matrix from the CSV file
def read_matrix(file_path):
    df = pd.read_csv(file_path, index_col=0)
    return df


# Step 2: Find the highest value in the matrix
def find_highest_value(df):
    max_value = df.to_numpy().max()
    return max_value


# Step 3: Find the element with the highest sum of all Ns
def find_highest_sum_element(df):
    row_sums = df.sum(axis=1)  # Sum across rows
    highest_sum_element = row_sums.idxmax()  # Get index of the highest sum
    return highest_sum_element


# Step 4: Recursively find highest values and create graph edges
def create_graph(df):
    G = nx.Graph()  # Create a new graph

    # # Step 5A: Start with the element with the highest sum
    # start_element = find_highest_sum_element(df)

    # Step 5B: Select a starting element
    start_element = 'kind'

    visited = set()  # To track visited elements
    visited.add(start_element)

    # Recursive function to find the highest related element and create nodes/edges
    def add_edges_from_element(current_element):
        row_values = df.loc[current_element]  # Get the row for the current element
        unvisited_neighbors = row_values[~row_values.index.isin(visited)]  # Filter unvisited elements

        if unvisited_neighbors.empty:
            return

        # Find the highest value among unvisited neighbors
        next_element = unvisited_neighbors.idxmax()
        visited.add(next_element)

        # Add edge to the graph
        G.add_edge(current_element, next_element, weight=row_values[next_element])

        # Recursively proceed to the next element
        add_edges_from_element(next_element)

    # Start the recursion
    add_edges_from_element(start_element)

    return G


# Step 6: Plot the graph with colored edges using seaborn
def plot_graph(G):
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)  # Increased figure width for more space

    pos = nx.circular_layout(G)

    # Get the edge weights to determine the coloring
    edge_weights = nx.get_edge_attributes(G, 'weight')
    edges = G.edges()

    # Normalize the weights to [0,1] for color mapping
    weights = list(edge_weights.values())
    norm = Normalize(vmin=min(weights), vmax=max(weights))

    # Use seaborn's color palette
    palette = sns.color_palette("viridis_r", as_cmap=True)

    # Map the edge weights to colors
    edge_colors = [palette(norm(weight)) for weight in weights]

    # Draw the graph with seaborn-styled colored edges
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold', ax=ax)

    # Draw edges with the corresponding colors
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=2, ax=ax)

    # Create a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    sm.set_array([])  # Set empty array for the ScalarMappable

    # Add a colorbar to represent the weight values
    plt.colorbar(sm, ax=ax, label="Edge Weights")

    # Add the title to the plot
    plt.title("Mutual Information Network of Key Elements", fontweight='bold')

    plt.tight_layout()
    fig_name = 'mutual_information_network.png'
    fig.savefig(os.path.join(".", fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved.")
    plt.close(fig)



# Main function to run the script
if __name__ == "__main__":
    file_path = "../outputs/analyses/cs_analyses/mutual_information.csv"
    df = read_matrix(file_path)
    df.drop(columns=['other','none'], axis=1)


    # Find and print the highest value in the matrix
    max_value = find_highest_value(df)

    # Create the graph and plot it
    graph = create_graph(df)
    plot_graph(graph)
