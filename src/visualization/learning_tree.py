import os

import networkx as nx
import numpy as np
import pandas as pd
import pydot
from loguru import logger


def load_csv(filename):
    # Load the CSV into a pandas dataframe
    df = pd.read_csv(filename, index_col=0)
    return df


def build_tree(df, root, tolerance):
    tree = nx.DiGraph()  # Create a directed graph
    selected = {root}  # Track selected nodes to avoid circular paths
    tree.add_node(root)  # Add root node
    build_branches(df, root, selected, tree, tolerance)
    return tree


def build_branches(df, current, selected, tree, tolerance):
    if df.empty or current not in df.index:
        return df  # Stop recursion if the DataFrame is empty or if the node is missing

    # Sort the values in descending order by absolute value for the current element
    sorted_relations = df.loc[current].abs().sort_values(ascending=False)

    # Get the maximum value excluding the current node's relationship with itself
    sorted_relations = sorted_relations[sorted_relations.index != current]
    if sorted_relations.empty:
        return df

    max_value = sorted_relations.iloc[0]  # Use the next highest value

    # Select children with values at most `tolerance` lower than the highest value
    children = sorted_relations[sorted_relations >= max_value - tolerance].index

    for child in children:
        if child not in selected and child in df.index:
            # Add the edge to the tree and mark the child as selected
            tree.add_edge(current, child)
            selected.add(child)

    # After linking the current node to its children, remove its entire row and column
    df = df.drop(columns=current)  # Remove the current column
    df = df.drop(index=current)  # Remove the current row

    # Now recursively process the child nodes
    for child in children:
        if child in df.index:  # Make sure the child hasn't been removed yet
            df = build_branches(df, child, selected, tree, tolerance)

    return df


def generate_dot(tree, output_file_path):
    output_file = os.path.join(output_file_path, f"learning_tree")
    output_file = output_file.replace(".csv", ".dot")

    # Set the title of the graph using graph attributes
    pydot_graph = nx.drawing.nx_pydot.to_pydot(tree)
    pydot_graph.set_label("Learning Tree")
    pydot_graph.set_labelloc("t")  # Place the title at the top
    pydot_graph.set_labeljust("c")  # Center the title

    # Write the graph to a DOT file
    pydot_graph.write(output_file)
    logger.success(f"Learning tree DOT file generated: {output_file}")

    # Generate PNG from the DOT file
    (graph,) = pydot.graph_from_dot_file(output_file)
    graph.write_png(output_file.replace(".dot", ".png"))
    logger.success(f"Learning tree PNG file generated: {output_file.replace('.dot', '.png')}")


def select_root_node(df):
    # Extract the diagonal values (where row and column indices are the same)
    diagonal_values = np.diag(df.values)

    # Find the index of the maximum value in the diagonal
    max_value_idx = diagonal_values.argmax()

    # Return the node (row/column label) corresponding to the maximum value
    root_node = df.index[max_value_idx]
    return root_node


