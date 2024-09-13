import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from CSV file
file_path = '../outputs/analyses/cs_analyses/mutual_information.csv'
df = pd.read_csv(file_path, index_col=0)

# 1. Heatmap

# Create a heatmap using Seaborn
plt.figure(figsize=(12, 10))
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

# Show the plot
plt.show()

# 2. Network Graph

# Convert the DataFrame to a long format
df_long = df.stack().reset_index()
df_long.columns = ['Construct 1', 'Construct 2', 'Mutual Information']

# Remove self-correlations (correlation of a construct with itself)
df_long = df_long[df_long['Construct 1'] != df_long['Construct 2']]

# Drop duplicate pairs (e.g., both (A, B) and (B, A))
df_long['Construct Pair'] = df_long.apply(lambda row: tuple(sorted([row['Construct 1'], row['Construct 2']])), axis=1)
df_long = df_long.drop_duplicates(subset='Construct Pair')

# Select the N most mutual information values
N = 10
top_mutual = df_long.nlargest(N, 'Mutual Information')

# Select the N least mutual information values
least_mutual = df_long.nsmallest(N, 'Mutual Information')

# Function to create a network graph with a legend
def plot_network(data, title, metric_label, palette):
    # Create an empty graph
    G = nx.Graph()

    # Add edges to the graph
    for _, row in data.iterrows():
        G.add_edge(row['Construct 1'], row['Construct 2'], weight=row[metric_label])

    # Draw the graph
    pos = nx.circular_layout(G)  # Use spring layout for better visualization
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color=weights,
            edge_cmap=palette, width=3, edge_vmin=min(weights), edge_vmax=max(weights), ax=ax)

    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap=palette, norm=plt.Normalize(vmin=min(weights), vmax=max(weights)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)  # Associate the color bar with the current axis
    cbar.set_label(metric_label)

    plt.title(title)
    plt.show()

# Plot network for the N most mutual information values
plot_network(top_mutual, f'Network Graph for Top {N} Most Mutual Information', 'Mutual Information', palette=plt.cm.autumn_r)

# Plot network for the N least mutual information values
plot_network(least_mutual, f'Network Graph for Top {N} Least Mutual Information', 'Mutual Information', palette=plt.cm.winter)

# 3. Rank Chart

# Calculate the sum of the mutual information values for each construct
construct_importance_mi = df.sum(axis=1)

# Sort the constructs by their importance in descending order
construct_importance_mi_sorted = construct_importance_mi.sort_values(ascending=False)

# Create a bar chart to visualize construct importance
plt.figure(figsize=(10, 8))
ax = sns.barplot(x=construct_importance_mi_sorted.values, y=construct_importance_mi_sorted.index,
                 hue=construct_importance_mi_sorted.index, palette='viridis', dodge=False)

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
plt.show()

# 4. Box Plot

# Convert the DataFrame to a long format
df_long = df.stack().reset_index()
df_long.columns = ['Construct 1', 'Construct 2', 'Mutual Information']

# Remove self-correlations (mutual information of a construct with itself)
df_long = df_long[df_long['Construct 1'] != df_long['Construct 2']]

# Create a box plot to show the distribution of mutual information for each construct
plt.figure(figsize=(12, 8))
ax = sns.boxplot(x='Construct 1', y='Mutual Information', data=df_long, palette='viridis')
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

plt.show()