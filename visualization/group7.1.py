import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
file_path = '../outputs/analyses/cs_analyses/spearman_correlation.csv'
df = pd.read_csv(file_path, index_col=0)

# 1. Heatmap

# Plotting the heatmap
plt.figure(figsize=(12, 10))
ax = sns.heatmap(df, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=0.5)
plt.title('Heatmap of the Correlation Matrix')

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

plt.xticks(rotation=45, ha='right')  # Adjust the rotation and alignment for x-axis labels
plt.yticks(rotation=0)  # Adjust the rotation for y-axis labels
plt.show()


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
def plot_network(data, title, metric_label, palette):
    # Create an empty graph
    G = nx.Graph()

    # Add edges to the graph
    for _, row in data.iterrows():
        G.add_edge(row['Construct 1'], row['Construct 2'], weight=row[metric_label])

    # Draw the graph
    pos = nx.circular_layout(G)
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

# Plot network for the N most positive Spearman Correlations
plot_network(top_positive, f'Network Graph for Top {N} Most Positive Spearman Correlations', 'Spearman Correlation', palette=plt.cm.autumn_r)

# Plot network for the N most negative Spearman Correlations
plot_network(top_negative, f'Network Graph for Top {N} Most Negative Spearman Correlations', 'Spearman Correlation', palette=plt.cm.winter)

# Plot network for the N correlations closest to zero
plot_network(near_zero, f'Network Graph for Top {N} Correlations Closest to Zero', 'Spearman Correlation', palette=plt.cm.summer)


# 3. Ranked Chart

# Calculate the sum of the absolute correlation values for each construct
construct_importance = df.abs().sum(axis=1)

# Sort the constructs by their importance in descending order
construct_importance_sorted = construct_importance.sort_values(ascending=False)

# Create a bar chart to visualize construct importance
plt.figure(figsize=(10, 8))
ay = sns.barplot(x=construct_importance_sorted.values, y=construct_importance_sorted.index,
            hue=construct_importance_sorted.index, palette='viridis', dodge=False, legend=False)

for label in ay.get_yticklabels():
    if label.get_text() == 'none':
        label.set_color('blue')
    elif label.get_text() == 'other':
        label.set_color('red')

plt.title('Construct Importance Ranking Based on Total Absolute Correlations')
plt.xlabel('Sum of Absolute Correlation Values')
plt.ylabel('Construct')
# Customize grid lines: Remove horizontal, keep vertical
plt.grid(axis='y', linestyle='')  # Remove horizontal grid lines
plt.grid(axis='x', linestyle='-', color='gray')  # Keep vertical grid lines
# Set x-axis ticks to be every 0.5 units
plt.xticks(np.arange(0, construct_importance_sorted.max() + 0.5, 0.5))
plt.show()