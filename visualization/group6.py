import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

# Read data from the CSV file
file_path = '../outputs/analyses/cs_analyses/similarity_measures.csv'
df = pd.read_csv(file_path)

# Extract constructs from the 'Construct Pair' column
df['Construct 1'] = df['Construct Pair'].apply(lambda x: eval(x)[0])
df['Construct 2'] = df['Construct Pair'].apply(lambda x: eval(x)[1])

# 1. Heatmap

# Create a pivot table for Jaccard Similarity
jaccard_pivot = df.pivot_table(index='Construct 1', columns='Construct 2', values='Jaccard Similarity')

# Mirror the matrix to ensure symmetry
jaccard_pivot = jaccard_pivot.combine_first(jaccard_pivot.T)

# Fill diagonal with 1s for Jaccard Similarity
for construct in jaccard_pivot.index:
    jaccard_pivot.loc[construct, construct] = 1

# Plot heatmap for Jaccard Similarity
plt.figure(figsize=(12, 8))
sns.heatmap(jaccard_pivot, annot=True, cmap='viridis', cbar_kws={'label': 'Jaccard Similarity'})
plt.title('Heatmap of Jaccard Similarity Between Construct Pairs')
plt.show()

# Create a pivot table for Dice Coefficient
dice_pivot = df.pivot_table(index='Construct 1', columns='Construct 2', values='Dice Coefficient')

# Mirror the matrix to ensure symmetry
dice_pivot = dice_pivot.combine_first(dice_pivot.T)

# Fill diagonal with 1s for Dice Coefficient
for construct in dice_pivot.index:
    dice_pivot.loc[construct, construct] = 1

# Plot heatmap for Dice Coefficient
plt.figure(figsize=(12, 8))
sns.heatmap(dice_pivot, annot=True, cmap='viridis', cbar_kws={'label': 'Dice Coefficient'})
plt.title('Heatmap of Dice Coefficient Between Construct Pairs')
plt.show()

# 2. Network Graph

# Number of top pairs to visualize
N = 15

# Select top N pairs for Jaccard Similarity
top_jaccard = df.nlargest(N, 'Jaccard Similarity')

# Select top N pairs for Dice Coefficient
top_dice = df.nlargest(N, 'Dice Coefficient')


# Function to create a network graph with a legend
def plot_network(data, title, metric_label):
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
            edge_cmap=plt.cm.viridis, width=3, edge_vmin=min(weights), edge_vmax=max(weights), ax=ax)

    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(weights), vmax=max(weights)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)  # Associate the color bar with the current axis
    cbar.set_label(metric_label)

    plt.title(title)
    plt.show()


# Plot network for Jaccard Similarity
plot_network(top_jaccard, f'Network Graph for Top {N} Pairs by Jaccard Similarity', 'Jaccard Similarity')

# Plot network for Dice Coefficient
plot_network(top_dice, f'Network Graph for Top {N} Pairs by Dice Coefficient', 'Dice Coefficient')

# 3. Scatter plot
plt.figure(figsize=(10, 6))
scatter_plot = sns.scatterplot(data=df, x='Jaccard Similarity', y='Dice Coefficient', hue='Construct Pair',
    # Different colors for different construct pairs
    palette='viridis',  # Color palette for the scatter plot
    markers=True,  # Use different markers for clarity
    s=100,  # Size of the markers
    legend=False  # Remove the legend for each point
)

# Customize the plot
plt.title('Scatter Plot: Jaccard Similarity vs. Dice Coefficient')
plt.xlabel('Jaccard Similarity')
plt.ylabel('Dice Coefficient')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()

# 4. Box Plot

# Create a mirrored dataframe for the missing (y, x) pairs
mirror_df = df.copy()
mirror_df['Construct 1'], mirror_df['Construct 2'] = df['Construct 2'], df['Construct 1']

# Concatenate the original and mirrored dataframes
full_df = pd.concat([df, mirror_df], ignore_index=True)

# Create box plots grouped by the first construct
plt.figure(figsize=(12, 8))
sns.boxplot(data=full_df, x='Construct 1', y='Jaccard Similarity', hue='Construct 1', palette='viridis', legend=False)
plt.title('Box Plot of Jaccard Similarity Grouped by Construct')
plt.xlabel('Construct')
plt.ylabel('Jaccard Similarity')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(data=full_df, x='Construct 1', y='Dice Coefficient', hue='Construct 1', palette='viridis', legend=False)
plt.title('Box Plot of Dice Coefficient Grouped by Construct')
plt.xlabel('Construct')
plt.ylabel('Dice Coefficient')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.grid(True)
plt.tight_layout()
plt.show()
