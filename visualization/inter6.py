import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from the CSV files
mutual_info_df = pd.read_csv('../outputs/analyses/cs_analyses/mutual_information.csv')
similarity_measures_df = pd.read_csv('../outputs/analyses/cs_analyses/similarity_measures.csv')

# Prepare a list for DataFrame creation
pairs_data = {
    'Construct Pair': [],
    'Jaccard Similarity': [],
    'Mutual Information': []
}

# Iterate through the similarity measures dataframe to extract construct pairs and their similarity measures
for i, row in similarity_measures_df.iterrows():
    construct_pair = eval(row['Construct Pair'])  # Convert string representation to a tuple
    construct1, construct2 = construct_pair

    # Retrieve mutual information for each pair
    mi_value = mutual_info_df.loc[mutual_info_df['Construct'] == construct1, construct2].values
    if len(mi_value) > 0:
        pairs_data['Construct Pair'].append(f"{construct1} - {construct2}")
        pairs_data['Jaccard Similarity'].append(row['Jaccard Similarity'])
        pairs_data['Mutual Information'].append(mi_value[0])

# Convert to DataFrame
pairs_df = pd.DataFrame(pairs_data)

# Save the results to a CSV file
output_file_path = './construct_pairs_similarity_info.csv'
pairs_df.to_csv(output_file_path, index=False)
print(f"CSV file saved to {output_file_path}")

# Calculate thresholds (medians)
jaccard_median = pairs_df['Jaccard Similarity'].median()
mutual_info_median = pairs_df['Mutual Information'].median()

# Define thresholds for significant points
jaccard_threshold = jaccard_median * 1.2  # Adjust this threshold as needed
mutual_info_threshold = mutual_info_median * 1.2  # Adjust this threshold as needed

# Create a scatter plot for Mutual Information vs. Jaccard Similarity
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pairs_df,
                x='Jaccard Similarity',
                y='Mutual Information',
                hue='Construct Pair',
                palette='viridis',
                s=100,
                legend=False)

# Customize the plot
plt.title('Scatter Plot of Mutual Information vs. Jaccard Similarity')
plt.xlabel('Jaccard Similarity')
plt.ylabel('Mutual Information')
plt.grid(True)

# Draw lines to create quadrants using medians
plt.axhline(y=mutual_info_median, color='grey', linestyle='--', linewidth=1)  # Horizontal line for Mutual Information
plt.axvline(x=jaccard_median, color='grey', linestyle='--', linewidth=1)  # Vertical line for Jaccard Similarity

# # Specify the pairs to annotate
# pairs_to_annotate = ['abstract - category', 'category - kind', 'role - subkind']  # Add desired pairs here
#
# # Annotate only specific points
# for i in range(len(pairs_df)):
#     construct_pair = pairs_df['Construct Pair'][i]
#     if construct_pair in pairs_to_annotate:  # Check if the pair is in the list of pairs to annotate
#         color = 'black'  # Default color
#         if 'none' in construct_pair:
#             color = 'blue'
#         elif 'other' in construct_pair:
#             color = 'red'
#         plt.text(x=pairs_df['Jaccard Similarity'][i],
#                  y=pairs_df['Mutual Information'][i],
#                  s=construct_pair,
#                  fontsize=8,
#                  ha='right',
#                  color=color)  # Use the color variable here

# Add annotations for the median lines
plt.text(x=0, y=mutual_info_median, s=f'Median Mutual Information: {mutual_info_median:.2f}', color='grey', fontsize=10,
         va='bottom', ha='left')
plt.text(x=jaccard_median, y=min(pairs_df['Mutual Information']), s=f'Median Jaccard Similarity: {jaccard_median:.2f}',
         color='grey', fontsize=10, va='bottom', ha='right', rotation=90)

# Show the plot
plt.show()
