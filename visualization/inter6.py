import ast
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

from src.create_figure_subdir import create_figures_subdir


def execute_visualization_mutual_info_vs_jaccard_similarity(file_path1, file_path2):
    # Load the data from the CSV files
    mutual_info_df = pd.read_csv(file_path1)
    similarity_measures_df = pd.read_csv(file_path2)
    save_dir = create_figures_subdir(file_path1)

    # Prepare a list for DataFrame creation
    pairs_data = {
        'Construct Pair': [],
        'Jaccard Similarity': [],
        'Mutual Information': []
    }

    # Iterate through the similarity measures dataframe to extract construct pairs and their similarity measures
    for i, row in similarity_measures_df.iterrows():
        construct_pair = ast.literal_eval(row['Construct Pair'])
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
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    sns.scatterplot(data=pairs_df,
                    x='Jaccard Similarity',
                    y='Mutual Information',
                    hue='Construct Pair',
                    palette='viridis',
                    s=100,
                    legend=False)

    # Customize the plot
    plt.title('Comparison of Mutual Information and Jaccard Similarity for Construct Pairs', fontweight='bold')
    plt.xlabel('Jaccard Similarity (Construct Pair)')
    plt.ylabel('Mutual Information (Construct Pair)')
    plt.grid(True)

    # Draw lines to create quadrants using medians
    plt.axhline(y=mutual_info_median, color='grey', linestyle='--',
                linewidth=1)  # Horizontal line for Mutual Information
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
    plt.text(x=max(pairs_df['Jaccard Similarity'])/2, y=mutual_info_median, s=f'Median Mutual Information: {mutual_info_median:.2f}', color='grey',
             fontsize=10,
             va='bottom', ha='left')
    plt.text(x=jaccard_median, y=max(pairs_df['Mutual Information']/2),
             s=f'Median Jaccard Similarity: {jaccard_median:.2f}',
             color='grey', fontsize=10, va='bottom', ha='right', rotation=90)

    fig_name = 'mutual_info_vs_jaccard_similarity.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")
    plt.close(fig)


execute_visualization_mutual_info_vs_jaccard_similarity('../outputs/analyses/cs_analyses/mutual_information.csv',
                             '../outputs/analyses/cs_analyses/similarity_measures.csv')
