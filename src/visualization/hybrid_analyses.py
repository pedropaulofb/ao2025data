import ast
import os

import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from src.utils import color_text


def execute_visualization_mutual_info_vs_jaccard_similarity(input_dir, output_dir, file1_name, file2_name):
    # Load the data from the CSV files
    mutual_info_df = pd.read_csv(os.path.join(input_dir, file1_name))
    similarity_measures_df = pd.read_csv(os.path.join(input_dir, file2_name))

    # Prepare a list for DataFrame creation
    pairs_data = {'Stereotype Pair': [], 'Jaccard Similarity': [], 'Mutual Information': []}

    # Iterate through the similarity measures dataframe to extract stereotype pairs and their similarity measures
    for i, row in similarity_measures_df.iterrows():
        stereotype_pair = ast.literal_eval(row['Stereotype Pair'])
        stereotype1, stereotype2 = stereotype_pair

        # Retrieve mutual information for each pair
        mi_value = mutual_info_df.loc[mutual_info_df['Stereotype'] == stereotype1, stereotype2].values
        if len(mi_value) > 0:
            pairs_data['Stereotype Pair'].append(f"{stereotype1} - {stereotype2}")
            pairs_data['Jaccard Similarity'].append(row['Jaccard Similarity'])
            pairs_data['Mutual Information'].append(mi_value[0])

    # Convert to DataFrame
    pairs_df = pd.DataFrame(pairs_data)

    # Save the results to a CSV file
    output_file_path = os.path.join(output_dir, 'stereotype_pairs_jaccard_similarity_info.csv')
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
    sns.scatterplot(data=pairs_df, x='Jaccard Similarity', y='Mutual Information', hue='Stereotype Pair',
                    palette='viridis', s=100, legend=False)

    # Customize the plot
    plt.title('Comparison of Mutual Information and Jaccard Similarity for Stereotype Pairs', fontweight='bold')
    plt.xlabel('Jaccard Similarity (Stereotype Pair)')
    plt.ylabel('Mutual Information (Stereotype Pair)')
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
    #     stereotype_pair = pairs_df['Stereotype Pair'][i]
    #     if stereotype_pair in pairs_to_annotate:  # Check if the pair is in the list of pairs to annotate
    #         color = 'black'  # Default color
    #         if 'none' in stereotype_pair:
    #             color = 'blue'
    #         elif 'other' in stereotype_pair:
    #             color = 'red'
    #         plt.text(x=pairs_df['Jaccard Similarity'][i],
    #                  y=pairs_df['Mutual Information'][i],
    #                  s=stereotype_pair,
    #                  fontsize=8,
    #                  ha='right',
    #                  color=color)  # Use the color variable here

    # Add annotations for the median lines
    plt.text(x=max(pairs_df['Jaccard Similarity']) / 2, y=mutual_info_median,
             s=f'Median Mutual Information: {mutual_info_median:.2f}', color='grey', fontsize=10, va='bottom',
             ha='left')
    plt.text(x=jaccard_median, y=max(pairs_df['Mutual Information'] / 2),
             s=f'Median Jaccard Similarity: {jaccard_median:.2f}', color='grey', fontsize=10, va='bottom', ha='right',
             rotation=90)

    fig_name = 'mutual_info_vs_jaccard_similarity.png'
    fig.savefig(os.path.join(output_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {output_dir}.")
    plt.close(fig)


def execute_visualization_simpson_diversity_vs_stereotype_frequency(input_dir, output_dir, file1_name, file2_name):
    # Load the data from the CSV files
    diversity_measures = pd.read_csv(os.path.join(input_dir, file1_name))
    frequency_analysis = pd.read_csv(os.path.join(input_dir, file2_name))

    # Merge the dataframes on the 'Stereotype' column
    merged_data = pd.merge(diversity_measures[['Stereotype', 'Simpson Index']],
                           frequency_analysis[['Stereotype', 'Total Frequency']], on='Stereotype')

    # Calculate thresholds (medians)
    frequency_threshold = merged_data['Total Frequency'].median()  # Median for Total Frequency
    simpson_threshold = merged_data['Simpson Index'].median()  # Median for Simpson Index

    # Create a scatter plot for Simpson's Index vs. Total Frequency
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    sns.scatterplot(data=merged_data, x='Total Frequency', y='Simpson Index', hue='Stereotype', palette='viridis',
                    s=100,
                    legend=False)

    # Customize the plot
    plt.title("Simpson's Diversity Index vs. Total Stereotype Frequency", fontweight='bold')
    plt.xlabel('Total Stereotype Frequency')
    plt.ylabel("Simpson's Diversity Index")
    plt.grid(True)

    # Draw lines to create quadrants using medians
    plt.axhline(y=simpson_threshold, color='grey', linestyle='--', linewidth=1)  # Horizontal line for Simpson Index
    plt.axvline(x=frequency_threshold, color='grey', linestyle='--', linewidth=1)  # Vertical line for Total Frequency

    # Create an empty list to store text objects
    texts = []

    # Annotate each point with the stereotype name
    for i in range(len(merged_data)):
        stereotype_name = merged_data['Stereotype'][i]

        # Create a text object without setting color initially
        text = plt.text(x=merged_data['Total Frequency'][i], y=merged_data['Simpson Index'][i] + 0.01,
                        s=stereotype_name,
                        fontsize=8, ha='center', color='black')  # Initial color set to black or any default

        # Append the text object to the list
        texts.append(text)

    # Apply the color_text function to the list of text objects
    color_text(texts)

    # Adjust text to avoid overlap
    adjust_text(texts)

    # Add annotations for the median lines
    plt.text(x=max(merged_data['Total Frequency']) / 2, y=simpson_threshold,
             s=f'Median Simpson Index: {simpson_threshold:.2f}', color='grey', fontsize=10, va='bottom', ha='left')
    plt.text(x=frequency_threshold, y=max(merged_data['Simpson Index']) / 2,
             s=f'Median Total Frequency: {frequency_threshold:.2f}', color='grey', fontsize=10, va='bottom', ha='right',
             rotation=90)

    fig_name = 'simpson_diversity_vs_stereotype_frequency.png'
    fig.savefig(os.path.join(output_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {output_dir}.")
    plt.close(fig)


def execute_visualization_mutual_information_vs_stereotype_rank(input_dir, output_dir, file1_name, file2_name):
    # Load the data from the CSV files
    rank_df = pd.read_csv(os.path.join(input_dir, file1_name))
    mutual_info_df = pd.read_csv(os.path.join(input_dir, file2_name))

    # Calculate the average mutual information for each stereotype
    mutual_info_avg = mutual_info_df.drop(columns='Stereotype').mean(axis=1)
    mutual_info_df['Avg Mutual Information'] = mutual_info_avg

    # Merge rank data with mutual information data
    merged_data = pd.merge(rank_df[['Stereotype', 'Rank']], mutual_info_df[['Stereotype', 'Avg Mutual Information']],
                           on='Stereotype')

    # Plot the scatter plot
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    sns.scatterplot(data=merged_data, x='Rank', y='Avg Mutual Information', hue='Stereotype', palette='viridis',
                    size='Avg Mutual Information', sizes=(20, 200), legend=False)

    # Customize the plot
    plt.title('Average Mutual Information vs. Stereotype Rank', fontweight='bold')
    plt.xlabel('Stereotype Rank')
    plt.ylabel('Mean Mutual Information')
    plt.grid(True)

    # Annotate each point with the stereotype name
    for i in range(len(merged_data)):
        plt.text(x=merged_data['Rank'][i], y=merged_data['Avg Mutual Information'][i] + 0.01,
                 s=merged_data['Stereotype'][i], fontsize=8, ha='center')

    # Show the plot
    fig_name = 'mutual_information_vs_stereotype_rank.png'
    fig.savefig(os.path.join(output_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {output_dir}.")
    plt.close(fig)


def execute_visualization_shannon_entropy_vs_global_frequency(input_dir, output_dir, file1_name, file2_name):
    # Load the data from the CSV files
    diversity_measures = pd.read_csv(os.path.join(input_dir, file1_name))
    frequency_analysis = pd.read_csv(os.path.join(input_dir, file2_name))

    # Merge the dataframes on the 'Stereotype' column
    merged_data = pd.merge(diversity_measures[['Stereotype', 'Shannon Entropy']],
                           frequency_analysis[['Stereotype', 'Global Relative Frequency (Occurrence-wise)']],
                           on='Stereotype')

    # Calculate thresholds
    entropy_threshold = merged_data['Shannon Entropy'].median()  # Using median as a threshold
    frequency_threshold = merged_data[
        'Global Relative Frequency (Occurrence-wise)'].median()  # Using median as a threshold

    # Plot with Seaborn scatter plot
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    sns.scatterplot(data=merged_data, x='Shannon Entropy', y='Global Relative Frequency (Occurrence-wise)',
                    hue='Stereotype', palette='viridis', size='Global Relative Frequency (Occurrence-wise)',
                    legend=False, sizes=(20, 200))

    # Customize the plot
    plt.yscale('log')
    plt.title('Shannon Entropy vs. Global Relative Frequency', fontweight='bold')
    plt.xlabel('Shannon Entropy')
    plt.ylabel('Log of Global Relative Frequency (Occurrence-wise)')
    plt.grid(True)

    # Draw lines to create quadrants
    plt.axhline(y=frequency_threshold, color='grey', linestyle='--', linewidth=1)  # Horizontal line
    plt.axvline(x=entropy_threshold, color='grey', linestyle='--', linewidth=1)  # Vertical line

    # Annotate each point with the stereotype name
    texts = []  # Initialize an empty list to store text objects
    for i in range(len(merged_data)):
        stereotype_name = merged_data['Stereotype'][i]

        # Create text object with default color (black)
        text = plt.text(x=merged_data['Shannon Entropy'][i],
                        y=merged_data['Global Relative Frequency (Occurrence-wise)'][i], s=stereotype_name, fontsize=8,
                        ha='right', color='black')
        texts.append(text)  # Append the text object to the list

    # Apply color_text function to set specific colors
    color_text(texts)

    # Add annotations for the median lines
    plt.text(x=(max(merged_data['Global Relative Frequency (Occurrence-wise)'])) / 2, y=frequency_threshold,
             s=f'Median Frequency: {frequency_threshold:.2e}', color='grey', fontsize=10, va='bottom', ha='left')
    plt.text(x=entropy_threshold, y=min(merged_data['Global Relative Frequency (Occurrence-wise)']),
             s=f'Median Entropy: {entropy_threshold:.2f}', color='grey', fontsize=10, va='bottom', ha='right',
             rotation=90)

    fig_name = 'shannon_entropy_vs_global_frequency.png'
    fig.savefig(os.path.join(output_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {output_dir}.")
    plt.close(fig)


def execute_visualization_shannon_entropy_vs_group_frequency_stereotypes(input_dir, output_dir, file1_name, file2_name):
    # Load the data from the CSV files
    diversity_measures = pd.read_csv(os.path.join(input_dir, file1_name))
    frequency_analysis = pd.read_csv(os.path.join(input_dir, file2_name))

    # Merge the dataframes on the 'Stereotype' column
    merged_data = pd.merge(diversity_measures[['Stereotype', 'Shannon Entropy']],
                           frequency_analysis[['Stereotype', 'Group Frequency']], on='Stereotype')

    # Calculate thresholds (medians)
    entropy_threshold = merged_data['Shannon Entropy'].median()  # Median for Shannon Entropy
    frequency_threshold = merged_data['Group Frequency'].median()  # Median for Group Frequency

    # Create a scatter plot for Shannon Entropy vs Group Frequency
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    sns.scatterplot(data=merged_data, x='Shannon Entropy', y='Group Frequency', hue='Stereotype', palette='viridis',
                    legend=False, s=100)

    # Customize the plot
    plt.title('Shannon Entropy vs. Group Frequency of Stereotypes', fontweight='bold')
    plt.xlabel('Shannon Entropy (Diversity Measure)')
    plt.ylabel('Stereotype Group Frequency')
    plt.grid(True)

    # Draw lines to create quadrants using medians
    plt.axhline(y=frequency_threshold, color='grey', linestyle='--', linewidth=1)  # Horizontal line for Group Frequency
    plt.axvline(x=entropy_threshold, color='grey', linestyle='--', linewidth=1)  # Vertical line for Shannon Entropy

    # Create an empty list to store text objects
    texts = []

    # Annotate each point with the stereotype name
    for i in range(len(merged_data)):
        stereotype_name = merged_data['Stereotype'][i]

        # Create a text object without setting color initially
        text = plt.text(x=merged_data['Shannon Entropy'][i], y=merged_data['Group Frequency'][i] + 1.2,
                        s=stereotype_name, fontsize=8, ha='center',
                        color='black')  # Set initial color to black or any default

        # Append the text object to the list
        texts.append(text)

    # Apply the color_text function to the list of text objects
    color_text(texts)

    # Add annotations for the median lines
    plt.text(x=0, y=frequency_threshold, s=f'Median Group Frequency: {frequency_threshold:.2f}', color='grey',
             fontsize=10, va='bottom', ha='left')
    plt.text(x=entropy_threshold, y=min(merged_data['Group Frequency'] + 50),
             s=f'Median Shannon Entropy: {entropy_threshold:.2f}', color='grey', fontsize=10, va='bottom', ha='right',
             rotation=90)

    fig_name = 'shannon_entropy_vs_group_frequency_stereotypes.png'
    fig.savefig(os.path.join(output_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {output_dir}.")
    plt.close(fig)


def execute_visualization_ubiquity_vs_gini_coefficient(input_dir, output_dir, file1_name, file2_name):
    # Load the data from the CSV files
    diversity_measures = pd.read_csv(os.path.join(input_dir, file1_name))
    frequency_analysis = pd.read_csv(os.path.join(input_dir, file2_name))

    # Merge the dataframes on the 'Stereotype' column
    merged_data = pd.merge(diversity_measures[['Stereotype', 'Gini Coefficient']],
                           frequency_analysis[['Stereotype', 'Ubiquity Index (Group Frequency per Group)']],
                           on='Stereotype')

    # Calculate thresholds
    gini_threshold = merged_data['Gini Coefficient'].median()  # Using median as a threshold
    ubiquity_threshold = merged_data[
        'Ubiquity Index (Group Frequency per Group)'].median()  # Using median as a threshold

    # Plot with Seaborn scatter plot
    fig = plt.figure(figsize=(16, 9))
    sns.scatterplot(data=merged_data, x='Gini Coefficient', y='Ubiquity Index (Group Frequency per Group)',
                    hue='Stereotype', palette='viridis', size='Ubiquity Index (Group Frequency per Group)',
                    legend=False,
                    sizes=(20, 200))

    # Customize the plot
    plt.title("Ubiquity Index vs. Gini Coefficient", fontweight='bold')
    plt.xlabel('Gini Coefficient (Inequality Measure)')
    plt.ylabel('Ubiquity Index (Frequency per Group)')
    plt.grid(True)

    # Draw lines to create quadrants
    plt.axhline(y=ubiquity_threshold, color='grey', linestyle='--', linewidth=1)  # Horizontal line
    plt.axvline(x=gini_threshold, color='grey', linestyle='--', linewidth=1)  # Vertical line

    # Annotate each point with the stereotype name
    texts = []  # Initialize an empty list to store text objects
    for i in range(len(merged_data)):
        stereotype_name = merged_data['Stereotype'][i]

        # Create text object with default color (black)
        text = plt.text(x=merged_data['Gini Coefficient'][i],
                        y=merged_data['Ubiquity Index (Group Frequency per Group)'][i] + 0.01, s=stereotype_name,
                        fontsize=8, ha='right', color='black')  # Initially set all colors to black
        texts.append(text)  # Append the text object to the list

    # Apply the color_text function to set specific colors
    color_text(texts)

    # Adjust text to avoid overlap
    adjust_text(texts)

    # Add annotations for the median lines
    plt.text(x=0, y=ubiquity_threshold, s=f'Median Ubiquity Index: {ubiquity_threshold:.2f}', color='grey', fontsize=10,
             va='bottom', ha='left')
    plt.text(x=gini_threshold, y=min(merged_data['Ubiquity Index (Group Frequency per Group)']),
             s=f'Median Gini Coefficient: {gini_threshold:.2f}', color='grey', fontsize=10, va='bottom', ha='right',
             rotation=90)

    fig_name = 'ubiquity_vs_gini_coefficient.png'
    fig.savefig(os.path.join(output_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {output_dir}.")
    plt.close(fig)


def execute_visualization_gini_coefficient_vs_global_frequency(input_dir, output_dir, file1_name, file2_name):
    # Load the data from the CSV files
    diversity_measures = pd.read_csv(os.path.join(input_dir, file1_name))
    frequency_analysis = pd.read_csv(os.path.join(input_dir, file2_name))

    # Merge the dataframes on the 'Stereotype' column
    merged_data = pd.merge(diversity_measures[['Stereotype', 'Gini Coefficient']],
                           frequency_analysis[['Stereotype', 'Global Relative Frequency (Occurrence-wise)']],
                           on='Stereotype')

    # Calculate thresholds
    gini_median = merged_data['Gini Coefficient'].median()  # Using median as a threshold
    frequency_threshold = merged_data[
        'Global Relative Frequency (Occurrence-wise)'].median()  # Using median as a threshold

    # Plot with Seaborn scatter plot
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    sns.scatterplot(data=merged_data, x='Gini Coefficient', y='Global Relative Frequency (Occurrence-wise)',
                    hue='Stereotype', palette='viridis', size='Global Relative Frequency (Occurrence-wise)',
                    legend=False, sizes=(20, 200))

    # Customize the plot
    plt.yscale('log')
    plt.title('Gini Coefficient vs. Global Relative Frequency', fontweight='bold')
    plt.xlabel('Gini Coefficient')
    plt.ylabel('Log of Global Relative Frequency (Occurrence-wise)')
    plt.grid(True)

    # Draw lines to create quadrants
    plt.axhline(y=frequency_threshold, color='grey', linestyle='--', linewidth=1)  # Horizontal line
    plt.axvline(x=gini_median, color='grey', linestyle='--', linewidth=1)  # Vertical line

    # Annotate each point with the stereotype name
    texts = []  # Initialize an empty list to store text objects
    for i in range(len(merged_data)):
        stereotype_name = merged_data['Stereotype'][i]

        # Create text object with default color (black)
        text = plt.text(x=merged_data['Gini Coefficient'][i],
                        y=merged_data['Global Relative Frequency (Occurrence-wise)'][i], s=stereotype_name, fontsize=8,
                        ha='right', color='black')
        texts.append(text)  # Append the text object to the list

    # Apply color_text function to set specific colors
    color_text(texts)

    # Add annotations for the median lines
    plt.text(x=merged_data['Gini Coefficient'].min(), y=frequency_threshold,
             s=f'Median Frequency: {frequency_threshold:.2e}', color='grey', fontsize=10, va='bottom', ha='left')
    plt.text(x=gini_median, y=merged_data['Global Relative Frequency (Occurrence-wise)'].min(),
             s=f'Median Gini Coefficient: {gini_median:.2f}', color='grey', fontsize=10, va='bottom', ha='right',
             rotation=90)

    fig_name = 'gini_coefficient_vs_global_frequency.png'
    fig.savefig(os.path.join(output_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {output_dir}.")
    plt.close(fig)


def execute_visualization_gini_coefficient_vs_group_frequency_stereotypes(input_dir, output_dir, file1_name,
                                                                          file2_name):
    # Load the data from the CSV files
    diversity_measures = pd.read_csv(os.path.join(input_dir, file1_name))
    frequency_analysis = pd.read_csv(os.path.join(input_dir, file2_name))

    # Merge the dataframes on the 'Stereotype' column
    merged_data = pd.merge(diversity_measures[['Stereotype', 'Gini Coefficient']],
                           frequency_analysis[['Stereotype', 'Group Frequency']], on='Stereotype')

    # Calculate thresholds (medians)
    gini_median = merged_data['Gini Coefficient'].median()  # Median for Gini Coefficient
    frequency_threshold = merged_data['Group Frequency'].median()  # Median for Group Frequency

    # Create a scatter plot for Gini Coefficient vs Group Frequency
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    sns.scatterplot(data=merged_data, x='Gini Coefficient', y='Group Frequency', hue='Stereotype', palette='viridis',
                    legend=False, s=100)

    # Customize the plot
    plt.title('Gini Coefficient vs. Group Frequency of Stereotypes', fontweight='bold')
    plt.xlabel('Gini Coefficient (Diversity Measure)')
    plt.ylabel('Stereotype Group Frequency')
    plt.grid(True)

    # Draw lines to create quadrants using medians
    plt.axhline(y=frequency_threshold, color='grey', linestyle='--', linewidth=1)  # Horizontal line for Group Frequency
    plt.axvline(x=gini_median, color='grey', linestyle='--', linewidth=1)  # Vertical line for Gini Coefficient

    # Create an empty list to store text objects
    texts = []

    # Annotate each point with the stereotype name
    for i in range(len(merged_data)):
        stereotype_name = merged_data['Stereotype'][i]

        # Create a text object without setting color initially
        text = plt.text(x=merged_data['Gini Coefficient'][i], y=merged_data['Group Frequency'][i] + 1.2,
                        s=stereotype_name, fontsize=8, ha='center',
                        color='black')  # Set initial color to black or any default

        # Append the text object to the list
        texts.append(text)

    # Apply the color_text function to the list of text objects
    color_text(texts)

    # Add annotations for the median lines
    plt.text(x=merged_data['Gini Coefficient'].min(), y=frequency_threshold,
             s=f'Median Group Frequency: {frequency_threshold:.2f}', color='grey', fontsize=10, va='bottom', ha='left')
    plt.text(x=gini_median, y=merged_data['Group Frequency'].min(), s=f'Median Gini Coefficient: {gini_median:.2f}',
             color='grey', fontsize=10, va='bottom', ha='right', rotation=90)

    fig_name = 'gini_coefficient_vs_group_frequency_stereotypes.png'
    fig.savefig(os.path.join(output_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {output_dir}.")
    plt.close(fig)


def execute_visualization_mutual_info_vs_dice_coefficient(input_dir, output_dir, file1_name, file2_name):
    # Load the data from the CSV files
    mutual_info_df = pd.read_csv(os.path.join(input_dir, file1_name))
    similarity_measures_df = pd.read_csv(os.path.join(input_dir, file2_name))

    # Prepare a list for DataFrame creation
    pairs_data = {'Stereotype Pair': [], 'Dice Coefficient': [], 'Mutual Information': []}

    # Iterate through the similarity measures dataframe to extract stereotype pairs and their similarity measures
    for i, row in similarity_measures_df.iterrows():
        stereotype_pair = ast.literal_eval(row['Stereotype Pair'])
        stereotype1, stereotype2 = stereotype_pair

        # Retrieve mutual information for each pair
        mi_value = mutual_info_df.loc[mutual_info_df['Stereotype'] == stereotype1, stereotype2].values
        if len(mi_value) > 0:
            pairs_data['Stereotype Pair'].append(f"{stereotype1} - {stereotype2}")
            pairs_data['Dice Coefficient'].append(row['Dice Coefficient'])
            pairs_data['Mutual Information'].append(mi_value[0])

    # Convert to DataFrame
    pairs_df = pd.DataFrame(pairs_data)

    # Save the results to a CSV file
    output_file_path = os.path.join(output_dir, 'stereotype_pairs_dice_similarity_info.csv')
    pairs_df.to_csv(output_file_path, index=False)
    print(f"CSV file saved to {output_file_path}")

    # Calculate thresholds (medians)
    dice_median = pairs_df['Dice Coefficient'].median()
    mutual_info_median = pairs_df['Mutual Information'].median()

    # Define thresholds for significant points
    dice_threshold = dice_median * 1.2  # Adjust this threshold as needed
    mutual_info_threshold = mutual_info_median * 1.2  # Adjust this threshold as needed

    # Create a scatter plot for Mutual Information vs. Dice Coefficient
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    sns.scatterplot(data=pairs_df, x='Dice Coefficient', y='Mutual Information', hue='Stereotype Pair',
                    palette='viridis', s=100, legend=False)

    # Customize the plot
    plt.title('Comparison of Mutual Information and Dice Coefficient for Stereotype Pairs', fontweight='bold')
    plt.xlabel('Dice Coefficient (Stereotype Pair)')
    plt.ylabel('Mutual Information (Stereotype Pair)')
    plt.grid(True)

    # Draw lines to create quadrants using medians
    plt.axhline(y=mutual_info_median, color='grey', linestyle='--',
                linewidth=1)  # Horizontal line for Mutual Information
    plt.axvline(x=dice_median, color='grey', linestyle='--', linewidth=1)  # Vertical line for Dice Coefficient

    # # Specify the pairs to annotate
    # pairs_to_annotate = ['abstract - category', 'category - kind', 'role - subkind']  # Add desired pairs here
    #
    # # Annotate only specific points
    # for i in range(len(pairs_df)):
    #     stereotype_pair = pairs_df['Stereotype Pair'][i]
    #     if stereotype_pair in pairs_to_annotate:  # Check if the pair is in the list of pairs to annotate
    #         color = 'black'  # Default color
    #         if 'none' in stereotype_pair:
    #             color = 'blue'
    #         elif 'other' in stereotype_pair:
    #             color = 'red'
    #         plt.text(x=pairs_df['Dice Coefficient'][i],
    #                  y=pairs_df['Mutual Information'][i],
    #                  s=stereotype_pair,
    #                  fontsize=8,
    #                  ha='right',
    #                  color=color)  # Use the color variable here

    # Add annotations for the median lines
    plt.text(x=max(pairs_df['Dice Coefficient']) / 2, y=mutual_info_median,
             s=f'Median Mutual Information: {mutual_info_median:.2f}', color='grey', fontsize=10, va='bottom',
             ha='left')
    plt.text(x=dice_median, y=max(pairs_df['Mutual Information'] / 2), s=f'Median Dice Coefficient: {dice_median:.2f}',
             color='grey', fontsize=10, va='bottom', ha='right', rotation=90)

    fig_name = 'mutual_info_vs_dice_coefficient.png'
    fig.savefig(os.path.join(output_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {output_dir}.")
    plt.close(fig)


def execute_visualization_coverage_percentage_all(in_dir_path, out_dir_path, file_path_occurrence, file_path_group):
    # Read the data from the CSV files
    data_occurrence = pd.read_csv(os.path.join(in_dir_path, file_path_occurrence))
    data_groupwise = pd.read_csv(os.path.join(in_dir_path, file_path_group))

    # 1. Plot Line Chart for Coverage vs. Percentage (Occurrence-wise and Group-wise)
    fig = plt.figure(figsize=(16, 9), tight_layout=True)  # Set the figure size

    # Plot occurrence-wise data
    sns.lineplot(data=data_occurrence, x='Percentage', y='Coverage', marker='o', label='Occurrence-wise Coverage',
                 color='blue')

    # Plot group-wise data
    sns.lineplot(data=data_groupwise, x='Percentage', y='Coverage', marker='o', label='Group-wise Coverage',
                 color='green')

    # Improved title and labels
    plt.title('Coverage vs. Top Percentages of Stereotypes (Occurrence-wise and Group-wise)', fontsize=14,
              fontweight='bold')
    plt.xlabel('Percentage of Stereotypes Considered (%)', fontsize=12)
    plt.ylabel('Coverage of Stereotype Occurrences', fontsize=12)

    # Add annotations for each point (occurrence-wise)
    for i in range(len(data_occurrence)):
        plt.text(data_occurrence['Percentage'][i] + 1,  # Slightly adjust the x position (move to the right)
                 data_occurrence['Coverage'][i] + 0.015,  # Slightly adjust the y position (move upwards)
                 f"k={data_occurrence['Top k Stereotypes'][i]}", fontsize=10, color='blue', ha='right'
                 # Horizontal alignment
                 )

    # Add annotations for each point (group-wise)
    for i in range(len(data_groupwise)):
        plt.text(data_groupwise['Percentage'][i] + 1,  # Slightly adjust the x position (move to the right)
                 data_groupwise['Coverage'][i] + 0.015,  # Slightly adjust the y position (move upwards)
                 f"k={data_groupwise['Top k Stereotypes'][i]}", fontsize=10, color='green', ha='right'
                 # Horizontal alignment
                 )

    # Additional formatting
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)  # Add a grid for better readability
    plt.gca().xaxis.set_major_locator(MultipleLocator(10))  # Set major ticks at intervals of 10

    # Add a legend to distinguish between occurrence-wise and group-wise
    plt.legend(loc='upper left')

    # Save the combined coverage plot
    fig_name = 'coverage_vs_stereotype_percentage_all.png'
    fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
    plt.close(fig)


def execute_visualization_pareto_combined(in_dir_path, out_dir_path, file_path_occurrence,
                                          file_path_group):
    # Load the CSV files into DataFrames
    data_occurrence = pd.read_csv(os.path.join(in_dir_path, file_path_occurrence))
    data_groupwise = pd.read_csv(os.path.join(in_dir_path, file_path_group))

    # Calculate the percentage frequency for both occurrence-wise and group-wise data
    data_occurrence['Percentage Frequency'] = (data_occurrence['Frequency'] / data_occurrence['Frequency'].sum()) * 100
    data_groupwise['Percentage Group-wise Frequency'] = (data_groupwise['Group-wise Frequency'] / data_groupwise[
        'Group-wise Frequency'].sum()) * 100

    # Calculate the cumulative percentage for both occurrence-wise and group-wise data
    data_occurrence['Cumulative Percentage'] = data_occurrence['Percentage Frequency'].cumsum()
    data_groupwise['Group-wise Cumulative Percentage'] = data_groupwise['Percentage Group-wise Frequency'].cumsum()

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(16, 9), tight_layout=True)

    # Bar plot for occurrence-wise frequency
    bar_width = 0.4
    ax1.bar(data_occurrence['Stereotype'], data_occurrence['Percentage Frequency'], width=bar_width,
            label='Occurrence-wise Frequency', align='center', color='skyblue')

    # Bar plot for group-wise frequency (shifted to the right)
    ax1.bar(data_groupwise['Stereotype'], data_groupwise['Percentage Group-wise Frequency'], width=bar_width,
            label='Group-wise Frequency', align='edge', color='lightgreen')

    # Set labels and title
    ax1.set_xlabel('Stereotype', fontsize=12)
    ax1.set_ylabel('Relative Frequency (%)', fontsize=12)
    ax1.set_title('Pareto Chart: Occurrence-wise and Group-wise Stereotype Frequencies', fontweight='bold', fontsize=14)

    # Rotate the labels by 90 degrees for better readability
    ax1.tick_params(axis='x', rotation=90)

    # Line plot for occurrence-wise cumulative percentage
    ax2 = ax1.twinx()  # Create a second y-axis
    ax2.plot(range(len(data_occurrence)), data_occurrence['Cumulative Percentage'], color='blue', marker='o',
             label='Occurrence-wise Cumulative', linestyle='-', linewidth=2)

    # Line plot for group-wise cumulative percentage
    ax2.plot(range(len(data_groupwise)), data_groupwise['Group-wise Cumulative Percentage'], color='green', marker='o',
             label='Group-wise Cumulative', linestyle='-', linewidth=2)

    # Set the y-axis label and ticks for the cumulative percentage
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax2.set_yticks(range(0, 101, 10))

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Save the plot
    fig_name = 'pareto_combined_occurrence_groupwise_rank_frequency.png'
    fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
    plt.close(fig)
