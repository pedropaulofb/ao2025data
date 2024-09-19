import ast
import os

import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from loguru import logger
from matplotlib import pyplot as plt

from src.color_legend import color_text


def execute_visualization_mutual_info_vs_jaccard_similarity(input_dir, output_dir, file1_name, file2_name):
    # Load the data from the CSV files
    mutual_info_df = pd.read_csv(os.path.join(input_dir, file1_name))
    similarity_measures_df = pd.read_csv(os.path.join(input_dir, file2_name))

    # Prepare a list for DataFrame creation
    pairs_data = {'Construct Pair': [], 'Jaccard Similarity': [], 'Mutual Information': []}

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
    output_file_path = os.path.join(output_dir, 'construct_pairs_jaccard_similarity_info.csv')
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
    sns.scatterplot(data=pairs_df, x='Jaccard Similarity', y='Mutual Information', hue='Construct Pair',
                    palette='viridis', s=100, legend=False)

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


def execute_visualization_simpson_diversity_vs_construct_frequency(input_dir, output_dir, file1_name, file2_name):
    # Load the data from the CSV files
    diversity_measures = pd.read_csv(os.path.join(input_dir, file1_name))
    frequency_analysis = pd.read_csv(os.path.join(input_dir, file2_name))

    # Merge the dataframes on the 'Construct' column
    merged_data = pd.merge(diversity_measures[['Construct', 'Simpson Index']],
                           frequency_analysis[['Construct', 'Total Frequency']], on='Construct')

    # Calculate thresholds (medians)
    frequency_threshold = merged_data['Total Frequency'].median()  # Median for Total Frequency
    simpson_threshold = merged_data['Simpson Index'].median()  # Median for Simpson Index

    # Create a scatter plot for Simpson's Index vs. Total Frequency
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    sns.scatterplot(data=merged_data, x='Total Frequency', y='Simpson Index', hue='Construct', palette='viridis', s=100,
                    legend=False)

    # Customize the plot
    plt.title("Simpson's Diversity Index vs. Total Construct Frequency", fontweight='bold')
    plt.xlabel('Total Construct Frequency')
    plt.ylabel("Simpson's Diversity Index")
    plt.grid(True)

    # Draw lines to create quadrants using medians
    plt.axhline(y=simpson_threshold, color='grey', linestyle='--', linewidth=1)  # Horizontal line for Simpson Index
    plt.axvline(x=frequency_threshold, color='grey', linestyle='--', linewidth=1)  # Vertical line for Total Frequency

    # Create an empty list to store text objects
    texts = []

    # Annotate each point with the construct name
    for i in range(len(merged_data)):
        construct_name = merged_data['Construct'][i]

        # Create a text object without setting color initially
        text = plt.text(x=merged_data['Total Frequency'][i], y=merged_data['Simpson Index'][i] + 0.01, s=construct_name,
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

    fig_name = 'simpson_diversity_vs_construct_frequency.png'
    fig.savefig(os.path.join(output_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {output_dir}.")
    plt.close(fig)


def execute_visualization_mutual_information_vs_construct_rank(input_dir, output_dir, file1_name, file2_name):
    # Load the data from the CSV files
    rank_df = pd.read_csv(os.path.join(input_dir, file1_name))
    mutual_info_df = pd.read_csv(os.path.join(input_dir, file2_name))

    # Calculate the average mutual information for each construct
    mutual_info_avg = mutual_info_df.drop(columns='Construct').mean(axis=1)
    mutual_info_df['Avg Mutual Information'] = mutual_info_avg

    # Merge rank data with mutual information data
    merged_data = pd.merge(rank_df[['Construct', 'Rank']], mutual_info_df[['Construct', 'Avg Mutual Information']],
                           on='Construct')

    # Plot the scatter plot
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    sns.scatterplot(data=merged_data, x='Rank', y='Avg Mutual Information', hue='Construct', palette='viridis',
                    size='Avg Mutual Information', sizes=(20, 200), legend=False)

    # Customize the plot
    plt.title('Average Mutual Information vs. Construct Rank', fontweight='bold')
    plt.xlabel('Construct Rank')
    plt.ylabel('Mean Mutual Information')
    plt.grid(True)

    # Annotate each point with the construct name
    for i in range(len(merged_data)):
        plt.text(x=merged_data['Rank'][i], y=merged_data['Avg Mutual Information'][i] + 0.01,
                 s=merged_data['Construct'][i], fontsize=8, ha='center')

    # Show the plot
    fig_name = 'mutual_information_vs_construct_rank.png'
    fig.savefig(os.path.join(output_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {output_dir}.")
    plt.close(fig)


def execute_visualization_shannon_entropy_vs_global_frequency(input_dir, output_dir, file1_name, file2_name):
    # Load the data from the CSV files
    diversity_measures = pd.read_csv(os.path.join(input_dir, file1_name))
    frequency_analysis = pd.read_csv(os.path.join(input_dir, file2_name))

    # Merge the dataframes on the 'Construct' column
    merged_data = pd.merge(diversity_measures[['Construct', 'Shannon Entropy']],
                           frequency_analysis[['Construct', 'Global Relative Frequency (Occurrence-wise)']],
                           on='Construct')

    # Calculate thresholds
    entropy_threshold = merged_data['Shannon Entropy'].median()  # Using median as a threshold
    frequency_threshold = merged_data[
        'Global Relative Frequency (Occurrence-wise)'].median()  # Using median as a threshold

    # Plot with Seaborn scatter plot
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    sns.scatterplot(data=merged_data, x='Shannon Entropy', y='Global Relative Frequency (Occurrence-wise)',
                    hue='Construct', palette='viridis', size='Global Relative Frequency (Occurrence-wise)',
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

    # Annotate each point with the construct name
    texts = []  # Initialize an empty list to store text objects
    for i in range(len(merged_data)):
        construct_name = merged_data['Construct'][i]

        # Create text object with default color (black)
        text = plt.text(x=merged_data['Shannon Entropy'][i],
                        y=merged_data['Global Relative Frequency (Occurrence-wise)'][i], s=construct_name, fontsize=8,
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


def execute_visualization_shannon_entropy_vs_group_frequency_constructs(input_dir, output_dir, file1_name, file2_name):
    # Load the data from the CSV files
    diversity_measures = pd.read_csv(os.path.join(input_dir, file1_name))
    frequency_analysis = pd.read_csv(os.path.join(input_dir, file2_name))

    # Merge the dataframes on the 'Construct' column
    merged_data = pd.merge(diversity_measures[['Construct', 'Shannon Entropy']],
                           frequency_analysis[['Construct', 'Group Frequency']], on='Construct')

    # Calculate thresholds (medians)
    entropy_threshold = merged_data['Shannon Entropy'].median()  # Median for Shannon Entropy
    frequency_threshold = merged_data['Group Frequency'].median()  # Median for Group Frequency

    # Create a scatter plot for Shannon Entropy vs Group Frequency
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    sns.scatterplot(data=merged_data, x='Shannon Entropy', y='Group Frequency', hue='Construct', palette='viridis',
                    legend=False, s=100)

    # Customize the plot
    plt.title('Shannon Entropy vs. Group Frequency of Constructs', fontweight='bold')
    plt.xlabel('Shannon Entropy (Diversity Measure)')
    plt.ylabel('Construct Group Frequency')
    plt.grid(True)

    # Draw lines to create quadrants using medians
    plt.axhline(y=frequency_threshold, color='grey', linestyle='--', linewidth=1)  # Horizontal line for Group Frequency
    plt.axvline(x=entropy_threshold, color='grey', linestyle='--', linewidth=1)  # Vertical line for Shannon Entropy

    # Create an empty list to store text objects
    texts = []

    # Annotate each point with the construct name
    for i in range(len(merged_data)):
        construct_name = merged_data['Construct'][i]

        # Create a text object without setting color initially
        text = plt.text(x=merged_data['Shannon Entropy'][i], y=merged_data['Group Frequency'][i] + 1.2,
                        s=construct_name, fontsize=8, ha='center',
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

    fig_name = 'shannon_entropy_vs_group_frequency_constructs.png'
    fig.savefig(os.path.join(output_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {output_dir}.")
    plt.close(fig)


def execute_visualization_ubiquity_vs_gini_coefficient(input_dir, output_dir, file1_name, file2_name):
    # Load the data from the CSV files
    diversity_measures = pd.read_csv(os.path.join(input_dir, file1_name))
    frequency_analysis = pd.read_csv(os.path.join(input_dir, file2_name))

    # Merge the dataframes on the 'Construct' column
    merged_data = pd.merge(diversity_measures[['Construct', 'Gini Coefficient']],
                           frequency_analysis[['Construct', 'Ubiquity Index (Group Frequency per Group)']],
                           on='Construct')

    # Calculate thresholds
    gini_threshold = merged_data['Gini Coefficient'].median()  # Using median as a threshold
    ubiquity_threshold = merged_data[
        'Ubiquity Index (Group Frequency per Group)'].median()  # Using median as a threshold

    # Plot with Seaborn scatter plot
    fig = plt.figure(figsize=(16, 9))
    sns.scatterplot(data=merged_data, x='Gini Coefficient', y='Ubiquity Index (Group Frequency per Group)',
                    hue='Construct', palette='viridis', size='Ubiquity Index (Group Frequency per Group)', legend=False,
                    sizes=(20, 200))

    # Customize the plot
    plt.title("Ubiquity Index vs. Gini Coefficient", fontweight='bold')
    plt.xlabel('Gini Coefficient (Inequality Measure)')
    plt.ylabel('Ubiquity Index (Frequency per Group)')
    plt.grid(True)

    # Draw lines to create quadrants
    plt.axhline(y=ubiquity_threshold, color='grey', linestyle='--', linewidth=1)  # Horizontal line
    plt.axvline(x=gini_threshold, color='grey', linestyle='--', linewidth=1)  # Vertical line

    # Annotate each point with the construct name
    texts = []  # Initialize an empty list to store text objects
    for i in range(len(merged_data)):
        construct_name = merged_data['Construct'][i]

        # Create text object with default color (black)
        text = plt.text(x=merged_data['Gini Coefficient'][i],
                        y=merged_data['Ubiquity Index (Group Frequency per Group)'][i] + 0.01, s=construct_name,
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

    # Merge the dataframes on the 'Construct' column
    merged_data = pd.merge(diversity_measures[['Construct', 'Gini Coefficient']],
                           frequency_analysis[['Construct', 'Global Relative Frequency (Occurrence-wise)']],
                           on='Construct')

    # Calculate thresholds
    gini_median = merged_data['Gini Coefficient'].median()  # Using median as a threshold
    frequency_threshold = merged_data[
        'Global Relative Frequency (Occurrence-wise)'].median()  # Using median as a threshold

    # Plot with Seaborn scatter plot
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    sns.scatterplot(data=merged_data, x='Gini Coefficient', y='Global Relative Frequency (Occurrence-wise)',
                    hue='Construct', palette='viridis', size='Global Relative Frequency (Occurrence-wise)',
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

    # Annotate each point with the construct name
    texts = []  # Initialize an empty list to store text objects
    for i in range(len(merged_data)):
        construct_name = merged_data['Construct'][i]

        # Create text object with default color (black)
        text = plt.text(x=merged_data['Gini Coefficient'][i],
                        y=merged_data['Global Relative Frequency (Occurrence-wise)'][i], s=construct_name, fontsize=8,
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


def execute_visualization_gini_coefficient_vs_group_frequency_constructs(input_dir, output_dir, file1_name, file2_name):
    # Load the data from the CSV files
    diversity_measures = pd.read_csv(os.path.join(input_dir, file1_name))
    frequency_analysis = pd.read_csv(os.path.join(input_dir, file2_name))

    # Merge the dataframes on the 'Construct' column
    merged_data = pd.merge(diversity_measures[['Construct', 'Gini Coefficient']],
                           frequency_analysis[['Construct', 'Group Frequency']], on='Construct')

    # Calculate thresholds (medians)
    gini_median = merged_data['Gini Coefficient'].median()  # Median for Gini Coefficient
    frequency_threshold = merged_data['Group Frequency'].median()  # Median for Group Frequency

    # Create a scatter plot for Gini Coefficient vs Group Frequency
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    sns.scatterplot(data=merged_data, x='Gini Coefficient', y='Group Frequency', hue='Construct', palette='viridis',
                    legend=False, s=100)

    # Customize the plot
    plt.title('Gini Coefficient vs. Group Frequency of Constructs', fontweight='bold')
    plt.xlabel('Gini Coefficient (Diversity Measure)')
    plt.ylabel('Construct Group Frequency')
    plt.grid(True)

    # Draw lines to create quadrants using medians
    plt.axhline(y=frequency_threshold, color='grey', linestyle='--', linewidth=1)  # Horizontal line for Group Frequency
    plt.axvline(x=gini_median, color='grey', linestyle='--', linewidth=1)  # Vertical line for Gini Coefficient

    # Create an empty list to store text objects
    texts = []

    # Annotate each point with the construct name
    for i in range(len(merged_data)):
        construct_name = merged_data['Construct'][i]

        # Create a text object without setting color initially
        text = plt.text(x=merged_data['Gini Coefficient'][i], y=merged_data['Group Frequency'][i] + 1.2,
                        s=construct_name, fontsize=8, ha='center',
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

    fig_name = 'gini_coefficient_vs_group_frequency_constructs.png'
    fig.savefig(os.path.join(output_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {output_dir}.")
    plt.close(fig)


def execute_visualization_mutual_info_vs_dice_coefficient(input_dir, output_dir, file1_name, file2_name):
    # Load the data from the CSV files
    mutual_info_df = pd.read_csv(os.path.join(input_dir, file1_name))
    similarity_measures_df = pd.read_csv(os.path.join(input_dir, file2_name))

    # Prepare a list for DataFrame creation
    pairs_data = {'Construct Pair': [], 'Dice Coefficient': [], 'Mutual Information': []}

    # Iterate through the similarity measures dataframe to extract construct pairs and their similarity measures
    for i, row in similarity_measures_df.iterrows():
        construct_pair = ast.literal_eval(row['Construct Pair'])
        construct1, construct2 = construct_pair

        # Retrieve mutual information for each pair
        mi_value = mutual_info_df.loc[mutual_info_df['Construct'] == construct1, construct2].values
        if len(mi_value) > 0:
            pairs_data['Construct Pair'].append(f"{construct1} - {construct2}")
            pairs_data['Dice Coefficient'].append(row['Dice Coefficient'])
            pairs_data['Mutual Information'].append(mi_value[0])

    # Convert to DataFrame
    pairs_df = pd.DataFrame(pairs_data)

    # Save the results to a CSV file
    output_file_path = os.path.join(output_dir, 'construct_pairs_dice_similarity_info.csv')
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
    sns.scatterplot(data=pairs_df, x='Dice Coefficient', y='Mutual Information', hue='Construct Pair',
                    palette='viridis', s=100, legend=False)

    # Customize the plot
    plt.title('Comparison of Mutual Information and Dice Coefficient for Construct Pairs', fontweight='bold')
    plt.xlabel('Dice Coefficient (Construct Pair)')
    plt.ylabel('Mutual Information (Construct Pair)')
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
    #     construct_pair = pairs_df['Construct Pair'][i]
    #     if construct_pair in pairs_to_annotate:  # Check if the pair is in the list of pairs to annotate
    #         color = 'black'  # Default color
    #         if 'none' in construct_pair:
    #             color = 'blue'
    #         elif 'other' in construct_pair:
    #             color = 'red'
    #         plt.text(x=pairs_df['Dice Coefficient'][i],
    #                  y=pairs_df['Mutual Information'][i],
    #                  s=construct_pair,
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
