import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

from src.color_legend import color_text
from src.create_figure_subdir import create_figures_subdir


def execute_visualization_shannon_entropy_vs_global_frequency(file_path1, file_path2):
    # Load the data from the CSV files
    diversity_measures = pd.read_csv(file_path1)
    frequency_analysis = pd.read_csv(file_path2)
    save_dir = create_figures_subdir(file_path1)

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
    plt.text(x=(max(merged_data['Global Relative Frequency (Occurrence-wise)']))/2, y=frequency_threshold, s=f'Median Frequency: {frequency_threshold:.2e}', color='grey', fontsize=10,
             va='bottom', ha='left')
    plt.text(x=entropy_threshold, y=min(merged_data['Global Relative Frequency (Occurrence-wise)']),
             s=f'Median Entropy: {entropy_threshold:.2f}', color='grey', fontsize=10, va='bottom', ha='right',
             rotation=90)

    fig_name = 'shannon_entropy_vs_global_frequency.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")
    plt.close(fig)


execute_visualization_shannon_entropy_vs_global_frequency('../outputs/analyses/cs_analyses/diversity_measures.csv',
                             '../outputs/analyses/cs_analyses/frequency_analysis.csv')
