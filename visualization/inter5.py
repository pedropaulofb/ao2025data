import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from loguru import logger

from src.color_legend import color_text
from src.create_figure_subdir import create_figures_subdir


def execute_visualization_simpson_diversity_vs_construct_frequency(file_path1, file_path2):
    # Load the data from the CSV files
    diversity_measures = pd.read_csv(file_path1)
    frequency_analysis = pd.read_csv(file_path2)
    save_dir = create_figures_subdir(file_path1)

    # Merge the dataframes on the 'Construct' column
    merged_data = pd.merge(diversity_measures[['Construct', 'Simpson Index']],
                           frequency_analysis[['Construct', 'Total Frequency']],
                           on='Construct')

    # Calculate thresholds (medians)
    frequency_threshold = merged_data['Total Frequency'].median()  # Median for Total Frequency
    simpson_threshold = merged_data['Simpson Index'].median()  # Median for Simpson Index

    # Create a scatter plot for Simpson's Index vs. Total Frequency
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    sns.scatterplot(data=merged_data,
                    x='Total Frequency',
                    y='Simpson Index',
                    hue='Construct',
                    palette='viridis',
                    s=100,
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
        text = plt.text(x=merged_data['Total Frequency'][i],
                        y=merged_data['Simpson Index'][i] + 0.01,
                        s=construct_name,
                        fontsize=8,
                        ha='center',
                        color='black')  # Initial color set to black or any default

        # Append the text object to the list
        texts.append(text)

    # Apply the color_text function to the list of text objects
    color_text(texts)

    # Adjust text to avoid overlap
    adjust_text(texts)

    # Add annotations for the median lines
    plt.text(x=max(merged_data['Total Frequency'])/2, y=simpson_threshold, s=f'Median Simpson Index: {simpson_threshold:.2f}', color='grey', fontsize=10,
             va='bottom', ha='left')
    plt.text(x=frequency_threshold, y=max(merged_data['Simpson Index'])/2,
             s=f'Median Total Frequency: {frequency_threshold:.2f}', color='grey', fontsize=10, va='bottom', ha='right',
             rotation=90)

    fig_name = 'simpson_diversity_vs_construct_frequency.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")
    plt.close(fig)


execute_visualization_simpson_diversity_vs_construct_frequency('../outputs/analyses/cs_analyses/diversity_measures.csv',
                             '../outputs/analyses/cs_analyses/frequency_analysis.csv')
