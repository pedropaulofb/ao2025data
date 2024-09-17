import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from loguru import logger

from src.color_legend import color_text
from src.create_figure_subdir import create_figures_subdir


def execute_visualization_ubiquity_vs_gini_coefficient(file_path1, file_path2):
    # Load the data from the CSV files
    diversity_measures = pd.read_csv(file_path1)
    frequency_analysis = pd.read_csv(file_path2)
    save_dir = create_figures_subdir(file_path1)

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
    sns.scatterplot(data=merged_data,
                    x='Gini Coefficient',
                    y='Ubiquity Index (Group Frequency per Group)',
                    hue='Construct',
                    palette='viridis',
                    size='Ubiquity Index (Group Frequency per Group)',
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

    # Annotate each point with the construct name
    texts = []  # Initialize an empty list to store text objects
    for i in range(len(merged_data)):
        construct_name = merged_data['Construct'][i]

        # Create text object with default color (black)
        text = plt.text(x=merged_data['Gini Coefficient'][i],
                        y=merged_data['Ubiquity Index (Group Frequency per Group)'][i]+0.01,
                        s=construct_name,
                        fontsize=8,
                        ha='right',
                        color='black')  # Initially set all colors to black
        texts.append(text)  # Append the text object to the list

    # Apply the color_text function to set specific colors
    color_text(texts)

    # Adjust text to avoid overlap
    adjust_text(texts)

    # Add annotations for the median lines
    plt.text(x=0, y=ubiquity_threshold, s=f'Median Ubiquity Index: {ubiquity_threshold:.2f}', color='grey', fontsize=10,
             va='bottom', ha='left')
    plt.text(x=gini_threshold,
             y=min(merged_data['Ubiquity Index (Group Frequency per Group)']),
             s=f'Median Gini Coefficient: {gini_threshold:.2f}', color='grey', fontsize=10, va='bottom', ha='right',
             rotation=90)

    fig_name = 'ubiquity_vs_gini_coefficient.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")
    plt.close(fig)


execute_visualization_ubiquity_vs_gini_coefficient('../outputs/analyses/cs_analyses/diversity_measures.csv',
                             '../outputs/analyses/cs_analyses/frequency_analysis.csv')
