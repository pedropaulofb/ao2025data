import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

from src.create_figure_subdir import create_figures_subdir


def execute_visualization_inter4(file_path1, file_path2):
    # Load the data from the CSV files
    diversity_measures = pd.read_csv(file_path1)
    frequency_analysis = pd.read_csv(file_path2)
    save_dir = create_figures_subdir(file_path1)

    # Merge the dataframes on the 'Construct' column
    merged_data = pd.merge(diversity_measures[['Construct', 'Shannon Entropy']],
                           frequency_analysis[['Construct', 'Group Frequency']],
                           on='Construct')

    # Calculate thresholds (medians)
    entropy_threshold = merged_data['Shannon Entropy'].median()  # Median for Shannon Entropy
    frequency_threshold = merged_data['Group Frequency'].median()  # Median for Group Frequency

    # Create a scatter plot for Shannon Entropy vs Group Frequency
    fig = plt.figure(figsize=(16, 9), tight_layout=True)
    sns.scatterplot(data=merged_data,
                    x='Shannon Entropy',
                    y='Group Frequency',
                    hue='Construct',
                    palette='viridis',
                    legend=False,
                    s=100)

    # Customize the plot
    plt.title('Scatter Plot of Shannon Entropy vs. Group Frequency')
    plt.xlabel('Shannon Entropy')
    plt.ylabel('Group Frequency')
    plt.grid(True)

    # Draw lines to create quadrants using medians
    plt.axhline(y=frequency_threshold, color='grey', linestyle='--', linewidth=1)  # Horizontal line for Group Frequency
    plt.axvline(x=entropy_threshold, color='grey', linestyle='--', linewidth=1)  # Vertical line for Shannon Entropy

    # Annotate each point with the construct name and color for 'none' and 'other'
    for i in range(len(merged_data)):
        construct_name = merged_data['Construct'][i]

        # Set the color based on the construct name
        if construct_name == 'none':
            color = 'blue'
        elif construct_name == 'other':
            color = 'red'
        else:
            color = 'black'  # Default color for other constructs

        plt.text(x=merged_data['Shannon Entropy'][i],
                 y=merged_data['Group Frequency'][i] + 1.2,
                 s=construct_name,
                 fontsize=8,
                 ha='center',
                 color=color)  # Use the color variable here

    # Add annotations for the median lines
    plt.text(x=0, y=frequency_threshold, s=f'Median Group Frequency: {frequency_threshold:.2f}', color='grey',
             fontsize=10,
             va='bottom', ha='left')
    plt.text(x=entropy_threshold, y=min(merged_data['Group Frequency']),
             s=f'Median Shannon Entropy: {entropy_threshold:.2f}', color='grey', fontsize=10, va='bottom', ha='right',
             rotation=90)

    fig_name = 'inter4_fig1.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")
    plt.close(fig)


execute_visualization_inter4('../outputs/analyses/cs_analyses/diversity_measures.csv',
                             '../outputs/analyses/cs_analyses/frequency_analysis.csv')
