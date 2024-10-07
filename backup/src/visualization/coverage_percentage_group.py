import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.ticker import MultipleLocator


def execute_visualization_coverage_percentage_group(in_dir_path, out_dir_path, file_path):
    # Read the data from the CSV file
    data = pd.read_csv(os.path.join(in_dir_path, file_path))

    # 1. Plot Line Chart for Group-wise Coverage vs. Percentage
    fig = plt.figure(figsize=(16, 9), tight_layout=True)  # Set the figure size
    sns.lineplot(data=data, x='Percentage', y='Coverage', marker='o')

    # Improved title and labels
    plt.title('Group-wise Coverage vs. Top Percentages of Stereotypes', fontsize=14, fontweight='bold')
    plt.xlabel('Percentage of Stereotypes Considered (%)', fontsize=12)
    plt.ylabel('Total Group-wise Coverage of Stereotype Occurrences', fontsize=12)

    # Add annotations for each point to display the corresponding "Top k Stereotypes" value
    for i in range(len(data)):
        plt.text(
            data['Percentage'][i] + 1,  # Slightly adjust the x position (move to the right)
            data['Coverage'][i] + 0.015,  # Slightly adjust the y position (move upwards)
            f"k={data['Top k Stereotypes'][i]}",
            fontsize=10,
            ha='right'  # Horizontal alignment
        )

    # Additional formatting
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)  # Add a grid for better readability
    plt.gca().xaxis.set_major_locator(MultipleLocator(10))  # Set major ticks at intervals of 10

    fig_name = 'groupwise_coverage_vs_stereotype_percentage.png'
    fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
    plt.close(fig)
