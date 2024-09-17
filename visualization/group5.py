import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.ticker import MultipleLocator

from src.create_figure_subdir import create_figures_subdir


def execute_visualization_group5(file_path):
    # Read the data from the CSV file
    data = pd.read_csv(file_path)
    save_dir = create_figures_subdir(file_path)

    # 1. Plot Line Chart for Coverage vs. Percentage
    fig = plt.figure(figsize=(12, 8), tight_layout=True)  # Set the figure size
    sns.lineplot(data=data, x='Percentage', y='Coverage', marker='o')

    # Improved title and labels
    plt.title('Coverage Achieved by Top Percentages of Constructs', fontsize=14)
    plt.xlabel('Top Percentage of Constructs Considered (%)', fontsize=12)
    plt.ylabel('Coverage of Total Occurrences', fontsize=12)

    # Add annotations for each point to display the corresponding "Top k Constructs" value
    for i in range(len(data)):
        plt.text(
            data['Percentage'][i] + 1,  # Slightly adjust the x position (move to the right)
            data['Coverage'][i] + 0.015,  # Slightly adjust the y position (move upwards)
            f"k={data['Top k Constructs'][i]}",
            fontsize=10,
            ha='right'  # Horizontal alignment
        )

    # Additional formatting
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)  # Add a grid for better readability
    plt.gca().xaxis.set_major_locator(MultipleLocator(10))  # Set major ticks at intervals of 10

    fig_name = 'group5_fig1.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")


execute_visualization_group5('../outputs/analyses/cs_analyses/coverage_percentage.csv')
