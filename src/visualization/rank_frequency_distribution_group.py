import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

from src.utils import color_text


def execute_visualization_rank_frequency_distribution_group(in_dir_path, out_dir_path, file_path):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(os.path.join(in_dir_path, file_path))

    # Calculate the percentage frequency for the bar plot
    data['Percentage Group-wise Frequency'] = (data['Group-wise Frequency'] / data['Group-wise Frequency'].sum()) * 100

    # Calculate the cumulative percentage
    data['Group-wise Cumulative Percentage'] = data['Percentage Group-wise Frequency'].cumsum()

    # 1. Pareto Chart for Group-wise Rank-Frequency and Cumulative Percentage
    fig, ax1 = plt.subplots(figsize=(16, 9), tight_layout=True)

    # Bar plot for Group-wise Rank-Percentage Frequency with uniform color
    sns.barplot(x='Stereotype', y='Percentage Group-wise Frequency', data=data, ax=ax1, color='skyblue')
    ax1.set_xlabel('Stereotype')
    ax1.set_ylabel('Group-wise Relative Frequency (%)')
    ax1.set_title('Pareto Chart: Group-wise Stereotype Frequency and Cumulative Percentage', fontweight='bold')

    # Apply color to specific labels
    color_text(ax1.get_xticklabels())

    # Rotate the labels by 45 degrees
    ax1.tick_params(axis='x', rotation=90)

    # Line plot for Group-wise Cumulative Percentage
    ax2 = ax1.twinx()
    ax2.plot(range(len(data)), data['Group-wise Cumulative Percentage'], color='red', marker='o')
    ax2.set_ylabel('Cumulative Group-wise Frequency (%)')
    ax2.set_yticks(range(0, 101, 10))  # Set cumulative percentage grid at every 10%

    plt.grid(True)
    fig.tight_layout()  # Ensure everything fits within the figure
    fig_name = 'pareto_chart_groupwise_stereotype_frequency_cumulative_percentage.png'
    fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
    plt.close(fig)
