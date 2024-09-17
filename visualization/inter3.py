import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

from src.create_figure_subdir import create_figures_subdir


def execute_visualization_inter3(file_path1, file_path2):
    # Load the data from the CSV files
    rank_df = pd.read_csv(file_path1)
    mutual_info_df = pd.read_csv(file_path2)
    save_dir = create_figures_subdir(file_path1)

    # Calculate the average mutual information for each construct
    mutual_info_avg = mutual_info_df.drop(columns='Construct').mean(axis=1)
    mutual_info_df['Avg Mutual Information'] = mutual_info_avg

    # Merge rank data with mutual information data
    merged_data = pd.merge(rank_df[['Construct', 'Rank']],
                           mutual_info_df[['Construct', 'Avg Mutual Information']],
                           on='Construct')

    # Plot the scatter plot
    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    sns.scatterplot(data=merged_data,
                    x='Rank',
                    y='Avg Mutual Information',
                    hue='Construct',
                    palette='viridis',
                    size='Avg Mutual Information',
                    sizes=(20, 200),
                    legend=False)

    # Customize the plot
    plt.title('Scatter Plot of Rank vs. Mutual Information')
    plt.xlabel('Rank')
    plt.ylabel('Average Mutual Information')
    plt.grid(True)

    # Annotate each point with the construct name
    for i in range(len(merged_data)):
        plt.text(x=merged_data['Rank'][i],
                 y=merged_data['Avg Mutual Information'][i] + 0.01,
                 s=merged_data['Construct'][i],
                 fontsize=8,
                 ha='center')

    # Show the plot
    fig_name = 'inter3_fig1.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")


execute_visualization_inter3('../outputs/analyses/cs_analyses/rank_frequency_distribution.csv',
                             '../outputs/analyses/cs_analyses/mutual_information.csv')
