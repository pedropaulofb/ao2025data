import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger

from backup.src.utils import color_text


def plot_distance_from_origin(in_file_path, out_dir_path):
    # Read the CSV file
    df = pd.read_csv(in_file_path, sep=',')

    # Automatically use the first column as the index (group/label)
    index_col = df.columns[0]  # The first column in the DataFrame
    distance_col = 'DistanceFromOrigin'  # Hardcoded column for distance

    # Sort the DataFrame by DistanceFromOrigin in descending order
    df_sorted = df.sort_values(by=distance_col, ascending=False)

    # Create a bar plot
    fig = plt.figure(figsize=(16, 9))
    ax = sns.barplot(x=index_col, y=distance_col, data=df_sorted)

    # Rotate x labels for readability
    plt.xticks(rotation=90)

    # Apply the color_text function to the x-axis labels
    color_text(ax.get_xticklabels())

    # Set the title and labels
    plt.title('Distance from Origin by Group', fontsize=16)
    plt.xlabel('Group', fontsize=12)
    plt.ylabel('Distance from Origin', fontsize=12)

    # Display the plot
    plt.tight_layout()
    fig_name = os.path.splitext(os.path.basename(in_file_path))[0]+".png"
    fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
    plt.close(fig)