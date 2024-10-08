import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

from src.utils import color_text, create_visualizations_out_dirs


def create_heatmap(dataset, output_dir: str) -> None:
    """
    Create heatmaps for the Spearman correlation of class and relation stereotype statistics (both raw and clean)
    and save them as separate images.

    :param dataset: Dataset object containing models and statistics.
    :param output_dir: Directory to save the generated heatmaps.
    """
    # Create output directories for each type of statistic
    class_raw_out, class_clean_out, relation_raw_out, relation_clean_out = create_visualizations_out_dirs(output_dir,
                                                                                                          dataset.name)

    # Define a helper function to plot the heatmap
    def plot_custom_heatmap(corr_df, title, fig_name, specific_output_dir):
        if corr_df.empty:
            logger.warning(f"No correlation data available for {title}.")
            return

        fig = plt.figure(figsize=(16, 9), tight_layout=True)
        ax = sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=0.5)
        plt.title(title, fontweight='bold')

        # Customize tick label colors using color_text function
        color_text(ax.get_xticklabels())
        color_text(ax.get_yticklabels())

        # Rotate x-axis labels for better visibility
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        # Save the figure
        fig.savefig(os.path.join(specific_output_dir, fig_name), dpi=300)
        logger.success(f"Figure {fig_name} successfully saved in {specific_output_dir}.")
        plt.close(fig)

    # Convert the necessary 'spearman_correlation' data into DataFrames
    class_raw_corr_df = pd.DataFrame(dataset.class_statistics_raw["spearman_correlation"])
    relation_raw_corr_df = pd.DataFrame(dataset.relation_statistics_raw["spearman_correlation"])
    class_clean_corr_df = pd.DataFrame(dataset.class_statistics_clean["spearman_correlation"])
    relation_clean_corr_df = pd.DataFrame(dataset.relation_statistics_clean["spearman_correlation"])

    # Plot heatmap for class raw statistics
    plot_custom_heatmap(class_raw_corr_df, 'Spearman Correlation Heatmap of Class Raw Stereotype Statistics',
                        'spearman_correlation_heatmap.png', class_raw_out)

    # Plot heatmap for relation raw statistics
    plot_custom_heatmap(relation_raw_corr_df, 'Spearman Correlation Heatmap of Relation Raw Stereotype Statistics',
                        'spearman_correlation_heatmap.png', relation_raw_out)

    # Plot heatmap for class clean statistics
    plot_custom_heatmap(class_clean_corr_df, 'Spearman Correlation Heatmap of Class Clean Stereotype Statistics',
                        'spearman_correlation_heatmap.png', class_clean_out)

    # Plot heatmap for relation clean statistics
    plot_custom_heatmap(relation_clean_corr_df, 'Spearman Correlation Heatmap of Relation Clean Stereotype Statistics',
                        'spearman_correlation_heatmap.png', relation_clean_out)
