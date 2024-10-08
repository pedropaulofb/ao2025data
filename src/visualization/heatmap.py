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
    class_raw_out, class_clean_out, relation_raw_out, relation_clean_out = create_visualizations_out_dirs(output_dir, dataset.name)

    # Define a helper function to plot the heatmap
    def plot_custom_heatmap(corr_df, title, fig_name, specific_output_dir):
        if corr_df.empty:
            logger.warning(f"No correlation data available for {title}.")
            return

        # Drop rows/columns with NaNs
        corr_df = corr_df.dropna(how='all', axis=0).dropna(how='all', axis=1)

        # Check if the dataframe is still non-empty after dropping NaNs
        if corr_df.empty:
            logger.warning(f"After dropping NaNs, no data left to plot for {title}.")
            return

        # Use the DataFrame with both the index and columns as stereotype names
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

    # Convert the necessary 'spearman_correlation' data into DataFrames, setting the first column as index
    class_raw_corr_df = pd.DataFrame(dataset.class_statistics_raw["spearman_correlation"]).set_index('Stereotype')
    relation_raw_corr_df = pd.DataFrame(dataset.relation_statistics_raw["spearman_correlation"]).set_index('Stereotype')
    class_clean_corr_df = pd.DataFrame(dataset.class_statistics_clean["spearman_correlation"]).set_index('Stereotype')
    relation_clean_corr_df = pd.DataFrame(dataset.relation_statistics_clean["spearman_correlation"]).set_index('Stereotype')

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
