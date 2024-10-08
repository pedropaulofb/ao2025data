import os

import matplotlib.pyplot as plt
import pandas as pd
from icecream import ic
from loguru import logger
from matplotlib import patches

from src.utils import color_text, create_visualizations_out_dirs


def create_boxplot(dataset, output_dir: str) -> None:
    """
    Create separate boxplots for class and relation stereotype statistics (both raw and clean)
    and save them as separate images.

    :param dataset: Dataset object containing models and statistics.
    :param output_dir: Directory to save the generated boxplots.
    """
    class_raw_out, class_clean_out, relation_raw_out, relation_clean_out = create_visualizations_out_dirs(output_dir, dataset.name)

    # Define a helper function to plot the boxplot for each dataframe
    def plot_custom_boxplot(df, title, fig_name, specific_output_dir):
        if df.empty:
            logger.warning(f"No data available in dataframe for {title}.")
            return

        # Ensure necessary columns exist
        required_columns = ['Stereotype', 'Min', 'Q1', 'Median', 'Q3', 'Max']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Dataframe for {title} lacks required columns for boxplot: {required_columns}")
            return

        fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)

        # Loop through each stereotype to plot the boxplot manually
        stereotypes = df['Stereotype'].unique()
        positions = range(len(stereotypes))

        # Collect data for each stereotype
        for pos, stereotype in zip(positions, stereotypes):
            subset = df[df['Stereotype'] == stereotype]

            # Fetch the Min, Q1, Median, Q3, and Max values
            min_value = subset['Min'].values[0]
            q1_value = subset['Q1'].values[0]
            median_value = subset['Median'].values[0]
            q3_value = subset['Q3'].values[0]
            max_value = subset['Max'].values[0]

            # No special handling for zero values, Q1 and Q3 will be respected as they are
            # Increase linewidth and markersize for visibility
            ax.plot([pos, pos], [min_value, max_value], color='black', linewidth=2, zorder=2)
            # Add a solid rectangle for Q1 to Q3 box
            rect = patches.Rectangle((pos - 0.35, q1_value), 0.7, q3_value - q1_value, color='blue', zorder=3)
            ax.add_patch(rect)
            ax.plot(pos, median_value, 'r_', markersize=30, markeredgewidth=2, linewidth=30, zorder=4)

        # Set y-axis limits to ensure the box starts at 0
        plt.ylim(0, max(df['Max'] * 1.05))

        # Customize the plot
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Stereotype')
        ax.set_ylabel('Value')
        ax.set_xticks(positions)
        ax.set_xticklabels(stereotypes, rotation=45, ha='right')  # Rotate x-axis labels for better visibility
        plt.grid(True)
        plt.tight_layout()
        color_text(ax.get_xticklabels())

        # Save the figure
        fig.savefig(os.path.join(specific_output_dir, fig_name), dpi=300)
        logger.success(f"Figure {fig_name} successfully saved in {specific_output_dir}.")
        plt.close(fig)

    # Convert the necessary columns from your dataset into DataFrames
    class_raw_df = pd.DataFrame(dataset.class_statistics_raw["central_tendency_dispersion"])
    relation_raw_df = pd.DataFrame(dataset.relation_statistics_raw["central_tendency_dispersion"])
    class_clean_df = pd.DataFrame(dataset.class_statistics_clean["central_tendency_dispersion"])
    relation_clean_df = pd.DataFrame(dataset.relation_statistics_clean["central_tendency_dispersion"])

    # Plot for class raw statistics
    plot_custom_boxplot(class_raw_df, 'Box Plot of Class Raw Stereotype Statistics',
                        'boxplot_stereotype_statistics.png', class_raw_out)

    # Plot for relation raw statistics
    plot_custom_boxplot(relation_raw_df, 'Box Plot of Relation Raw Stereotype Statistics',
                        'boxplot_stereotype_statistics.png', relation_raw_out)

    # Plot for class clean statistics
    plot_custom_boxplot(class_clean_df, 'Box Plot of Class Clean Stereotype Statistics',
                        'boxplot_stereotype_statistics.png', class_clean_out)

    # Plot for relation clean statistics
    plot_custom_boxplot(relation_clean_df, 'Box Plot of Relation Clean Stereotype Statistics',
                        'boxplot_stereotype_statistics.png', relation_clean_out)


