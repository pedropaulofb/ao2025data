import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from loguru import logger

from src.utils import create_visualizations_out_dirs, color_text


def new_create_boxplot(dataset, input_dir, output_dir: str, log_scale: bool = False) -> None:
    """
    Create separate boxplots for class and relation stereotype statistics (both raw and clean)
    and save them as separate images.

    :param dataset: The dataset name or object used to generate paths to CSV files.
    :param output_dir: Directory to save the generated boxplots.
    :param log_scale: Whether to use logarithmic scale on the y-axis.
    """

    class_raw_out, class_clean_out, relation_raw_out, relation_clean_out = create_visualizations_out_dirs(output_dir, dataset.name)

    if log_scale:
        suffix_title = " (Log-scale)"
        suffix_name = "_log"
    else:
        suffix_title = ""
        suffix_name = ""

    # Helper function to read the CSV files and optionally clean the data
    def read_csv_and_clean(file_path, clean=False):
        df = pd.read_csv(file_path)
        if clean:
            df = df.drop(columns=['none', 'other'], errors='ignore')  # Drop 'none' and 'other' if they exist
        return df

    # Convert the DataFrame to a long format for plotting
    def reshape_data_for_plotting(df):
        # Melt the DataFrame from wide format to long format
        df_long = df.melt(id_vars=['model'], var_name='Stereotype', value_name='Occurrences')
        return df_long

    # Plot boxplots using the reshaped data
    def plot_custom_boxplot(df, title, fig_name, specific_output_dir, log_scale):
        if df.empty:
            logger.warning(f"No data available in dataframe for {title}.")
            return

        df_long = reshape_data_for_plotting(df)

        fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)

        if log_scale:
            ax.set_yscale('log')

        # Use seaborn to create the boxplot
        sns.boxplot(x='Stereotype', y='Occurrences', data=df_long, ax=ax,whis=[0, 100])

        # Customize the plot
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Stereotype')
        ax.set_ylabel('Occurrences' + suffix_title)

        color_text(ax.get_xticklabels())

        # Rotate x-axis labels for better visibility
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)
        plt.tight_layout()

        # Save the figure
        fig.savefig(os.path.join(specific_output_dir, fig_name), dpi=300)
        logger.success(f"Figure {fig_name} successfully saved in {specific_output_dir}.")
        plt.close(fig)

    # File paths for the four CSVs
    class_file = os.path.join(input_dir, dataset.name, f"{dataset.name}_class_data.csv")
    relation_file = os.path.join(input_dir, dataset.name, f"{dataset.name}_relation_data.csv")

    # Read CSV files and plot for each case
    class_raw_df = read_csv_and_clean(class_file, clean=False)
    plot_custom_boxplot(class_raw_df, 'Box Plot of Class Raw Stereotype Statistics' + suffix_title,
                        f'boxplot{suffix_name}.png', class_raw_out, log_scale)

    class_clean_df = read_csv_and_clean(class_file, clean=True)
    plot_custom_boxplot(class_clean_df, 'Box Plot of Class Clean Stereotype Statistics' + suffix_title,
                        f'boxplot{suffix_name}.png', class_clean_out, log_scale)

    relation_raw_df = read_csv_and_clean(relation_file, clean=False)
    plot_custom_boxplot(relation_raw_df, 'Box Plot of Relation Raw Stereotype Statistics' + suffix_title,
                        f'boxplot{suffix_name}.png', relation_raw_out, log_scale)

    relation_clean_df = read_csv_and_clean(relation_file, clean=True)
    plot_custom_boxplot(relation_clean_df, 'Box Plot of Relation Clean Stereotype Statistics' + suffix_title,
                        f'boxplot{suffix_name}.png', relation_clean_out, log_scale)
