import os
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from loguru import logger
import pandas as pd

from src.utils import format_metric_name


# Helper function to plot a scatter plot for a given DataFrame and save the figure
def plot_scatter_plot(df, index_col, x_metric, y_metric, stat_label, output_dir, plot_medians):
    """
    Helper function to create and save a scatter plot.

    :param df: DataFrame containing the statistics.
    :param index_col: The column representing the index (e.g., 'Stereotype').
    :param x_metric: The column name to be used for the x-axis.
    :param y_metric: The column name to be used for the y-axis.
    :param stat_label: The label for the statistics type (e.g., 'Class Raw').
    :param output_dir: Directory to save the plot.
    :param plot_medians: Boolean flag to indicate if medians should be plotted.
    """
    # Ensure that Stereotype is categorical and ordered
    df[index_col] = pd.Categorical(df[index_col], categories=df[index_col].unique(), ordered=True)

    fig = plt.figure(figsize=(16, 9), tight_layout=True)

    # Dynamically generate a color palette based on the number of unique categories
    num_unique_values = len(df[index_col].unique())

    # Check if the number of unique values is less than or equal to 23
    if num_unique_values <= 23:
        # Use 'tab10' (10 colors) + extended palette (23 colors total)
        base_palette = sns.color_palette('tab10', n_colors=12)
        extended_palette = base_palette + base_palette[:11]  # Repeat some colors to get up to 23
        palette = extended_palette[:num_unique_values]  # Use only as many colors as needed
    else:
        # Use 'hsv' palette if more than 23 unique values
        palette = sns.color_palette('hsv', n_colors=num_unique_values)

    texts = []
    # Plotting the scatter plot with circles and adding labels to each point
    for i, index_value in enumerate(df[index_col].unique()):
        subset = df[df[index_col] == index_value]

        # Filter out rows with non-finite values in either x_metric or y_metric
        subset = subset[pd.notnull(subset[x_metric]) & pd.notnull(subset[y_metric])]

        # Skip if no valid data to plot for this index
        if subset.empty:
            continue

        # Plot the point (circle marker)
        plt.scatter(subset[x_metric], subset[y_metric],
                    color=palette[i], marker='o', s=100, edgecolor='w', label=index_value)

        # Add label next to each point
        for j in range(len(subset)):
            text = plt.text(subset[x_metric].values[j],
                            subset[y_metric].values[j], index_value, fontsize=8,
                            color='black', ha='left', va='top')
            texts.append(text)

    # Adjust text to avoid overlap
    adjust_text(texts)

    # Adding labels and title
    plt.xlabel(x_metric.replace('_', ' ').title())
    plt.ylabel(y_metric.replace('_', ' ').title())
    plt.title(f'{x_metric.replace("_", " ").title()} vs. {y_metric.replace("_", " ").title()} ({stat_label})',
              fontweight='bold')

    # Check if we need to plot medians
    if plot_medians:
        # Calculate median values
        median_y = df[y_metric].median()
        median_x = df[x_metric].median()

        # Adding a cross to separate the plot into four quadrants using the median
        plt.axhline(y=median_y, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=median_x, color='black', linestyle='--', linewidth=1)

        # Adding text for the median values at the extreme (highest values) parts of the lines
        plt.text(df[x_metric].max(), median_y, f'median: {median_y:.2f}', color='gray', fontsize=10,
                 ha='right', va='bottom')
        plt.text(median_x, df[y_metric].max(), f'median: {median_x:.2f}', color='gray',
                 fontsize=10, ha='right', va='top', rotation=90)

    # Remove the legend (as labels are now added directly to the points)
    plt.legend().remove()

    formatted_x_metric = format_metric_name(x_metric)  # Apply formatting to x_metric
    formatted_y_metric = format_metric_name(y_metric)  # Apply formatting to y_metric

    plt.tight_layout()
    fig_name = f'scatter_plot_{formatted_x_metric}_vs_{formatted_y_metric}_{stat_label}.png'
    fig.savefig(os.path.join(output_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {output_dir}.")
    plt.close(fig)


# Main function to create scatter plots for class/relation, raw/clean statistics
def execute_visualization_scatter(dataset, output_dir, plot_medians: bool = True):
    """
    Create scatter plots for Global Relative Frequency (Occurrence-wise) vs Global Relative Frequency (Group-wise)
    for both class and relation statistics, in raw and clean forms.

    :param dataset: Dataset object containing models and statistics.
    :param output_dir: Directory to save the generated scatter plots.
    :param plot_medians: Boolean flag to determine if medians should be plotted.
    """

    # Define the statistics types and corresponding dataset attributes to loop through
    stat_types = [
        ('class_statistics_raw', 'Class Raw'),
        ('class_statistics_clean', 'Class Clean'),
        ('relation_statistics_raw', 'Relation Raw'),
        ('relation_statistics_clean', 'Relation Clean')
    ]

    # Define the x and y metrics to be plotted
    x_metric = 'Global Relative Frequency (Occurrence-wise)'
    y_metric = 'Global Relative Frequency (Group-wise)'

    # Iterate through each statistics type and plot the scatter plot
    for stat_attr, stat_label in stat_types:
        stats = getattr(dataset, stat_attr)
        if "frequency_analysis" not in stats:
            logger.warning(f"No frequency analysis data for {stat_label}. Skipping.")
            continue

        # Convert the frequency analysis data into a DataFrame
        df = pd.DataFrame(stats["frequency_analysis"])

        # Check if the necessary metrics exist in the DataFrame
        if x_metric not in df.columns or y_metric not in df.columns:
            logger.warning(f"Missing metrics in {stat_label}: {x_metric}, {y_metric}. Skipping.")
            continue

        # Plot the scatter plot for the current statistics type
        plot_scatter_plot(df, 'Stereotype', x_metric, y_metric, stat_label, output_dir, plot_medians)
