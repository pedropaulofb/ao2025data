import os
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from loguru import logger

from backup.src.calculate_distance_from_origin import calculate_distance_from_origin_and_save
from backup.src.calculate_quadrants import calculate_quadrants_and_save
from backup.src.utils import format_metric_name


# Function to create all possible scatter plots for the metric combinations
def execute_visualization_scatter(in_file_path, out_dir_path, plot_medians: bool = True, selected_combinations=None):
    # Read the CSV file
    df = pd.read_csv(in_file_path)

    # Identify the first column as the index column (for stereotypes/groups, etc.)
    index_col = df.columns[0]  # Dynamically get the first column name
    df[index_col] = pd.Categorical(df[index_col], categories=df[index_col].unique(), ordered=True)

    # Get all numeric columns (excluding the first column)
    numeric_columns = df.select_dtypes(include='number').columns

    # Check if selected_combinations is provided, otherwise use all combinations of numeric columns
    if selected_combinations is None or not selected_combinations:
        # Generate all unique combinations of two numeric columns if no specific combinations are provided
        combinations_to_plot = list(combinations(numeric_columns, 2))
    else:
        # Use the provided list of combinations (assumed to be a list of tuples)
        combinations_to_plot = selected_combinations

    # Generate the scatter plots for each combination
    for x_metric, y_metric in combinations_to_plot:
        # Check if the x_metric and y_metric are valid numeric columns
        if x_metric not in numeric_columns or y_metric not in numeric_columns:
            logger.warning(
                f"Skipping invalid combination ({x_metric}, {y_metric}) - One or both metrics are not numeric columns.")
            continue

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
        plt.title(f'{x_metric.replace("_", " ").title()} vs. {y_metric.replace("_", " ").title()}', fontweight='bold')

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

        # Call the quadrant calculation and save the results
        calculate_quadrants_and_save(df, x_metric, y_metric, out_dir_path, index_col)

        # Call the 'distance from center' calculation and save the results
        calculate_distance_from_origin_and_save(df, x_metric, y_metric, out_dir_path, index_col)

        plt.tight_layout()
        fig_name = f'scatter_plot_{formatted_x_metric}_vs_{formatted_y_metric}.png'
        fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
        logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
        plt.close(fig)

