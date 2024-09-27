import os
import re
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from loguru import logger

def calculate_distance_from_origin_and_save(df, x_metric, y_metric, out_dir_path, index_col):
    # Create a list to hold the results
    results = []

    # Iterate over each index (construct/group)
    for index_value in df[index_col].unique():
        subset = df[df[index_col] == index_value]

        # Get the x and y values for this index
        x, y = subset[x_metric].values[0], subset[y_metric].values[0]

        # Calculate the distance from the origin (0, 0)
        distance = (x**2 + y**2)**0.5

        # Append the results as a dictionary
        results.append({index_col: index_value, 'DistanceFromOrigin': distance})

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    formatted_x_metric = format_metric_name(x_metric)  # Apply formatting to x_metric
    formatted_y_metric = format_metric_name(y_metric)  # Apply formatting to y_metric

    # Save the results to a CSV file
    output_file = os.path.join(out_dir_path, f'distance_from_origin_{formatted_x_metric}_vs_{formatted_y_metric}.csv')
    results_df.to_csv(output_file, index=False, sep=',')

    logger.success(f"Distance from origin saved to {output_file}.")

    return results_df


def calculate_quadrants_and_save(df, x_metric, y_metric, out_dir_path, index_col):
    # Calculate median values for x and y metrics
    median_y = df[y_metric].median()
    median_x = df[x_metric].median()

    # Function to determine the quadrant for a given point
    def get_quadrant(x, y, median_x, median_y):
        if x >= median_x and y >= median_y:
            return 'Q1'  # Top right
        elif x < median_x and y >= median_y:
            return 'Q2'  # Top left
        elif x < median_x and y < median_y:
            return 'Q3'  # Bottom left
        else:
            return 'Q4'  # Bottom right

    # Create a list to hold the results
    results = []

    # Iterate over each construct/group and determine its quadrant
    for index_value in df[index_col].unique():
        subset = df[df[index_col] == index_value]

        # Get the x and y values for this index
        x, y = subset[x_metric].values[0], subset[y_metric].values[0]

        # Determine the quadrant
        quadrant = get_quadrant(x, y, median_x, median_y)

        # Append the results as a dictionary
        results.append({index_col: index_value, 'x_value': x, 'y_value': y, 'quadrant': quadrant})

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file
    output_file = os.path.join(out_dir_path, f'quadrant_analysis_{x_metric}_vs_{y_metric}.csv')
    results_df.to_csv(output_file, index=False)

    logger.success(f"Quadrant analysis saved to {output_file}.")

    return results_df  # Return the result for further use


def format_metric_name(metric):
    # Convert to lowercase
    formatted_metric = metric.lower().strip()
    # Replace any sequence of one or more spaces or underscores with a single underscore
    formatted_metric = re.sub(r'[\s_]+', '_', formatted_metric)
    # Remove text inside parentheses (including parentheses)
    formatted_metric = re.sub(r'\s*\(.*?\)', '', formatted_metric)
    # Remove any leading or trailing underscores
    formatted_metric = formatted_metric.strip('_')
    return formatted_metric




# Function to create all possible scatter plots for the metric combinations
def execute_visualization_scatter(in_file_path, out_dir_path, plot_medians: bool = True):
    # Read the CSV file
    df = pd.read_csv(in_file_path)

    # Identify the first column as the index column (for constructs/groups, etc.)
    index_col = df.columns[0]  # Dynamically get the first column name
    df[index_col] = pd.Categorical(df[index_col], categories=df[index_col].unique(), ordered=True)

    # Get all numeric columns (excluding the first column)
    numeric_columns = df.select_dtypes(include='number').columns

    # Generate all unique combinations of two numeric columns
    for x_metric, y_metric in combinations(numeric_columns, 2):
        fig = plt.figure(figsize=(16, 9), tight_layout=True)

        # Define the base colors for the plot (12 distinct colors)
        base_palette = sns.color_palette('tab10', n_colors=12)
        extended_palette = base_palette + base_palette[:11]  # 12 colors + 11 more to make 23 total

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
                        color=extended_palette[i], marker='o', s=100, edgecolor='w', label=index_value)

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
        # calculate_quadrants_and_save(df, x_metric, y_metric, out_dir_path, index_col)

        # Call the 'distance from center' calculation and save the results
        calculate_distance_from_origin_and_save(df, x_metric, y_metric, out_dir_path, index_col)

        plt.tight_layout()
        fig_name = f'scatter_plot_{formatted_x_metric}_vs_{formatted_y_metric}.png'
        fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
        logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
        plt.close(fig)
