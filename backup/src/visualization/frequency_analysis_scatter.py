import os
import re
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from loguru import logger

from backup.src.utils import color_text


def calculate_quadrants_and_save(df, x_metric, y_metric, out_dir_path):
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

    # Iterate over each stereotype and determine its quadrant
    for stereotype in df['Stereotype'].unique():
        subset = df[df['Stereotype'] == stereotype]

        # Get the x and y values for this stereotype
        x, y = subset[x_metric].values[0], subset[y_metric].values[0]

        # Determine the quadrant
        quadrant = get_quadrant(x, y, median_x, median_y)

        # Append the results as a dictionary
        results.append({'stereotype': stereotype, 'x_value': x, 'y_value': y, 'quadrant': quadrant})

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file
    output_file = os.path.join(out_dir_path, f'quadrant_analysis_{x_metric}_vs_{y_metric}.csv')
    results_df.to_csv(output_file, index=False)

    logger.success(f"Quadrant analysis saved to {output_file}.")

    return results_df  # Return the result for further use


# Function to format the metric name
def format_metric_name(metric):
    # Convert to lowercase
    formatted_metric = metric.lower()
    # Replace spaces with underscores
    formatted_metric = formatted_metric.replace(' ', '_')
    # Remove text inside parentheses (including parentheses)
    formatted_metric = re.sub(r'\s*\(.*?\)', '', formatted_metric)
    return formatted_metric


# Function to create all possible scatter plots for the metric combinations
def execute_visualization_frequency_analysis_scatter(in_dir_path, out_dir_path, file_path, aggr: bool = False):
    base_name = os.path.splitext(file_path)[0] + "_" if aggr else ""

    df = pd.read_csv(os.path.join(in_dir_path, file_path))

    # Filter out rows (stereotypes) with zero occurrences in 'Total Frequency' (or other relevant metrics)
    df = df[df['Total Frequency'] > 0]

    # Convert the 'Stereotype' column to a categorical type to ensure proper ordering in the plots
    df['Stereotype'] = pd.Categorical(df['Stereotype'], categories=df['Stereotype'].unique(), ordered=True)

    # Get all numeric columns (excluding 'Stereotype')
    numeric_columns = df.select_dtypes(include='number').columns

    # Generate all unique combinations of two metrics
    for x_metric, y_metric in combinations(numeric_columns, 2):
        fig = plt.figure(figsize=(16, 9), tight_layout=True)

        # Define the base colors for the plot (12 distinct colors)
        base_palette = sns.color_palette('tab10', n_colors=12)
        extended_palette = base_palette + base_palette[:11]  # 12 colors + 11 more to make 23 total

        texts = []
        # Plotting the scatter plot with circles and adding labels to each point
        for i, stereotype in enumerate(df['Stereotype'].unique()):
            subset = df[df['Stereotype'] == stereotype]

            # Filter out rows with non-finite values in either x_metric or y_metric
            subset = subset[pd.notnull(subset[x_metric]) & pd.notnull(subset[y_metric])]

            # Skip if no valid data to plot for this stereotype
            if subset.empty:
                continue

            # Plot the point (circle marker)
            plt.scatter(subset[x_metric], subset[y_metric],
                        color=extended_palette[i], marker='o', s=100, edgecolor='w', label=stereotype)

            # Add label next to each point
            for j in range(len(subset)):
                text = plt.text(subset[x_metric].values[j],
                                subset[y_metric].values[j], stereotype, fontsize=8,
                                color='black', ha='left', va='top')
                texts.append(text)

        # Apply the color_text function to the list of text objects
        color_text(texts)

        # Adjust text to avoid overlap
        adjust_text(texts)

        # Adding labels and title
        plt.xlabel(x_metric.replace('_', ' ').title())
        plt.ylabel(y_metric.replace('_', ' ').title())
        plt.title(f'{x_metric.replace("_", " ").title()} vs. {y_metric.replace("_", " ").title()}', fontweight='bold')

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

        # Call the quadrant calculation and save the results
        calculate_quadrants_and_save(df, x_metric, y_metric, out_dir_path)

        formatted_x_metric = format_metric_name(x_metric)
        formatted_y_metric = format_metric_name(y_metric)

        plt.tight_layout()
        fig_name = f'{base_name}frequency_analysis_{formatted_x_metric}_vs_{formatted_y_metric}.png'
        fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
        logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
        plt.close(fig)
