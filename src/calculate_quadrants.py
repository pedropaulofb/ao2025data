import os

import pandas as pd
from loguru import logger

from src.utils import format_metric_name


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

    # Iterate over each stereotype/group and determine its quadrant
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

    formatted_x_metric = format_metric_name(x_metric)  # Apply formatting to x_metric
    formatted_y_metric = format_metric_name(y_metric)  # Apply formatting to y_metric

    # Save the results to a CSV file
    output_file = os.path.join(out_dir_path, f'quadrant_analysis_{formatted_x_metric}_vs_{formatted_y_metric}.csv')
    results_df.to_csv(output_file, index=False)

    logger.success(f"Quadrant analysis saved to {output_file}.")

    return results_df
