import os

import pandas as pd
from loguru import logger

from src.utils import format_metric_name


def calculate_distance_from_origin_and_save(df, x_metric, y_metric, out_dir_path, index_col):
    # Create a list to hold the results
    results = []

    # Iterate over each index (stereotype/group)
    for index_value in df[index_col].unique():
        subset = df[df[index_col] == index_value]

        # Get the x and y values for this index
        x, y = subset[x_metric].values[0], subset[y_metric].values[0]

        # Calculate the distance from the origin (0, 0)
        distance = (x ** 2 + y ** 2) ** 0.5

        # Append the results as a dictionary
        results.append({index_col: index_value, 'DistanceFromOrigin': distance})

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Rank the results based on distance from origin (farthest ranked first)
    results_df['Rank'] = results_df['DistanceFromOrigin'].rank(ascending=False, method='dense').astype(int)

    # Sort the DataFrame by rank for better readability (optional)
    results_df = results_df.sort_values(by='Rank')

    formatted_x_metric = format_metric_name(x_metric)  # Apply formatting to x_metric
    formatted_y_metric = format_metric_name(y_metric)  # Apply formatting to y_metric

    # Save the results to a CSV file
    output_file = os.path.join(out_dir_path, f'distance_from_origin_{formatted_x_metric}_vs_{formatted_y_metric}.csv')
    results_df.to_csv(output_file, index=False, sep=',')

    logger.success(f"Distance from origin saved to {output_file}.")

    return results_df
