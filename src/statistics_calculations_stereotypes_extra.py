import os

import pandas as pd
from icecream import ic
from loguru import logger

from src.utils import format_metric_name


def classify_and_save_spearman_correlations(spearman_correlation: pd.DataFrame, output_filepath: str) -> None:
    """
    Classify Spearman correlation values and save them to a CSV file.

    :param spearman_correlation: DataFrame with Spearman correlation values.
    :param output_filepath: Filepath to save the classified correlations CSV.
    """
    # Melt the DataFrame to get pairwise correlations (excluding self-correlations)
    correlations = spearman_correlation.melt(id_vars=['Stereotype'], var_name='Stereotype2', value_name='Correlation')
    correlations = correlations[correlations['Stereotype'] != correlations['Stereotype2']]  # Exclude diagonal

    # Function to classify correlation based on absolute value
    def classify_correlation(value):
        abs_value = abs(value)
        if abs_value < 0.2:
            return 'very_weak'
        elif 0.2 <= abs_value < 0.4:
            return 'weak'
        elif 0.4 <= abs_value < 0.6:
            return 'moderate'
        elif 0.6 <= abs_value < 0.8:
            return 'strong'
        else:
            return 'very_strong'

    # Add classification and other columns
    correlations['Absolute Value'] = correlations['Correlation'].abs()
    correlations['Sign'] = correlations['Correlation'].apply(lambda x: 'positive' if x > 0 else 'negative')
    correlations['Classification'] = correlations['Correlation'].apply(classify_correlation)

    # Select and rename columns for the final output
    output_df = correlations[['Stereotype', 'Stereotype2', 'Absolute Value', 'Sign', 'Classification']]

    # Save the result to a CSV file
    output_df.to_csv(output_filepath, index=False)


def calculate_quadrants_and_save(df, x_metric, y_metric, out_dir_path) -> pd.DataFrame:
    """
    Perform quadrant analysis based on the provided x and y metrics, calculate the distance from the origin,
    and save the result to a CSV file. The result includes a rank for the distance from the origin.

    :param df: DataFrame containing the data.
    :param x_metric: Name of the column to be used as the x-axis metric.
    :param y_metric: Name of the column to be used as the y-axis metric.
    :param out_dir_path: Directory where the output CSV file should be saved.
    :return: DataFrame with quadrant analysis results.
    """

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

    # Function to calculate the distance from origin (0, 0)
    def calculate_distance_from_origin(x, y):
        return (x**2 + y**2) ** 0.5

    # Create a list to hold the results
    results = []

    # Iterate over each row in the DataFrame and determine its quadrant and distance from the origin
    for index, row in df.iterrows():
        x, y = row[x_metric], row[y_metric]
        quadrant = get_quadrant(x, y, median_x, median_y)
        distance_from_origin = calculate_distance_from_origin(x, y)

        # Append the results as a dictionary, including the stereotype (index)
        results.append({
            'Stereotype': index,
            'x_value': x,
            'y_value': y,
            'distance_from_origin': distance_from_origin,
            'quadrant': quadrant
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Add a rank column based on the distance_from_origin, with 1 being the farthest to the origin
    results_df['rank'] = results_df['distance_from_origin'].rank(ascending=False).astype(int)

    formatted_x_metric = format_metric_name(x_metric)  # Apply formatting to x_metric
    formatted_y_metric = format_metric_name(y_metric)  # Apply formatting to y_metric

    # Ensure the output directory exists
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    # Save the results to a CSV file with the requested columns
    output_file = os.path.join(out_dir_path, f'quadrant_analysis_{formatted_x_metric}_vs_{formatted_y_metric}.csv')
    results_df.to_csv(output_file, index=False, columns=['Stereotype', 'x_value', 'y_value', 'distance_from_origin', 'rank', 'quadrant'])

    logger.success(f"Quadrant analysis saved to {output_file}.")

    return results_df

