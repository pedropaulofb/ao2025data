import os

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import mode


# Function to calculate group statistics (support, coverage, total occurrences, and average occurrences)
def calculate_basic_statistics(df):
    total_models = len(df)  # Get the total number of models
    results_dict = {}

    # Iterate through all columns (excluding the 'model' column)
    for column in df.columns[1:]:  # Skip the 'model' column
        # Check if the element is present in each model (at least one non-zero occurrence)
        group_present_in_models = (df[column] > 0)

        # Support: Count of models where the element occurs at least once
        support_count = group_present_in_models.sum()

        # Coverage: Support / Total Models * 100
        coverage_percentage = support_count / total_models * 100

        # Total Occurrences: Sum of all occurrences of the element across all models
        total_occurrences = df[column].sum()

        # Average Occurrences per Model: Total Occurrences / Total Models
        average_occurrences_per_model = total_occurrences / total_models

        # Store the computed statistics in the dictionary
        results_dict[column] = {
            'Support (Model Count)': support_count,
            'Coverage (%)': coverage_percentage,
            'Total Occurrences': total_occurrences,
            'Average Occurrences per Model': average_occurrences_per_model
        }

    # Convert the results dictionary to a DataFrame
    results_df = pd.DataFrame.from_dict(results_dict, orient='index').reset_index()
    results_df.columns = ['Group', 'Support (Model Count)', 'Coverage (%)', 'Total Occurrences',
                          'Average Occurrences per Model']

    return results_df


# Function to calculate advanced statistics for each element
def calculate_advanced_statistics(df):
    results_dict = {}

    # Iterate through all columns (excluding the 'model' column)
    for column in df.columns[1:]:  # Skip the 'model' column
        # Get occurrences for this element
        occurrences = df[column]

        # Max Occurrences
        max_occurrences = occurrences.max()

        # Min Occurrences (including zeros)
        min_occurrences = occurrences.min()

        # Min Non-zero Occurrences
        min_non_zero_occurrences = occurrences[occurrences > 0].min() if (occurrences > 0).any() else 0

        # Range (Max - Min)
        range_occurrences = max_occurrences - min_occurrences

        # Range Non-zero (Max - Min Non-zero)
        range_non_zero = max_occurrences - min_non_zero_occurrences

        # Median Occurrences
        median_occurrences = occurrences.median()

        # Mode Occurrences
        mode_result = mode(occurrences)
        # Handle both scalar and array cases
        mode_occurrences = mode_result.mode[0] if np.ndim(mode_result.mode) > 0 else mode_result.mode

        # Standard Deviation (SD)
        std_deviation = occurrences.std()

        # Variance
        variance = occurrences.var()

        # Quartiles (Q1, Q2, Q3)
        q1 = np.percentile(occurrences, 25)
        q2 = np.percentile(occurrences, 50)  # This is equivalent to the median
        q3 = np.percentile(occurrences, 75)

        # Interquartile Range (IQR)
        iqr = q3 - q1

        # Skewness
        skewness = occurrences.skew()

        # Kurtosis
        kurtosis = occurrences.kurtosis()

        # Store all the metrics in the dictionary
        results_dict[column] = {
            'Max Occurrences': max_occurrences,
            'Min Occurrences': min_occurrences,
            'Min Non-zero Occurrences': min_non_zero_occurrences,
            'Range (Max - Min)': range_occurrences,
            'Range Non-zero': range_non_zero,
            'Median Occurrences': median_occurrences,
            'Mode Occurrences': mode_occurrences,
            'Standard Deviation': std_deviation,
            'Variance': variance,
            'Q1 (25th Percentile)': q1,
            'Q2 (Median)': q2,
            'Q3 (75th Percentile)': q3,
            'IQR (Q3 - Q1)': iqr,
            'Skewness': skewness,
            'Kurtosis': kurtosis
        }

    # Convert the results dictionary to a DataFrame
    results_df = pd.DataFrame.from_dict(results_dict, orient='index').reset_index()
    results_df.columns = ['Group', 'Max Occurrences', 'Min Occurrences', 'Min Non-zero Occurrences',
                          'Range', 'Range Non-zero', 'Median Occurrences', 'Mode Occurrences',
                          'Standard Deviation', 'Variance', 'Q1', 'Q2',
                          'Q3', 'IQR', 'Skewness', 'Kurtosis']

    return results_df


if __name__ == "__main__":
    # Load the input CSV file (the one generated before with aggregated data)
    input_dir = 'outputs/consolidated_data/aggregated/'
    input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    output_base_dir = "outputs/statistics/"

    filters = ["t", "f"]

    for input_file in input_files:

        input_file_path = os.path.join(input_dir, input_file)
        analysis = input_file.replace(".csv", "")
        logger.info(f"Generating statistics for {input_file}.")

        for filter in filters:

            if filter == "t":
                df = pd.read_csv(input_file_path)
                df = df.drop(columns=["none", "other", "undef"])
            else:
                df = pd.read_csv(input_file_path)

            # Calculate the group coverage based on the CSV headers
            basic_df = calculate_basic_statistics(df)

            # Save the result to a new CSV file
            output_dir = os.path.join(output_base_dir, analysis + "_" + filter)
            output_file_path = os.path.join(output_dir, "aggregated_basic_statistics.csv")

            basic_df.to_csv(output_file_path, index=False)
            logger.success(f"Aggregated basic statistics successfully saved in {output_file_path}.")

            # Calculate advanced statistics for each element
            advanced_df = calculate_advanced_statistics(df)

            # Save the result to a new CSV file
            output_file_path = os.path.join(output_dir, "aggregated_advanced_statistics.csv")
            advanced_df.to_csv(output_file_path, index=False)
            print(f"Aggregated advanced statistics successfully saved in {output_file_path}.")
