import csv

import pandas as pd
from loguru import logger

from src.CSStats import CSStats
from src.SPOStats import SPOStats
from src.generate_stats import read_csv_and_update_instances, write_stats_to_csv


def consolidate_spo() -> None:
    """
    Main function to process multiple CSV files, update SPOStats instances, and write the results to an output CSV.
    """
    # Dictionary to hold SPOStats instances by their name
    models_stats: dict[str, SPOStats] = {}

    read_csv_and_update_instances('./outputs/queries_results/query_spo_consolidated.csv', models_stats)

    # Specify the output CSV file path
    output_file = "outputs/consolidated_spo.csv"

    # Write the results to the output CSV file
    write_stats_to_csv(models_stats, output_file)
    logger.success(f"Successfully saved file: {output_file}")


def consolidate_s(specific_file_path: str, general_file_path: str, out_path: str) -> None:
    # Dictionary to store CSStats instances by model_id
    instances = {}

    # Function to process the CSV file and create/update instances
    with open(specific_file_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)

        for row in csvreader:
            model_id = row['model_id']
            stereotype = row['stereotype']
            count = int(row['count'])

            # Check if an instance already exists for this model_id
            if model_id not in instances:
                # Create a new instance if it doesn't exist
                instances[model_id] = CSStats(name=model_id)

            # Update the appropriate attribute in the existing instance
            if stereotype in instances[model_id].stats:
                # If stereotype matches an attribute, update it
                instances[model_id].stats[stereotype] = count
            else:
                # If stereotype does not match, sum the value to 'other'
                instances[model_id].stats['other'] += count

    process_none(general_file_path, instances)
    write_stats_to_csv(instances, out_path)
    logger.success(f"Successfully saved file: {out_path}")


# Function to process the second CSV file and calculate 'none' attribute
def process_none(file_path: str, instances: dict) -> None:
    with open(file_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)

        for row in csvreader:
            model_id = row['model_id']
            total_count = int(row['count'])

            # Check if an instance exists for this model_id
            if model_id in instances:
                # Calculate the sum of all attributes except 'none'
                total_attributes_sum = sum(value for key, value in instances[model_id].stats.items() if key != 'none')

                # Calculate 'none' as the difference
                instances[model_id].stats['none'] = total_count - total_attributes_sum


def filter_csv_by_headers(input_csv_path: str, output_csv_path: str, valid_headers: list) -> None:
    """
    Filter columns from a CSV file based on a list of valid headers while maintaining the order of columns,
    particularly preserving the first column's position, and save the modified CSV.

    :param input_csv_path: The path to the input CSV file.
    :param output_csv_path: The path to save the modified CSV file.
    :param valid_headers: A list of headers that should be kept in the CSV file.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv_path)

    # Get the original order of columns
    original_columns = df.columns.tolist()

    # Create a list of columns to retain, preserving their original order
    filtered_columns = [col for col in original_columns if col in valid_headers]

    # Filter the DataFrame to keep only the valid columns in their original order
    filtered_df = df[filtered_columns]

    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_csv_path, index=False)
    logger.success(f"Successfully saved file: {output_csv_path}")


if __name__ == "__main__":

    consolidate_s("./outputs/queries_results/query_cs_consolidated.csv",
                  "./outputs/queries_results/query_c_consolidated.csv", "./outputs/consolidated_cs.csv")
    consolidate_s("./outputs/queries_results/query_rs_consolidated.csv",
                  "./outputs/queries_results/query_r_consolidated.csv", "./outputs/consolidated_rs.csv")

# TODO: If the number of groups is not 121, then I need to manually insert the missing ones for correct calculations.
