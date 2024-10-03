import csv

import pandas as pd
from loguru import logger

from src.ClassStereotypesData import ClassStereotypesData
from src.RelationStereotypesData import RelationStereotypesData
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


def consolidate_stereotypes_data(type: str, list_models: str, stereotypes_data_file_path: str, type_data_file_path: str,
                                 out_path: str) -> None:
    # Dictionary to store ClassStereotypesData or RelationStereotypesData instances by model_id
    instances = {}

    # Step 1: Read the `list_models` file (assumed to be a text file) and create instances for each model_id
    with open(list_models, 'r') as txtfile:
        for line in txtfile:
            model_id = line.strip()  # Strip any extra whitespace or newline characters
            if type == 'class':
                instances[model_id] = ClassStereotypesData(name=model_id)  # Initialize instances
            elif type == 'relation':
                instances[model_id] = RelationStereotypesData(name=model_id)  # Initialize instances
            else:
                logger.error("Unrecognized type.")
                exit(1)

    # Step 2: Read the CSV file and populate the already initialized instances
    with open(stereotypes_data_file_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)

        for row in csvreader:
            model_id = row['model_id']
            stereotype = row['stereotype']
            count = int(row['count'])

            # Ensure the model_id exists in the `instances` dict (it should be pre-populated from list_models)
            if model_id in instances:
                if stereotype in instances[model_id].stats:
                    # Update the existing stat if the stereotype matches an attribute
                    instances[model_id].stats[stereotype] = count
                else:
                    # If stereotype does not match an attribute, sum the value to 'other'
                    instances[model_id].stats['other'] += count
            else:
                logger.warning(f"Model ID {model_id} from CSV not found in the list from {list_models}")

    # Step 3: Process the type_data_file and write results to the output
    process_none(type_data_file_path, instances)
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
    consolidate_stereotypes_data("class",
                                 "./outputs/loaded_models/ontouml_no_classroom_after_2015.txt",
                                 "./outputs/queries_results/ontouml_no_classroom_after_2015/query_cs_consolidated.csv",
                                 "./outputs/queries_results/ontouml_no_classroom_after_2015/query_c_consolidated.csv",
                                 "outputs/consolidated_data/cs_ontouml_no_classroom_after_2015.csv")
    consolidate_stereotypes_data("relation",
                                 "./outputs/loaded_models/ontouml_no_classroom_after_2015.txt",
                                 "./outputs/queries_results/ontouml_no_classroom_after_2015/query_rs_consolidated.csv",
                                 "./outputs/queries_results/ontouml_no_classroom_after_2015/query_r_consolidated.csv",
                                 "outputs/consolidated_data/rs_ontouml_no_classroom_after_2015.csv")
