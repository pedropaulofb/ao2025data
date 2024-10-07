import csv
import os.path
from loguru import logger

def save_datasets_statistics_to_csv(datasets: list, output_csv_dir: str) -> None:
    """
    Save statistics from a list of Dataset objects to a CSV file.

    :param datasets: List of Dataset objects.
    :param output_csv_dir: Directory in which the output CSV file will be saved.
    """
    # Use a list to preserve insertion order and avoid duplicates
    all_keys = []

    for dataset in datasets:
        append_unique_preserving_order(all_keys, dataset.statistics.keys())

    output_path = os.path.join(output_csv_dir, "datasets_statistics.csv")

    # Open the CSV file for writing
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header (dataset, stat1, stat2, ..., statN)
        writer.writerow(['dataset'] + all_keys)

        # Write the statistics for each dataset
        for dataset in datasets:
            row = [dataset.name]  # Start the row with the dataset name

            # Add the statistics for this dataset, ensuring to handle missing values
            row.extend([dataset.statistics.get(key, 'N/A') for key in all_keys])

            # Write the row to the CSV file
            writer.writerow(row)

    logger.success(f"Datasets' statistics successfully saved in {output_csv_dir}.")


def append_unique_preserving_order(existing_list, new_keys):
    """Append keys to the list, preserving the original order and ensuring no duplicates."""
    for key in new_keys:
        if key not in existing_list:
            existing_list.append(key)
