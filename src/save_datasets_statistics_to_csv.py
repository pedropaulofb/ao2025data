import csv
import os


def save_datasets_statistics_to_csv(datasets: list, output_csv_dir: str) -> None:
    """
    Save statistics from a list of Dataset objects to a CSV file.

    :param datasets: List of Dataset objects.
    :param output_csv_dir: Directory in which the output CSV file will be saved.
    """
    # Define the output file path
    output_path = os.path.join(output_csv_dir, "datasets_statistics.csv")

    # Open the CSV file for writing
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Collect all keys from the first dataset's statistics as headers
        # Assuming that all datasets have the same statistics structure
        if datasets:
            all_keys = list(datasets[0].statistics.keys())

        # Write the header, adding "number_models" and "dataset" at the start
        writer.writerow(['dataset', 'number_models'] + all_keys)

        # Write the statistics for each dataset
        for dataset in datasets:
            row = [dataset.name, len(dataset.models)]  # Start with dataset name and number of models

            # Add the statistics for this dataset
            row.extend([dataset.statistics.get(key, 'N/A') for key in all_keys])

            # Write the row to the CSV file
            writer.writerow(row)

    print(f"Datasets' statistics successfully saved in {output_path}.")
