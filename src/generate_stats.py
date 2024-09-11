import csv
from pathlib import Path
from typing import Union

from src.SPOStats import SPOStats


def read_csv_and_update_instances(file_path: Union[str, Path], instances: dict[str, SPOStats]) -> None:
    """
    Reads a CSV file and updates ModelStats instances based on the CSV content.

    :param file_path: Path to the CSV file.
    :type file_path: Union[str, Path]
    :param instances: Dictionary holding instances of ModelStats by name.
    :type instances: dict[str, SPOStats]

    :raises FileNotFoundError: If the specified CSV file path does not exist.
    :raises ValueError: If the CSV contains invalid data that cannot be processed.
    """
    try:
        with open(file_path, mode='r', newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile, delimiter=',')

            # Check if required columns exist in the CSV
            if 'model_id' not in csv_reader.fieldnames or 'element' not in csv_reader.fieldnames or 'count' not in csv_reader.fieldnames:
                raise ValueError(f"CSV file {file_path} is missing required headers: 'model_id', 'element', 'count'.")

            for row in csv_reader:
                model_id: str = row['model_id']
                element: str = row['element']
                count: int = int(row['count'])  # Convert value to integer

                # Check if the instance with the given name already exists
                if model_id in instances:
                    # If it exists, update the stats dictionary
                    if element in instances[model_id].stats:
                        instances[model_id].stats[element] += count
                else:
                    # Create a new instance if it does not exist
                    new_instance = SPOStats(name=model_id)
                    if element in new_instance.stats:
                        new_instance.stats[element] = count
                    instances[model_id] = new_instance
    except FileNotFoundError as e:
        print(f"Error: The file {file_path} does not exist.")
        raise e
    except ValueError as e:
        print(f"Error: Invalid data in the CSV file {file_path}. {e}")
        raise e

def write_stats_to_csv(instances: dict, output_file: Union[str, Path]) -> None:
    """
    Writes the statistics from all ModelStats instances to a CSV file.

    :param instances: Dictionary holding instances of SPOStats by name.
    :type instances: dict[str, SPOStats]
    :param output_file: Path to the output CSV file.
    :type output_file: Union[str, Path]

    :raises IOError: If the file cannot be written.
    """
    try:
        with open(output_file, mode='w', newline='') as csvfile:
            # Prepare CSV writer
            csv_writer = csv.writer(csvfile, delimiter=',')

            # Write header
            headers = ['model'] + list(next(iter(instances.values())).stats.keys())
            csv_writer.writerow(headers)

            # Write data for each instance
            for name, instance in instances.items():
                row = [name] + [instance.stats[attr] for attr in instance.stats]
                csv_writer.writerow(row)

    except IOError as e:
        print(f"Error: Could not write to file {output_file}.")
        raise e