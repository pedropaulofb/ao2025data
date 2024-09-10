from pathlib import Path
from typing import Union

from src.OntoUMLElement import ModelStats
from src.generate_stats import read_csv_and_update_instances, write_stats_to_csv


def main() -> None:
    """
    Main function to process multiple CSV files, update ModelStats instances, and write the results to an output CSV.
    """
    # Dictionary to hold ModelStats instances by their name
    models_stats: dict[str, ModelStats] = {}

    # Read and process the CSV files
    csv_files: list[Union[str, Path]] = [
        "./outputs/queries_results/query_o_consolidated.csv",
        './outputs/queries_results/query_p_consolidated.csv'
    ]

    for file_path in csv_files:
        read_csv_and_update_instances(file_path, models_stats)

    # Specify the output CSV file path
    output_file = "./outputs/models_stats.csv"

    # Write the results to the output CSV file
    write_stats_to_csv(models_stats, output_file)


if __name__ == "__main__":
    main()