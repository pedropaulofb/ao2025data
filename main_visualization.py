import os

from icecream import ic
from loguru import logger

from src.visualization.distance_from_origin import plot_distance_from_origin

if __name__ == "__main__":
    base_in_dir = "outputs/visualizations/"
    base_out_dir = "outputs/visualizations/"
    input_files = ["distance_from_origin_global_relative_frequency_vs_global_relative_frequency.csv","distance_from_origin_total_frequency_vs_group_frequency.csv"]

    input_information = []
    for input_file in input_files:
        # List comprehension to get directory names and paths
        input_information.extend([(f, os.path.join(base_in_dir, f, input_file)) for f in os.listdir(base_in_dir) if
                             os.path.isdir(os.path.join(base_in_dir, f))])

    # Iterate over the list of tuples
    for analysis, input_file_path in input_information:
        out_dir = os.path.join(base_out_dir, analysis)

        # Create folder if it does not exist
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        logger.info(f"Generating distance plots to {analysis}.")
        plot_distance_from_origin(input_file_path, out_dir)
