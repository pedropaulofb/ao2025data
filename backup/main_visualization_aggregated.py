import os

from loguru import logger

from backup.src.visualization.scatter_generic import execute_visualization_scatter

if __name__ == "__main__":
    base_in_dir = "outputs/statistics/"
    base_out_dir = "outputs/visualizations/"
    input_file = "aggregated_basic_statistics.csv"

    # List comprehension to get directory names and paths
    input_information = [(f, os.path.join(base_in_dir, f, input_file)) for f in os.listdir(base_in_dir) if
                         os.path.isdir(os.path.join(base_in_dir, f)) and f.startswith("cs_")]

    # Iterate over the list of tuples
    for analysis, input_file_path in input_information:
        out_dir = os.path.join(base_out_dir, analysis, "aggregated")

        # Create the 'aggregated' folder if it does not exist
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        logger.info(f"Generating scatter plots to {analysis}.")
        execute_visualization_scatter(input_file_path, out_dir, False)
