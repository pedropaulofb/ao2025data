import os

from loguru import logger

from src.visualization.movement_analysis import execute_visualization_movement

if __name__ == "__main__":
    base_in_dir = "outputs/statistics/"
    base_out_dir = "outputs/visualizations/"
    base_mid_dir = "_ontouml_no_classroom_"
    input_data = "frequency_analysis.csv"

    stereotype_types = ["cs", "rs"]
    filters = ["t", "f"]

    for stereotype_type in stereotype_types:
        for filter in filters:
            analysis = stereotype_type + base_mid_dir + filter
            input_file_a = os.path.join(base_in_dir, stereotype_type + base_mid_dir + "until_2017_" + filter,
                                        input_data)
            input_file_b = os.path.join(base_in_dir, stereotype_type + base_mid_dir + "after_2018_" + filter,
                                        input_data)
            output_dir = os.path.join(base_out_dir, analysis, "movement")

            logger.info(f"Generating movement for {analysis}.")
            execute_visualization_movement(input_file_a, input_file_b, output_dir, plot_medians=True)
