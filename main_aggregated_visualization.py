import os

from loguru import logger

base_in_dir_path = "outputs/statistics/"
base_out_dir_path = "outputs/visualizations/"
statistics_name = "aggregated.csv"

analysis_types = ["cs", "rs"]
filter_types = ["t", "f"]


for analysis in analysis_types:
    for filter in filter_types:
        in_dir_path = base_in_dir_path + analysis + "_ontouml_no_classroom_" + filter
        out_dir_path = base_out_dir_path + analysis + "_ontouml_no_classroom_" + filter

        # Create the 'aggregated' folder if it does not exist
        if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)

        logger.info(f"Starting plot for {in_dir_path}.")