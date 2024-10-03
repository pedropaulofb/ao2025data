import os

from icecream import ic

from src.visualization.non_ontouml_analysis_combined import generate_non_ontouml_combined_visualization
from src.visualization.non_ontouml_analysis_visualization import generate_non_ontouml_visualization

if __name__ == "__main__":
    base_in_dir = "outputs/statistics/"
    base_out_dir = r"outputs/visualizations/"
    analyses = ["cs_ontouml_no_classroom_f", "rs_ontouml_no_classroom_f"]
    input_files = ["temporal_overall_modelwise_stats.csv","temporal_overall_stats.csv","temporal_yearly_modelwise_stats.csv","temporal_yearly_stats.csv"]

    combined_in_files = [("temporal_overall_stats.csv", "temporal_overall_modelwise_stats.csv"),
                         ("temporal_yearly_stats.csv", "temporal_yearly_modelwise_stats.csv")]

    for analysis in analyses:

        output_dir_path = os.path.join(base_out_dir, analysis, "non_ontouml")
        output_dir_path = os.path.normpath(output_dir_path)

        input_file_path = os.path.join(base_in_dir, analysis)

        for combined_in_file_ow, combined_in_file_mw  in combined_in_files:
            generate_non_ontouml_combined_visualization(input_file_path, output_dir_path, combined_in_file_ow, combined_in_file_mw)

        for input_file in input_files:
            generate_non_ontouml_visualization(input_file_path, output_dir_path, input_file)

