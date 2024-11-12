import os

from src.calculations.quadrants_temporal import compare_and_generate_quadrant_csv
from src.directories_global import OUTPUT_DIR_02, OUTPUT_DIR_03
from src.step0_setup import initialize_output_directories, get_catalog_path
from src.step1_input import load_data_from_catalog, query_data, calculate_models_data, \
    create_and_save_specific_datasets_instances
from src.step2_processing import calculate_and_save_datasets_statistics, \
    calculate_and_save_datasets_stereotypes_statistics
from src.step3_output import generate_visualizations
from src.utils import save_object


def calculate_and_save_datasets_quadrants():
    dataset_types = [""]

    compared_datasets = []
    for dataset_type in dataset_types:
        d_before = "ontouml_non_classroom_until_2018" + dataset_type
        d_after = "ontouml_non_classroom_after_2019" + dataset_type
        d_general = "ontouml_non_classroom" + dataset_type
        compared_datasets.append((d_before, d_after, d_general))

    out_file_name = "quadrants_movement_2018_2019"

    st_types = ['class', 'relation', 'combined']
    st_cases = ['raw', 'clean']

    for d_before, d_after, d_general in compared_datasets:

        for st_type in st_types:
            for st_case in st_cases:
                output_file_path = os.path.join(OUTPUT_DIR_02, d_general, f"{st_type}_{st_case}",
                                                f"{out_file_name}.csv")

                q_before_path = os.path.join(OUTPUT_DIR_02, d_before, f"{st_type}_{st_case}",
                                             "quadrant_analysis_global_relative_frequency_vs_ubiquity_index.csv")
                q_after_path = os.path.join(OUTPUT_DIR_02, d_after, f"{st_type}_{st_case}",
                                            "quadrant_analysis_global_relative_frequency_vs_ubiquity_index.csv")
                q_general_path = os.path.join(OUTPUT_DIR_02, d_general, f"{st_type}_{st_case}",
                                              "quadrant_analysis_global_relative_frequency_vs_ubiquity_index.csv")

                compare_and_generate_quadrant_csv(output_file_path, q_before_path, q_after_path, q_general_path)


if __name__ == "__main__":
    # Step 0: Initial setup
    initialize_output_directories()
    catalog_path = get_catalog_path()

    # Step 1: Data input - load all models' data, execute queries and create dataset
    all_models_data = load_data_from_catalog(catalog_path)
    query_data(all_models_data)
    all_models_data = calculate_models_data()
    datasets = create_and_save_specific_datasets_instances(all_models_data)

    # Step 2: Data processing - generate statistics
    calculate_and_save_datasets_statistics(datasets, OUTPUT_DIR_02)
    calculate_and_save_datasets_stereotypes_statistics(datasets, OUTPUT_DIR_02)
    calculate_and_save_datasets_quadrants()
    save_object(datasets, OUTPUT_DIR_02, "datasets", "Updated datasets")

    # Step 3: Data output - visualizations
    generate_visualizations(datasets, OUTPUT_DIR_03)
