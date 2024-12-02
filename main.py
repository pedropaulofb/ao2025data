import os

from src.directories_global import OUTPUT_DIR_02, OUTPUT_DIR_03, OUTPUT_DIR_01
from src.step0_setup import initialize_output_directories, get_catalog_path
from src.step1_input import load_data_from_catalog, calculate_models_data, create_and_save_specific_datasets_instances, \
    query_data
from src.step2_processing import calculate_and_save_datasets_statistics, \
    calculate_and_save_datasets_stereotypes_statistics
from src.step3_output import generate_visualizations
from src.utils import save_object, load_object

if __name__ == "__main__":
    # Step 0: Initial setup
    initialize_output_directories()
    catalog_path = get_catalog_path()

    # Step 1: Data input - load all models' data, execute queries and create dataset
    all_models_data = load_data_from_catalog(catalog_path)
    query_data(all_models_data)
    all_models_data = calculate_models_data()
    datasets = create_and_save_specific_datasets_instances(all_models_data)

    # datasets = load_object(os.path.join(OUTPUT_DIR_01, "datasets.object.gz"), "Datasets")

    # Step 2: Data processing - generate statistics
    calculate_and_save_datasets_statistics(datasets, OUTPUT_DIR_02)
    calculate_and_save_datasets_stereotypes_statistics(datasets, OUTPUT_DIR_02)
    save_object(datasets, OUTPUT_DIR_02, "datasets", "Updated datasets")

    # datasets = load_object(os.path.join(OUTPUT_DIR_02, "datasets.object.gz"), "Datasets")

    # Step 3: Data output - visualizations
    generate_visualizations(datasets, OUTPUT_DIR_03)
