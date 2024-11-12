import gzip
import os
import pickle

import pandas as pd
from loguru import logger

from src.directories_global import OUTPUT_DIR_02
from src.quadrants_temporal import compare_and_generate_quadrant_csv
from src.save_datasets_statistics_to_csv import save_datasets_statistics_to_csv
from src.step0_setup import initialize_output_directories, get_catalog_path
from src.step1_input import load_data_from_catalog, query_data, calculate_models_data, \
    create_and_save_specific_datasets_instances
from src.step2_processing import calculate_and_save_datasets_statistics, \
    calculate_and_save_datasets_stereotypes_statistics
from src.visualization.learning_tree import build_tree, generate_dot
from src.visualization.trend_analysis import generate_trend_visualization


def create_list_outliers(datasets, output_dir):
    # Initialize outlier lists to avoid UnboundLocalError if no outliers are found

    outliers: dict = {}

    # Identify outliers for each dataset
    for dataset in datasets:
        if 'ontouml_all' in dataset.name:
            outliers["ontouml_all_outliers"] = dataset.identify_outliers(output_dir)
        elif 'non_classroom' in dataset.name:
            outliers["ontouml_non_classroom_outliers"] = dataset.identify_outliers(output_dir)
        elif 'ontouml_classroom' in dataset.name:
            outliers["ontouml_classroom_outliers"] = dataset.identify_outliers(output_dir)
        else:
            logger.error(f"No outlier calculated to dataset {dataset.name}.")

    return outliers


def calculate_and_save_datasets_statistics_outliers(datasets, outliers, output_dir):
    # Create new datasets without the identified outliers

    filtered_datasets = []

    for dataset in datasets:
        if 'non_classroom' in dataset.name:
            filtered_datasets.append(
                dataset.fork_without_outliers(outliers["ontouml_all_outliers"], "_general_filtered"))
            filtered_datasets.append(
                dataset.fork_without_outliers(outliers["ontouml_non_classroom_outliers"], "_specific_filtered"))
        elif 'classroom' in dataset.name:
            filtered_datasets.append(
                dataset.fork_without_outliers(outliers["ontouml_all_outliers"], "_general_filtered"))
            filtered_datasets.append(
                dataset.fork_without_outliers(outliers["ontouml_classroom_outliers"], "_specific_filtered"))
        elif dataset.name == 'ontouml_all':
            filtered_datasets.append(dataset.fork_without_outliers(outliers["ontouml_all_outliers"], "_filtered"))
        else:
            logger.warning(f"Dataset {dataset.name} had no outliers cleaned.")

    # Calculate and save statistics for the datasets without outliers
    calculate_and_save_datasets_statistics(filtered_datasets, OUTPUT_DIR_02)

    # Combine original datasets with filtered ones and save combined statistics

    all_datasets = datasets + filtered_datasets

    new_outliers = create_list_outliers(filtered_datasets, output_dir)

    double_filtered_datasets = []
    for dataset in all_datasets:
        if "non_classroom" in dataset.name and "general_filtered" in dataset.name:
            new_dataset = dataset.fork_without_outliers(outliers["ontouml_non_classroom_outliers"], "_double")
            new_dataset.name = new_dataset.name.replace("_general_filtered_double", "_double_general_filtered")
            double_filtered_datasets.append(new_dataset)
            new_dataset = dataset.fork_without_outliers(new_outliers["ontouml_non_classroom_outliers"], "_double")
            new_dataset.name = new_dataset.name.replace("_general_filtered_double", "_double_specific_filtered")
            double_filtered_datasets.append(new_dataset)
        elif "classroom" in dataset.name and "general_filtered" in dataset.name:
            new_dataset = dataset.fork_without_outliers(outliers["ontouml_classroom_outliers"], "_double")
            new_dataset.name = new_dataset.name.replace("_general_filtered_double", "_double_general_filtered")
            double_filtered_datasets.append(new_dataset)
            new_dataset = dataset.fork_without_outliers(new_outliers["ontouml_classroom_outliers"], "_double")
            new_dataset.name = new_dataset.name.replace("_general_filtered_double", "_double_specific_filtered")
            double_filtered_datasets.append(new_dataset)

    calculate_and_save_datasets_statistics(double_filtered_datasets, OUTPUT_DIR_02)
    all_datasets += double_filtered_datasets

    save_datasets_statistics_to_csv(all_datasets, OUTPUT_DIR_02)

    return all_datasets


def select_root_element(in_root_file_path: str) -> str:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(in_root_file_path)

    # Ensure 'rank' column exists before proceeding
    if 'rank' not in df.columns:
        raise KeyError("'rank' column not found in the file.")

    # Filter the DataFrame to get the row where 'rank' is 1
    root_row = df[df['rank'] == 1]

    # Extract and return the 'stereotype' value from the filtered row
    if not root_row.empty:
        return root_row.iloc[0]['stereotype']
    else:
        raise ValueError("No rank 1.")


def plot_learning_tree(dataset, in_dir_path, out_dir_path):
    st_types = ['class', 'relation', 'combined']
    st_cases = ['raw', 'clean']
    analyses = ['geometric_mean', 'model_wise', 'occurrence_wise']

    for st_type in st_types:
        for st_case in st_cases:

            in_file_dir = os.path.join(in_dir_path, dataset.name, f"{st_type}_{st_case}")
            final_out_dir = os.path.join(out_dir_path, dataset.name, f"{st_type}_{st_case}")

            # Create folder if it does not exist
            os.makedirs(final_out_dir, exist_ok=True)

            # Using CORRELATION (mutual information can also be used)
            for analysis in analyses:

                in_data_file_path = os.path.join(in_file_dir, f"spearman_correlation_{analysis}.csv")
                in_root_file_path = os.path.join(in_file_dir, f"spearman_correlation_total_{analysis}.csv")

                df = pd.read_csv(in_data_file_path, index_col=0)

                tolerances = [0.05, 0.1, 0.2, 0.25]
                for tolerance in tolerances:
                    root_element = select_root_element(in_root_file_path)
                    tree = build_tree(df, root_element, tolerance)
                    generate_dot(tree, final_out_dir, tolerance)


def execute_non_ontouml_analysis(dataset, out_dir_path):
    st_types = ['class', 'relation']
    st_norms = ['yearly', 'overall']
    st_wises = ['ow', 'mw']

    years_start = [2015, 2017, 2019]

    for st_type in st_types:

        final_out_dir = os.path.join(out_dir_path, dataset.name, f"{st_type}_raw")

        # Create folder if it does not exist
        os.makedirs(final_out_dir, exist_ok=True)

        for st_norm in st_norms:
            df_occurrence = dataset.years_stereotypes_data[f'{st_type}_ow_{st_norm}']
            df_modelwise = dataset.years_stereotypes_data[f'{st_type}_mw_{st_norm}']

            # 1. Call generate_non_ontouml_combined_visualization without year_start (plot everything)
            # generate_non_ontouml_combined_visualization(df_occurrence, df_modelwise, final_out_dir,
            #                                             f'{st_type}_{st_norm}')
            generate_trend_visualization(df_occurrence, df_modelwise, final_out_dir, f'{st_type}_{st_norm}')

            # for year_start in years_start:  # 2. Call generate_non_ontouml_combined_visualization with year_start (plot from 'year_start' onwards)  # generate_non_ontouml_combined_visualization(df_occurrence, df_modelwise, final_out_dir,  #                                             f'{st_type}_{st_norm}_{year_start}', year_start=year_start)

            # for st_wise in st_wises:  #     analysis = f'{st_type}_{st_wise}_{st_norm}'  #  #     # 3. Call generate_non_ontouml_visualization without year_start (plot everything)  #     generate_non_ontouml_visualization(dataset.years_stereotypes_data[analysis], final_out_dir, analysis)  #  #     for year_start in years_start:  #         # 4. Call generate_non_ontouml_visualization with year_start (plot from 'year_start' onwards)  #         generate_non_ontouml_visualization(dataset.years_stereotypes_data[analysis], final_out_dir,  #                                            f"{analysis}_{year_start}", year_start=year_start)


def generate_visualizations(datasets, output_dir):
    if isinstance(datasets, str):
        # Deserialize (unpickle) the object
        logger.info(f"Loading datasets from {datasets}.")
        with gzip.open(datasets, "rb") as file:
            datasets = pickle.load(file)
        logger.success(f"Successfully loaded {len(datasets)} datasets.")

    coverages = [0.5, 0.75, 0.9, 0.95]
    for dataset in datasets:
        # plot_boxplot(dataset, OUTPUT_DIR_02, output_dir)
        # plot_boxplot(dataset, OUTPUT_DIR_02, output_dir, True)
        # plot_heatmap(dataset, output_dir)
        # plot_pareto(dataset, output_dir, "occurrence")
        # plot_pareto(dataset, output_dir, "group")
        # for coverage in coverages:
        #     plot_pareto_combined(dataset, output_dir, coverage)
        # plot_scatter(dataset, output_dir)
        # plot_learning_tree(dataset, OUTPUT_DIR_02, output_dir)
        execute_non_ontouml_analysis(dataset, output_dir)


def quadrants_calculation():
    dataset_types = ["", "_specific_filtered", "_general_filtered", "_double_specific_filtered",
                     "_double_general_filtered"]

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

    # Step 1: Input data - load all models' data, execute queries and create dataset
    all_models_data = load_data_from_catalog(catalog_path)
    query_data(all_models_data)
    all_models_data = calculate_models_data()
    datasets = create_and_save_specific_datasets_instances(all_models_data)

    # Step 2: Data processing - generate statistics
    calculate_and_save_datasets_statistics(datasets, OUTPUT_DIR_02)
    # calculate_and_save_datasets_stereotypes_statistics(datasets)
    # quadrants_calculation()
    # save_datasets(all_datasets, OUTPUT_DIR_02)
    #  generate_visualizations("outputs/02_datasets/datasets.object.gz", OUTPUT_DIR_03)
    # generate_visualizations(all_datasets, OUTPUT_DIR_03)
