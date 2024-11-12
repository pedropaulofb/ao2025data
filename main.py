import os

import pandas as pd

from src.directories_global import OUTPUT_DIR_02, OUTPUT_DIR_01, OUTPUT_DIR_03
from src.quadrants_temporal import compare_and_generate_quadrant_csv
from src.step0_setup import initialize_output_directories, get_catalog_path
from src.step1_input import load_data_from_catalog, query_data, calculate_models_data, \
    create_and_save_specific_datasets_instances
from src.step2_processing import calculate_and_save_datasets_statistics, \
    calculate_and_save_datasets_stereotypes_statistics
from src.utils import load_object, save_object
from src.visualization.learning_tree import build_tree, generate_dot
from src.visualization.pareto import plot_pareto, plot_pareto_combined
from src.visualization.scatter import plot_scatter
from src.visualization.trend_analysis import generate_trend_visualization


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

    datasets = load_object(datasets,"datasets")

    coverages = [0.5, 0.75, 0.9, 0.95]
    for dataset in datasets:
        # plot_pareto(dataset, output_dir, "occurrence")
        # plot_pareto(dataset, output_dir, "group")
        for coverage in coverages:
            plot_pareto_combined(dataset, output_dir, coverage)
        plot_scatter(dataset, output_dir)
        execute_non_ontouml_analysis(dataset, output_dir)


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
    # initialize_output_directories()
    # catalog_path = get_catalog_path()

    # # Step 1: Data input - load all models' data, execute queries and create dataset
    # all_models_data = load_data_from_catalog(catalog_path)
    # query_data(all_models_data)
    # all_models_data = calculate_models_data()
    # datasets = create_and_save_specific_datasets_instances(all_models_data)

    # # Step 2: Data processing - generate statistics
    # calculate_and_save_datasets_statistics(datasets, OUTPUT_DIR_02)
    # calculate_and_save_datasets_stereotypes_statistics(datasets, OUTPUT_DIR_02)
    # calculate_and_save_datasets_quadrants()
    # save_object(datasets, OUTPUT_DIR_02, "datasets","Updated datasets")

    datasets = os.path.join(OUTPUT_DIR_02, "datasets.object.gz")

    # # Step 3: Data output - visualizations
    # generate_visualizations(datasets, OUTPUT_DIR_03)
