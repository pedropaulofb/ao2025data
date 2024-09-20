import os

from loguru import logger

from src.visualization.learning_line import execute_learning_line


def create_visualizations(input_dir: str):
    output_dir = input_dir.replace("statistics", "visualizations")

    # Create the directory to save if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.success(f"Created directory: {output_dir}")
    else:
        logger.info(f"Directory already exists: {output_dir}")

    # GROUPED ANALYSES
    # execute_visualization_central_tendency_dispersion(input_dir, output_dir, 'central_tendency_dispersion.csv')
    # execute_visualization_coverage_percentage_occurrence(input_dir, output_dir, 'coverage_percentage_occurrence.csv')
    # execute_visualization_coverage_percentage_group(input_dir, output_dir, 'coverage_percentage_group.csv')
    # execute_visualization_diversity_measures(input_dir, output_dir, 'diversity_measures.csv')
    # execute_visualization_frequency_analysis_general(input_dir, output_dir, "frequency_analysis.csv")
    # execute_visualization_frequency_analysis_scatter(input_dir, output_dir, "frequency_analysis.csv")
    # execute_visualization_mutual_information(input_dir, output_dir, 'mutual_information.csv')
    # execute_visualization_rank_frequency_distribution_occurrence(input_dir, output_dir, "rank_frequency_distribution.csv")
    # execute_visualization_rank_frequency_distribution_group(input_dir, output_dir, "rank_groupwise_frequency_distribution.csv")
    # execute_visualization_similarity_measures(input_dir, output_dir, 'similarity_measures.csv')
    # execute_visualization_spearman_correlation(input_dir, output_dir, 'spearman_correlation.csv')
    execute_learning_line(input_dir, output_dir, 'mutual_information.csv')
    execute_learning_line(input_dir, output_dir, 'spearman_correlation.csv')

    # HYBRID ANALYSES  # execute_visualization_mutual_info_vs_jaccard_similarity(input_dir, output_dir, 'mutual_information.csv',  #                                                         'similarity_measures.csv')  # execute_visualization_mutual_information_vs_construct_rank(input_dir, output_dir, 'rank_frequency_distribution.csv',  #                                                            'mutual_information.csv')  # execute_visualization_shannon_entropy_vs_global_frequency(input_dir, output_dir, 'diversity_measures.csv',  #                                                           'frequency_analysis.csv')  # execute_visualization_shannon_entropy_vs_group_frequency_constructs(input_dir, output_dir, 'diversity_measures.csv',  #                                                                     'frequency_analysis.csv')  # execute_visualization_simpson_diversity_vs_construct_frequency(input_dir, output_dir, 'diversity_measures.csv',  #                                                                'frequency_analysis.csv')  # execute_visualization_ubiquity_vs_gini_coefficient(input_dir, output_dir, 'diversity_measures.csv',  #                                                    'frequency_analysis.csv')  # execute_visualization_gini_coefficient_vs_global_frequency(input_dir, output_dir, 'diversity_measures.csv',  #                                                            'frequency_analysis.csv')  # execute_visualization_gini_coefficient_vs_group_frequency_constructs(input_dir, output_dir,  #                                                                      'diversity_measures.csv',  #                                                                      'frequency_analysis.csv')  # execute_visualization_mutual_info_vs_dice_coefficient(input_dir, output_dir, 'mutual_information.csv',  #                                                       'similarity_measures.csv')  # execute_visualization_coverage_percentage_all(input_dir, output_dir, 'coverage_percentage_occurrence.csv',  #                                                       'coverage_percentage_group.csv')  # execute_visualization_pareto_combined(input_dir, output_dir, 'rank_frequency_distribution.csv','rank_groupwise_frequency_distribution.csv')


def get_subdirectories(directory):
    # List all first-level subdirectories
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return subdirs


if __name__ == "__main__":
    # Statistics directory path
    statistics_dir = os.path.normpath('./outputs/statistics')  # Normalize the directory path
    # Get the list of first-level subdirectories
    subdirectories = get_subdirectories(statistics_dir)

    analysis_paths = []
    for analysis_path in subdirectories:
        analysis_paths.append(os.path.normpath(os.path.join(statistics_dir, analysis_path)))

    for analysis_path in analysis_paths:
        logger.info(f"Generating visualizations to {analysis_path}.")
        create_visualizations(analysis_path)

    # TIME COMPARISON CLASS  # path_file_A = './outputs/statistics/cs_ontouml_no_classroom_until_2017_f/frequency_analysis.csv'  # path_file_B = './outputs/statistics/cs_ontouml_no_classroom_after_2018_f/frequency_analysis.csv'  # out_dir_path = './outputs/visualizations/movement/cs_ontouml_no_classroom_f'  # execute_visualization_movement(path_file_A, path_file_B, out_dir_path)  #  # path_file_A = './outputs/statistics/cs_ontouml_no_classroom_until_2017_t/frequency_analysis.csv'  # path_file_B = './outputs/statistics/cs_ontouml_no_classroom_after_2018_t/frequency_analysis.csv'  # out_dir_path = './outputs/visualizations/movement/cs_ontouml_no_classroom_t'  # execute_visualization_movement(path_file_A, path_file_B, out_dir_path)  #  # # TIME COMPARISON RELATION  # path_file_A = './outputs/statistics/rs_ontouml_no_classroom_until_2017_f/frequency_analysis.csv'  # path_file_B = './outputs/statistics/rs_ontouml_no_classroom_after_2018_f/frequency_analysis.csv'  # out_dir_path = './outputs/visualizations/movement/rs_ontouml_no_classroom_f'  # execute_visualization_movement(path_file_A, path_file_B, out_dir_path)  #  # path_file_A = './outputs/statistics/rs_ontouml_no_classroom_until_2017_t/frequency_analysis.csv'  # path_file_B = './outputs/statistics/rs_ontouml_no_classroom_after_2018_t/frequency_analysis.csv'  # out_dir_path = './outputs/visualizations/movement/rs_ontouml_no_classroom_t'  # execute_visualization_movement(path_file_A, path_file_B, out_dir_path)
