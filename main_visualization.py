import os

from loguru import logger

from src.visualization.central_tendency_dispersion import execute_visualization_central_tendency_dispersion
from src.visualization.coverage_percentage import execute_visualization_coverage_percentage
from src.visualization.create_learning_line import execute_learning_line
from src.visualization.diversity_measures import execute_visualization_diversity_measures
from src.visualization.frequency_analysis_general import execute_visualization_frequency_analysis_general
from src.visualization.frequency_analysis_scatter import execute_visualization_frequency_analysis_scatter
from src.visualization.hybrid_analyses import execute_visualization_mutual_info_vs_jaccard_similarity, \
    execute_visualization_mutual_information_vs_construct_rank, \
    execute_visualization_shannon_entropy_vs_global_frequency, \
    execute_visualization_shannon_entropy_vs_group_frequency_constructs, \
    execute_visualization_simpson_diversity_vs_construct_frequency, execute_visualization_ubiquity_vs_gini_coefficient, \
    execute_visualization_gini_coefficient_vs_global_frequency, \
    execute_visualization_gini_coefficient_vs_group_frequency_constructs, \
    execute_visualization_mutual_info_vs_dice_coefficient
from src.visualization.mutual_information import execute_visualization_mutual_information
from src.visualization.rank_frequency_distribution import execute_visualization_rank_frequency_distribution
from src.visualization.similarity_measures import execute_visualization_similarity_measures
from src.visualization.spearman_correlation import execute_visualization_spearman_correlation


def create_visualizations(input_dir: str):
    output_dir = input_dir.replace("statistics", "visualizations")

    # Create the directory to save if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.success(f"Created directory: {output_dir}")
    else:
        logger.info(f"Directory already exists: {output_dir}")

    # GROUPED ANALYSES
    execute_visualization_central_tendency_dispersion(input_dir, output_dir, 'central_tendency_dispersion.csv')
    execute_visualization_coverage_percentage(input_dir, output_dir, 'coverage_percentage.csv')
    execute_visualization_diversity_measures(input_dir, output_dir, 'diversity_measures.csv')
    execute_visualization_frequency_analysis_general(input_dir, output_dir, "frequency_analysis.csv")
    execute_visualization_frequency_analysis_scatter(input_dir, output_dir, "frequency_analysis.csv")
    execute_visualization_mutual_information(input_dir, output_dir, 'mutual_information.csv')
    execute_visualization_rank_frequency_distribution(input_dir, output_dir, "rank_frequency_distribution.csv")
    execute_visualization_similarity_measures(input_dir, output_dir, 'similarity_measures.csv')
    execute_visualization_spearman_correlation(input_dir, output_dir, 'spearman_correlation.csv')
    execute_learning_line(input_dir, output_dir, 'mutual_information.csv')
    execute_learning_line(input_dir, output_dir, 'spearman_correlation.csv')

    # HYBRID ANALYSES
    execute_visualization_mutual_info_vs_jaccard_similarity(input_dir, output_dir, 'mutual_information.csv',
                                                            'similarity_measures.csv')
    execute_visualization_mutual_information_vs_construct_rank(input_dir, output_dir, 'rank_frequency_distribution.csv',
                                                               'mutual_information.csv')
    execute_visualization_shannon_entropy_vs_global_frequency(input_dir, output_dir, 'diversity_measures.csv',
                                                              'frequency_analysis.csv')
    execute_visualization_shannon_entropy_vs_group_frequency_constructs(input_dir, output_dir, 'diversity_measures.csv',
                                                                        'frequency_analysis.csv')
    execute_visualization_simpson_diversity_vs_construct_frequency(input_dir, output_dir, 'diversity_measures.csv',
                                                                   'frequency_analysis.csv')
    execute_visualization_ubiquity_vs_gini_coefficient(input_dir, output_dir, 'diversity_measures.csv',
                                                       'frequency_analysis.csv')
    execute_visualization_gini_coefficient_vs_global_frequency(input_dir, output_dir, 'diversity_measures.csv',
                                                               'frequency_analysis.csv')
    execute_visualization_gini_coefficient_vs_group_frequency_constructs(input_dir, output_dir,
                                                                         'diversity_measures.csv',
                                                                         'frequency_analysis.csv')
    execute_visualization_mutual_info_vs_dice_coefficient(input_dir, output_dir, 'mutual_information.csv',
                                                          'similarity_measures.csv')


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
