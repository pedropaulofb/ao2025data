from visualization.central_tendency_dispersion import execute_visualization_central_tendency_dispersion
from visualization.coverage_percentage import execute_visualization_coverage_percentage
from visualization.diversity_measures import execute_visualization_diversity_measures
from visualization.frequency_analysis import execute_visualization_frequency_analysis
from visualization.hybrid_analyses import execute_visualization_mutual_info_vs_jaccard_similarity, \
    execute_visualization_simpson_diversity_vs_construct_frequency, \
    execute_visualization_mutual_information_vs_construct_rank, \
    execute_visualization_shannon_entropy_vs_global_frequency, \
    execute_visualization_shannon_entropy_vs_group_frequency_constructs, \
    execute_visualization_ubiquity_vs_gini_coefficient, execute_visualization_gini_coefficient_vs_global_frequency, \
    execute_visualization_gini_coefficient_vs_group_frequency_constructs, \
    execute_visualization_mutual_info_vs_dice_coefficient
from visualization.mutual_information import execute_visualization_mutual_information
from visualization.rank_frequency_distribution import execute_visualization_rank_frequency_distribution
from visualization.similarity_measures import execute_visualization_similarity_measures
from visualization.spearman_correlation import execute_visualization_spearman_correlation


def create_visualizations():
    # GROUPED ANALYSES

    execute_visualization_central_tendency_dispersion('./outputs/analyses/cs_analyses/central_tendency_dispersion.csv')
    execute_visualization_coverage_percentage('./outputs/analyses/cs_analyses/coverage_percentage.csv')
    execute_visualization_diversity_measures('./outputs/analyses/cs_analyses/diversity_measures.csv')
    execute_visualization_frequency_analysis("./outputs/analyses/cs_analyses/frequency_analysis.csv")
    execute_visualization_mutual_information('./outputs/analyses/cs_analyses/mutual_information.csv')
    execute_visualization_rank_frequency_distribution("./outputs/analyses/cs_analyses/rank_frequency_distribution.csv")
    execute_visualization_similarity_measures('./outputs/analyses/cs_analyses/similarity_measures.csv')
    execute_visualization_spearman_correlation('./outputs/analyses/cs_analyses/spearman_correlation.csv')

    # HYBRID ANALYSES

    execute_visualization_mutual_info_vs_jaccard_similarity('./outputs/analyses/cs_analyses/mutual_information.csv',
                                                            './outputs/analyses/cs_analyses/similarity_measures.csv')
    execute_visualization_mutual_information_vs_construct_rank(
        './outputs/analyses/cs_analyses/rank_frequency_distribution.csv',
        './outputs/analyses/cs_analyses/mutual_information.csv')
    execute_visualization_shannon_entropy_vs_global_frequency('./outputs/analyses/cs_analyses/diversity_measures.csv',
                                                              './outputs/analyses/cs_analyses/frequency_analysis.csv')
    execute_visualization_shannon_entropy_vs_group_frequency_constructs(
        './outputs/analyses/cs_analyses/diversity_measures.csv',
        './outputs/analyses/cs_analyses/frequency_analysis.csv')
    execute_visualization_simpson_diversity_vs_construct_frequency(
        './outputs/analyses/cs_analyses/diversity_measures.csv',
        './outputs/analyses/cs_analyses/frequency_analysis.csv')
    execute_visualization_ubiquity_vs_gini_coefficient('./outputs/analyses/cs_analyses/diversity_measures.csv',
                                                       './outputs/analyses/cs_analyses/frequency_analysis.csv')
    execute_visualization_gini_coefficient_vs_global_frequency('./outputs/analyses/cs_analyses/diversity_measures.csv',
                                                               './outputs/analyses/cs_analyses/frequency_analysis.csv')
    execute_visualization_gini_coefficient_vs_group_frequency_constructs(
        './outputs/analyses/cs_analyses/diversity_measures.csv',
        './outputs/analyses/cs_analyses/frequency_analysis.csv')
    execute_visualization_mutual_info_vs_dice_coefficient('./outputs/analyses/cs_analyses/mutual_information.csv',
                                                          './outputs/analyses/cs_analyses/similarity_measures.csv')


if __name__ == "__main__":
    create_visualizations()
