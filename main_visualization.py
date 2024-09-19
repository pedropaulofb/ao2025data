from src.visualization.central_tendency_dispersion import execute_visualization_central_tendency_dispersion
from src.visualization.coverage_percentage import execute_visualization_coverage_percentage
from src.visualization.create_learning_line import execute_learning_line
from src.visualization.diversity_measures import execute_visualization_diversity_measures
from src.visualization.frequency_analysis_general import execute_visualization_frequency_analysis_general
from src.visualization.frequency_analysis_scatter import execute_visualization_frequency_analysis_scatter
from src.visualization.hybrid_analyses import execute_visualization_mutual_info_vs_jaccard_similarity, \
    execute_visualization_simpson_diversity_vs_construct_frequency, \
    execute_visualization_mutual_information_vs_construct_rank, \
    execute_visualization_shannon_entropy_vs_global_frequency, \
    execute_visualization_shannon_entropy_vs_group_frequency_constructs, \
    execute_visualization_ubiquity_vs_gini_coefficient, execute_visualization_gini_coefficient_vs_global_frequency, \
    execute_visualization_gini_coefficient_vs_group_frequency_constructs, \
    execute_visualization_mutual_info_vs_dice_coefficient
from src.visualization.mutual_information import execute_visualization_mutual_information
from src.visualization.rank_frequency_distribution import execute_visualization_rank_frequency_distribution
from src.visualization.similarity_measures import execute_visualization_similarity_measures
from src.visualization.spearman_correlation import execute_visualization_spearman_correlation


def create_visualizations():
    # GROUPED ANALYSES
    execute_visualization_central_tendency_dispersion('outputs/statistics/cs_analyses/central_tendency_dispersion.csv')
    execute_visualization_coverage_percentage('outputs/statistics/cs_analyses/coverage_percentage.csv')
    execute_visualization_diversity_measures('outputs/statistics/cs_analyses/diversity_measures.csv')
    execute_visualization_frequency_analysis_general("outputs/statistics/cs_analyses/frequency_analysis.csv")
    execute_visualization_frequency_analysis_scatter("outputs/statistics/cs_analyses/frequency_analysis.csv")
    execute_visualization_mutual_information('outputs/statistics/cs_analyses/mutual_information.csv')
    execute_visualization_rank_frequency_distribution("outputs/statistics/cs_analyses/rank_frequency_distribution.csv")
    execute_visualization_similarity_measures('outputs/statistics/cs_analyses/similarity_measures.csv')
    execute_visualization_spearman_correlation('outputs/statistics/cs_analyses/spearman_correlation.csv')
    execute_learning_line('outputs/statistics/cs_analyses/mutual_information.csv')
    execute_learning_line('outputs/statistics/cs_analyses/spearman_correlation.csv')
    #
    # # HYBRID ANALYSES
    execute_visualization_mutual_info_vs_jaccard_similarity('outputs/statistics/cs_analyses/mutual_information.csv',
                                                            'outputs/statistics/cs_analyses/similarity_measures.csv')
    execute_visualization_mutual_information_vs_construct_rank(
        'outputs/statistics/cs_analyses/rank_frequency_distribution.csv',
        'outputs/statistics/cs_analyses/mutual_information.csv')
    execute_visualization_shannon_entropy_vs_global_frequency('outputs/statistics/cs_analyses/diversity_measures.csv',
                                                              'outputs/statistics/cs_analyses/frequency_analysis.csv')
    execute_visualization_shannon_entropy_vs_group_frequency_constructs(
        'outputs/statistics/cs_analyses/diversity_measures.csv',
        'outputs/statistics/cs_analyses/frequency_analysis.csv')
    execute_visualization_simpson_diversity_vs_construct_frequency(
        'outputs/statistics/cs_analyses/diversity_measures.csv',
        'outputs/statistics/cs_analyses/frequency_analysis.csv')
    execute_visualization_ubiquity_vs_gini_coefficient('outputs/statistics/cs_analyses/diversity_measures.csv',
                                                       'outputs/statistics/cs_analyses/frequency_analysis.csv')
    execute_visualization_gini_coefficient_vs_global_frequency('outputs/statistics/cs_analyses/diversity_measures.csv',
                                                               'outputs/statistics/cs_analyses/frequency_analysis.csv')
    execute_visualization_gini_coefficient_vs_group_frequency_constructs(
        'outputs/statistics/cs_analyses/diversity_measures.csv',
        'outputs/statistics/cs_analyses/frequency_analysis.csv')
    execute_visualization_mutual_info_vs_dice_coefficient('outputs/statistics/cs_analyses/mutual_information.csv',
                                                          'outputs/statistics/cs_analyses/similarity_measures.csv')


if __name__ == "__main__":
    create_visualizations()
