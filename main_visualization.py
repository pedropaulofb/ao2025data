from visualization.hybrid_analyses import execute_visualization_mutual_info_vs_jaccard_similarity, \
    execute_visualization_simpson_diversity_vs_construct_frequency, \
    execute_visualization_mutual_information_vs_construct_rank, \
    execute_visualization_shannon_entropy_vs_global_frequency, \
    execute_visualization_shannon_entropy_vs_group_frequency_constructs, \
    execute_visualization_ubiquity_vs_gini_coefficient


def create_visualizations():

    execute_visualization_mutual_info_vs_jaccard_similarity('./outputs/analyses/cs_analyses/mutual_information.csv',
                             './outputs/analyses/cs_analyses/similarity_measures.csv')

    execute_visualization_simpson_diversity_vs_construct_frequency('./outputs/analyses/cs_analyses/diversity_measures.csv',
                             './outputs/analyses/cs_analyses/frequency_analysis.csv')

    execute_visualization_mutual_information_vs_construct_rank('./outputs/analyses/cs_analyses/rank_frequency_distribution.csv',
                             './outputs/analyses/cs_analyses/mutual_information.csv')

    execute_visualization_shannon_entropy_vs_global_frequency('./outputs/analyses/cs_analyses/diversity_measures.csv',
                                                              './outputs/analyses/cs_analyses/frequency_analysis.csv')

    execute_visualization_shannon_entropy_vs_group_frequency_constructs(
        './outputs/analyses/cs_analyses/diversity_measures.csv',
        './outputs/analyses/cs_analyses/frequency_analysis.csv')

    execute_visualization_ubiquity_vs_gini_coefficient('./outputs/analyses/cs_analyses/diversity_measures.csv',
                                                       './outputs/analyses/cs_analyses/frequency_analysis.csv')


if __name__ == "__main__":
    create_visualizations()