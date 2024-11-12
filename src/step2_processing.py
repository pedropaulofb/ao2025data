from src.directories_global import OUTPUT_DIR_02


def calculate_and_save_datasets_statistics(datasets, output_dir):
    for dataset in datasets:
        save_dataset_info(dataset)

        dataset.calculate_dataset_statistics()
        dataset.calculate_models_statistics()
        dataset.save_models_statistics_to_csv(output_dir)
        dataset.calculate_and_save_stereotypes_by_year(output_dir)
        dataset.calculate_and_save_models_by_year(output_dir)
        dataset.save_stereotypes_count_by_year(output_dir)

def save_dataset_info(dataset):
    dataset.save_dataset_general_data_csv(OUTPUT_DIR_02)
    dataset.save_dataset_class_data_csv(OUTPUT_DIR_02)
    dataset.save_dataset_relation_data_csv(OUTPUT_DIR_02)

def calculate_and_save_datasets_stereotypes_statistics(datasets):
    for dataset in datasets:
        dataset.calculate_stereotype_statistics()
        dataset.save_stereotype_statistics(OUTPUT_DIR_02)
        dataset.calculate_and_save_average_model(OUTPUT_DIR_02)
        dataset.classify_and_save_spearman_correlation(OUTPUT_DIR_02)
        dataset.classify_and_save_total_correlation(OUTPUT_DIR_02)
        dataset.classify_and_save_geometric_mean_correlation(OUTPUT_DIR_02)
        dataset.classify_and_save_geometric_mean_pairwise_correlation(OUTPUT_DIR_02)
        dataset.calculate_and_save_quadrants(OUTPUT_DIR_02, 'frequency_analysis',
                                             'Global Relative Frequency (Occurrence-wise)', 'Ubiquity Index')