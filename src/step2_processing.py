from src.utils import load_object


def calculate_and_save_datasets_statistics(datasets, output_dir):
    datasets = load_object(datasets, "datasets")

    for dataset in datasets:
        save_dataset_info(dataset, output_dir)

        dataset.calculate_dataset_statistics()
        dataset.calculate_models_statistics()
        dataset.save_models_statistics_to_csv(output_dir)
        dataset.calculate_and_save_stereotypes_by_year(output_dir)
        dataset.calculate_and_save_models_by_year(output_dir)
        dataset.save_stereotypes_count_by_year(output_dir)


def save_dataset_info(dataset, output_dir):
    dataset.save_dataset_general_data_csv(output_dir)
    dataset.save_dataset_class_data_csv(output_dir)
    dataset.save_dataset_relation_data_csv(output_dir)


def calculate_and_save_datasets_stereotypes_statistics(datasets, output_dir):
    datasets = load_object(datasets, "datasets")

    for dataset in datasets:
        dataset.calculate_stereotype_statistics()
        dataset.save_stereotype_statistics(output_dir)
        dataset.calculate_invalid_stereotypes_metrics()
        dataset.save_invalid_stereotypes_metrics_to_csv(output_dir)
        dataset.calculate_and_save_average_model(output_dir)
        dataset.classify_and_save_spearman_correlation(output_dir)
        dataset.classify_and_save_total_correlation(output_dir)
        dataset.classify_and_save_geometric_mean_correlation(output_dir)
        dataset.classify_and_save_geometric_mean_pairwise_correlation(output_dir)
        dataset.calculate_and_save_quadrants(output_dir, 'frequency_analysis',
                                             'Global Relative Frequency (Occurrence-wise)', 'Ubiquity Index')

        dataset.general_validation()
