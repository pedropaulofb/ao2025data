import os

from src.Dataset import Dataset
from src.collect_data import load_and_save_catalog_models, generate_list_models_data_csv, query_models
from src.load_models_data import instantiate_models_from_csv
from src.save_datasets_stats_to_csv import save_datasets_statistics_to_csv

CATALOG_PATH = "C:/Users/FavatoBarcelosPP/Dev/ontouml-models"
OUTPUT_DIR = "./outputs"


def load_data_from_catalog(catalog_path):
    """Load and save catalog models, and generate a CSV for model data."""
    all_models = load_and_save_catalog_models(catalog_path, os.path.join(OUTPUT_DIR, "01_loaded_data/"))
    generate_list_models_data_csv(all_models, os.path.join(OUTPUT_DIR, "01_loaded_data", "models_data.csv"))
    return all_models


def query_data(all_models):
    """Query class and relation stereotypes data for all models."""
    query_models(all_models, "queries", os.path.join(OUTPUT_DIR, "01_loaded_data/"))


def create_datasets_instances(models_list):
    """Create datasets based on classroom and non-classroom models."""

    datasets = []
    datasets.append(Dataset("all_models", models_list))

    classroom_models = [model for model in models_list if model.is_classroom]
    datasets.append(Dataset("classroom_models", classroom_models))

    non_classroom_models = [model for model in models_list if not model.is_classroom]
    datasets.append(Dataset("non_classroom_models", non_classroom_models))

    non_classroom_models_until_2018 = [model for model in non_classroom_models if model.year <= 2018]
    datasets.append(Dataset("non_classroom_models_until_2018", non_classroom_models_until_2018))

    non_classroom_models_after_2019 = [model for model in non_classroom_models if model.year >= 2019]
    datasets.append(Dataset("non_classroom_models_after_2019", non_classroom_models_after_2019))

    return datasets


def create_datasets(datasets):
    datasets = create_datasets_instances(datasets)

    for dataset in datasets:
        save_dataset_info(dataset)
        dataset.calculate_statistics()

    save_datasets_statistics_to_csv(datasets, os.path.join(OUTPUT_DIR, "02_datasets"))

    return datasets


def load_models_data():
    """Load model data and count stereotypes for each model."""
    models_list = instantiate_models_from_csv(os.path.join(OUTPUT_DIR, "01_loaded_data/models_data.csv"),
        os.path.join(OUTPUT_DIR, "01_loaded_data/query_count_number_classes_relations_consolidated.csv"))

    class_csv = os.path.join(OUTPUT_DIR, "01_loaded_data/query_get_all_class_stereotypes_consolidated.csv")
    relation_csv = os.path.join(OUTPUT_DIR, "01_loaded_data/query_get_all_relation_stereotypes_consolidated.csv")

    # Count stereotypes and calculate 'none' for each model
    for model in models_list:
        model.count_stereotypes("class", class_csv)
        model.count_stereotypes("relation", relation_csv)
        model.calculate_none()

    return models_list


def save_dataset_info(dataset):
    dataset.generate_dataset_general_data_csv(os.path.join(OUTPUT_DIR, "02_datasets"))
    dataset.generate_dataset_class_data_csv(os.path.join(OUTPUT_DIR, "02_datasets"))
    dataset.generate_dataset_relation_data_csv(os.path.join(OUTPUT_DIR, "02_datasets"))

if __name__ == "__main__":
    # all_models = load_data_from_catalog(CATALOG_PATH)   # Uncomment to load models
    # query_data(all_models)     # Uncomment to query stereotypes

    all_models = load_models_data()  # Load model data and count stereotypes
    datasets = create_datasets(all_models)
