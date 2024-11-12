import csv
import os

from src.Dataset import Dataset
from src.ModelData import ModelData
from src.directories_global import OUTPUT_DIR_01
from src.load_data import load_and_save_catalog_models, generate_list_models_data_csv, query_models
from src.utils import save_object


def load_data_from_catalog(catalog_path):
    """Load and save catalog models, and generate a CSV for model data."""

    all_models = load_and_save_catalog_models(catalog_path, OUTPUT_DIR_01)
    generate_list_models_data_csv(all_models, os.path.join(OUTPUT_DIR_01, "models_data.csv"))
    return all_models


def query_data(all_models):
    """Query class and relation stereotypes data for all models."""
    query_models(all_models, "queries", OUTPUT_DIR_01)


def instantiate_models_from_csv(input_models_data_csv_path: str, input_number_stereotypes_csv_path: str) -> list[
    ModelData]:
    models_list = []

    # Step 1: Read the from input_number_class_relation_csv to map model_id to count_class and count_relation
    class_relation_data = {}
    with open(input_number_stereotypes_csv_path, mode='r', newline='') as file2:
        reader2 = csv.DictReader(file2)
        for row in reader2:
            model_id = row["model_id"]
            count_class = int(row["count_class"])
            count_relation = int(row["count_relation"])
            class_relation_data[model_id] = {"count_class": count_class, "count_relation": count_relation}

    # Step 2: Read from input_models_data_csv and instantiate ModelData objects, merging the data from both files
    with open(input_models_data_csv_path, mode='r', newline='') as file1:
        reader1 = csv.DictReader(file1)
        for row in reader1:
            name = row["model"]
            year = int(row["year"])
            is_classroom = row["is_classroom"] == "True"  # Convert string to boolean

            # Get total_class and total_relation from the second file using the model_id
            total_class = class_relation_data[name]["count_class"]
            total_relation = class_relation_data[name]["count_relation"]

            # Instantiate ModelData with merged data
            model_data = ModelData(name, year, is_classroom, total_class, total_relation)
            models_list.append(model_data)

    return models_list


def calculate_models_data():
    """Load model data and count stereotypes for each model."""
    models_list = instantiate_models_from_csv(os.path.join(OUTPUT_DIR_01, "models_data.csv"),
                                              os.path.join(OUTPUT_DIR_01,
                                                           "query_count_number_classes_relations_consolidated.csv"))

    class_csv = os.path.join(OUTPUT_DIR_01, "query_get_all_class_stereotypes_consolidated.csv")
    relation_csv = os.path.join(OUTPUT_DIR_01, "query_get_all_relation_stereotypes_consolidated.csv")

    # Count stereotypes and calculate 'none' for each model
    for model in models_list:
        model.count_stereotypes("class", class_csv)
        model.count_stereotypes("relation", relation_csv)
        model.calculate_none()

    return models_list

def create_and_save_specific_datasets_instances(models_list):
    """Create datasets based on classroom and non-classroom models."""

    datasets = []

    ontouml_non_classroom = [model for model in models_list if not model.is_classroom]
    datasets.append(Dataset("ontouml_non_classroom", ontouml_non_classroom))

    ontouml_non_classroom_until_2018 = [model for model in ontouml_non_classroom if model.year <= 2018]
    datasets.append(Dataset("ontouml_non_classroom_until_2018", ontouml_non_classroom_until_2018))

    ontouml_non_classroom_after_2019 = [model for model in ontouml_non_classroom if model.year >= 2019]
    datasets.append(Dataset("ontouml_non_classroom_after_2019", ontouml_non_classroom_after_2019))

    save_object(datasets,OUTPUT_DIR_01, "datasets","List of datasets instances")

    return datasets