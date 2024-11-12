import csv
import os
import time
from typing import Union

from loguru import logger
from ontouml_models_lib import OntologyRepresentationStyle, OntologyDevelopmentContext, Model, Query, Catalog

from src.Dataset import Dataset
from src.ModelData import ModelData
from src.directories_global import OUTPUT_DIR_01
from src.utils import save_object, load_object


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

    save_object(datasets, OUTPUT_DIR_01, "datasets", "List of datasets instances")

    return datasets


def load_and_save_catalog_models(input_catalog_path: str, output_dir: str):
    """Load models from an OntoUML/UFO catalog and save them to an output directory."""
    # Load the catalog models
    catalog = Catalog(input_catalog_path)

    # Save the loaded models to the specified output directory
    save_object(catalog.models, output_dir, "loaded_models", "Loaded catalog models")

    # Return the loaded list of models
    return catalog.models


def generate_list_models_data_csv(input_models_list, output_file_path):
    input_models_list = load_object(input_models_list, "list of models")

    # Normalizing RELEASE DATE, creating IS_CLASSROOM attribute, and preparing data to be saved
    models_data = []
    for model in input_models_list:

        # IMPORTANT: Only OntoUML models. UFO-only models are discarded.
        if OntologyRepresentationStyle.ONTOUML_STYLE not in model.representationStyle:
            continue

        model.modified = model.issued if not model.modified else model.modified
        model.is_classroom = True if OntologyDevelopmentContext.CLASSROOM in model.context else False
        models_data.append((model.id, model.modified, model.is_classroom))

    # Generating CSV output
    header = ['model', 'year', 'is_classroom']

    # Open the CSV file in write mode
    with open(output_file_path, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(header)

        # Write the data (each tuple as a row)
        writer.writerows(models_data)

    logger.success(f"Models' data successfully saved in {output_file_path}.")


def query_models(models_to_query: Union[list[Model], str], queries_dir: str, output_dir: str):
    # Load models from file, if necessary

    models_to_query = load_object(models_to_query, "models to query")

    # Load and execute queries on the filtered models
    queries = Query.load_queries(queries_dir)

    for query in queries:
        start_time = time.perf_counter()
        query.execute_on_models(models_to_query, output_dir)
        end_time = time.perf_counter()
        elapsed_time_ms = (end_time - start_time) * 1000
        logger.info(f"Query {query.name} took {elapsed_time_ms:.2f} ms to perform.")
