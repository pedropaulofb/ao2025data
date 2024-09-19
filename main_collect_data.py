import os
import pickle
import time

from icecream import ic
from loguru import logger
from ontouml_models_lib import Catalog, Query, OntologyDevelopmentContext
from ontouml_models_lib.model import OntologyRepresentationStyle


def load_all_models(output_file_name):
    # Initialize the catalog
    catalog = Catalog("C:/Users/FavatoBarcelosPP/Dev/ontouml-models")

    # Save loaded models to a file
    output_name = os.path.join("./outputs/", output_file_name + ".object")
    with open(output_name, "wb") as file:
        pickle.dump(catalog.models, file)
    logger.success(f"Loaded models successfully saved in {output_name}.")

    return catalog.models


def filter_models(models_to_filter, output_file_name) -> None:
    if isinstance(models_to_filter, str):
        # Deserialize (unpickle) the object
        logger.info(f"Loading models from {models_to_filter}.")
        with open(models_to_filter, "rb") as file:
            models_to_filter = pickle.load(file)
        logger.success(f"Successfully loaded {len(models_to_filter)} models.")

    # Custom filtering to remove models without ONTOUML_STYLE in representationStyle
    filtered = [model for model in models_to_filter if
        OntologyRepresentationStyle.ONTOUML_STYLE in model.representationStyle]

    # Custom filtering to remove models with just CLASSROOM as context
    filtered = [model for model in filtered if ((OntologyDevelopmentContext.RESEARCH in model.context) or (
            OntologyDevelopmentContext.INDUSTRY in model.context))]

    # # Normalizing modified date
    # for model in filtered:
    #     model.modified = model.issued if not model.modified else model.modified
    #
    # # Custom filtering to remove models modified from 2018 onwards
    # filtered = [model for model in filtered if (model.issued > 2017) or (model.modified > 2017)]

    # Save filtered models to a file
    output_name = os.path.join("./outputs/", output_file_name + ".object")
    with open(output_name, "wb") as file:
        pickle.dump(filtered, file)
    logger.success(f"{len(filtered)} filtered models successfully saved in {output_name}.")

    output_list = os.path.join("./outputs/", output_file_name + ".txt")
    generate_list_models(filtered, output_list)

    return filtered


def query_filtered_models(models_to_query, output_dir_path):

    if isinstance(models_to_query, str):
        # Deserialize (unpickle) the object
        logger.info(f"Loading models from {models_to_query}.")
        with open(models_to_query, "rb") as file:
            models_to_query = pickle.load(file)
        logger.success(f"Successfully loaded {len(models_to_query)} models.")

    # Load and execute queries on the filtered models
    queries = Query.load_queries("queries")

    for query in queries:
        start_time = time.perf_counter()
        query.execute_on_models(models_to_query, output_dir_path)
        end_time = time.perf_counter()
        elapsed_time_ms = (end_time - start_time) * 1000
        logger.info(f"Query {query.name} took {elapsed_time_ms:.2f} ms to perform.")

def generate_list_models(models_to_list,output_file_path):

    if isinstance(models_to_list, str):
        # Deserialize (unpickle) the object
        logger.info(f"Loading models from {models_to_list}.")
        with open(models_to_list, "rb") as file:
            loaded_models = pickle.load(file)
        logger.success(f"Successfully loaded {len(loaded_models)} models.")

    list_models_id = [models.id for models in loaded_models]

    # Save the list of strings to a text file
    with open(output_file_path, "w") as file:
        for item in list_models_id:
            file.write(item + "\n")
    logger.success(f"Successfully written {len(list_models_id)} models' ids in {output_file_path}.")


if __name__ == "__main__":
    # loaded_models = load_all_models("all_models")
    # filter_models("./outputs/all_models.object", "models_ontouml_no-classroom")
    # query_filtered_models("./outputs/ontouml_no-classroom.object", "outputs/queries_results/ontouml_no-classroom")
    # query_filtered_models("./outputs/ontouml_no-classroom_after_2018.object",
    #                       "outputs/queries_results/ontouml_no-classroom_after_2018")
    # query_filtered_models("./outputs/ontouml_no-classroom_until_2017.object",
    #                       "outputs/queries_results/ontouml_no-classroom_until_2017")

    # generate_list_models("./outputs/loaded_models/ontouml_no-classroom_until_2017.object",
    #                      "outputs/loaded_models/ontouml_no-classroom.object_until_2017.txt")
    pass