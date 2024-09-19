import os
import pickle
import time

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
        with open(models_to_filter, "rb") as file:
            models_to_filter = pickle.load(file)

    # Custom filtering to remove models without ONTOUML_STYLE in representationStyle
    filtered = [model for model in models_to_filter if
        OntologyRepresentationStyle.ONTOUML_STYLE in model.representationStyle]

    # Custom filtering to remove models with just CLASSROOM as context
    filtered = [model for model in filtered if ((OntologyDevelopmentContext.RESEARCH in model.context) or (
            OntologyDevelopmentContext.INDUSTRY in model.context))]

    # Normalizing modified date
    for model in filtered:
        model.modified = model.issued if not model.modified else model.modified

    # Custom filtering to remove models modified from 2018 onwards
    filtered = [model for model in filtered if (model.issued <= 2017) or (model.modified <= 2017)]

    # Save filtered models to a file
    output_name = os.path.join("./outputs/", output_file_name + ".object")
    with open(output_name, "wb") as file:
        pickle.dump(filtered, file)
    logger.success(f"Filtered models successfully saved in {output_name}.")

    return filtered


def query_filtered_models(models_to_query):
    # Load and execute queries on the filtered models
    queries = Query.load_queries("queries")

    for query in queries:
        start_time = time.perf_counter()
        query.execute_on_models(models_to_query, "./outputs/queries_results")
        end_time = time.perf_counter()
        elapsed_time_ms = (end_time - start_time) * 1000
        logger.info(f"Query {query.name} took {elapsed_time_ms:.2f} ms to perform.")


if __name__ == "__main__":
    loaded_models = load_all_models(
        "all_models")  # filtered_models = filter_models(loaded_models, "models_ontouml_no-classroom_until_2017")  # query_filtered_models(filtered_models)
