import time

from loguru import logger
from ontouml_models_lib import Catalog, Query, OntologyDevelopmentContext
from ontouml_models_lib.model import OntologyRepresentationStyle

from src.generate_stats import simple_write_to_csv


def main() -> None:
    """
    Main function to collect data from all catalog models
    """
    # Initialize the catalog
    catalog = Catalog("C:/Users/FavatoBarcelosPP/Dev/ontouml-models")

    # Custom filtering to remove models without ONTOUML_STYLE in representationStyle
    filtered = [
        model for model in catalog.models
        if OntologyRepresentationStyle.ONTOUML_STYLE in model.representationStyle
    ]

    # Custom filtering to remove models with just CLASSROOM as context
    filtered = [
        model for model in filtered
        if ((OntologyDevelopmentContext.RESEARCH in model.context) or (
                OntologyDevelopmentContext.INDUSTRY in model.context))
    ]

    # Saving to file all filetered models
    filtered_models = []
    for model in filtered:
        filtered_models.append(model.id)
    simple_write_to_csv(filtered_models, "./outputs/list_models.txt")

    # Load and execute queries on the filtered models
    queries = Query.load_queries("queries")

    for query in queries:
        start_time = time.perf_counter()
        query.execute_on_models(filtered, "./outputs/queries_results")
        end_time = time.perf_counter()
        elapsed_time_ms = (end_time - start_time) * 1000
        logger.info(f"Query {query.name} took {elapsed_time_ms:.2f} ms to perform.")


if __name__ == "__main__":
    main()
