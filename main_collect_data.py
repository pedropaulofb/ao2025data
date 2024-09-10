from ontouml_models_lib import Catalog, Query, OntologyDevelopmentContext
from ontouml_models_lib.model import OntologyRepresentationStyle

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
        if ((OntologyDevelopmentContext.RESEARCH in model.context) or (OntologyDevelopmentContext.INDUSTRY in model.context))
    ]

    # Load and execute queries on the filtered models
    queries = Query.load_queries("queries")

    for query in queries:
        query.execute_on_models(filtered, "./outputs/queries_results")

    query = Query("./queries/query_s.sparql")
    query.execute_on_models(filtered)


if __name__ == "__main__":
    main()