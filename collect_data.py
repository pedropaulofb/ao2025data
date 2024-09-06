from ontouml_models_lib import *

catalog = Catalog("C:/Users/FavatoBarcelosPP/Dev/ontouml-models", limit_num_models=10)

queries = Query.load_queries("./queries")

for query in queries:
    query.execute_on_models(catalog.models)
