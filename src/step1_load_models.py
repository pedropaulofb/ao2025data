import os

from src.directories_global import OUTPUT_DIR_01
from src.load_data import load_and_save_catalog_models, generate_list_models_data_csv, query_models


def load_data_from_catalog(catalog_path):
    """Load and save catalog models, and generate a CSV for model data."""

    all_models = load_and_save_catalog_models(catalog_path, OUTPUT_DIR_01)
    generate_list_models_data_csv(all_models, os.path.join(OUTPUT_DIR_01, "models_data.csv"))
    return all_models


def query_data(all_models):
    """Query class and relation stereotypes data for all models."""
    query_models(all_models, "queries", OUTPUT_DIR_01)