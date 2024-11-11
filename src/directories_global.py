import os

# Output directories
BASE_OUTPUT_DIR = "./outputs"
OUTPUT_DIR_01 = os.path.join(BASE_OUTPUT_DIR, "01_loaded_models_data")
OUTPUT_DIR_02 = os.path.join(BASE_OUTPUT_DIR, "02_datasets_statistics")
OUTPUT_DIR_03 = os.path.join(BASE_OUTPUT_DIR, "03_visualizations")

# Default OntoUML/UFO Catalog path (can be overridden if user provides as argument)
CATALOG_PATH = "../ontouml-models"
