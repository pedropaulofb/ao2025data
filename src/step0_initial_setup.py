import argparse
import os

from loguru import logger

from src.directories_global import BASE_OUTPUT_DIR, OUTPUT_DIR_01, OUTPUT_DIR_02, OUTPUT_DIR_03, CATALOG_PATH


def initialize_output_directories():
    """Create output directories with exception handling."""
    directories = [BASE_OUTPUT_DIR, OUTPUT_DIR_01, OUTPUT_DIR_02, OUTPUT_DIR_03]

    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            # logger.success(f"Successfully created or verified existence of directory: {directory}")
        except OSError as e:
            logger.error(f"Failed to create directory {directory}. Error: {e}")

def create_parser():
    """Creates and returns an argument parser for the catalog path."""
    parser = argparse.ArgumentParser(description="Receives OntoUML models path.")
    parser.add_argument("catalog_path", nargs="?", default=CATALOG_PATH,
        help=f"Path to the input data source directory. Defaults to '{CATALOG_PATH}' if not provided.")
    return parser


def get_catalog_path():
    """Parses the catalog path from command-line arguments and logs the result."""
    parser = create_parser()
    args = parser.parse_args()

    catalog_path = args.catalog_path
    # logger.info(f"Using catalog path: {catalog_path}")

    return catalog_path