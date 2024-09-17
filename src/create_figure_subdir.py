import os

from loguru import logger


def create_figures_subdir(file_path):
    """
    Create a 'figures' subdirectory in the directory of the given file path if it doesn't already exist.

    Args:
        file_path (str): The path to a file whose directory will be used.
    """
    # Get the directory of the given file path
    directory = os.path.dirname(file_path)

    # Define the path for the 'figures' subdirectory
    figures_dir = os.path.join(directory, 'figures')

    # Normalize the path to ensure consistent formatting
    figures_dir = os.path.normpath(figures_dir)

    # Create the 'figures' subdirectory if it does not exist
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
        logger.success(f"Created directory: {figures_dir}")
    else:
        logger.info(f"Directory already exists: {figures_dir}")

    return figures_dir
