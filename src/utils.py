# Utility function to save to CSV
import gzip
import os
import pickle
import re

import pandas as pd
from loguru import logger


def color_text(texts):
    """Function to color specific texts for legends or axis labels."""
    for text in texts:
        if 'none' in text.get_text():
            text.set_color('#91138d')
        elif 'other' in text.get_text():
            text.set_color('red')


def bold_left_labels(texts, red_line_index):
    """
    Function to apply bold font to x-axis labels that are on or to the left of the red line.

    :param texts: List of x-axis label texts.
    :param red_line_index: The index where the red line is drawn.
    """
    for i, text in enumerate(texts):
        if i <= red_line_index:
            text.set_fontweight('bold')


def format_metric_name(metric):
    # Convert to lowercase
    formatted_metric = metric.lower().strip()
    # Replace any sequence of one or more spaces or underscores with a single underscore
    formatted_metric = re.sub(r'[\s_]+', '_', formatted_metric)
    # Remove text inside parentheses (including parentheses)
    formatted_metric = re.sub(r'\s*\(.*?\)', '', formatted_metric)
    # Remove any leading or trailing underscores
    formatted_metric = formatted_metric.strip('_')
    return formatted_metric


def save_to_csv(data, filepath, message):
    """
    Utility function to save a DataFrame to a CSV file with error handling.
    """
    try:
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        logger.success(message)
    except Exception as e:
        logger.error(f"Failed to save {filepath}: {e}")


def append_unique_preserving_order(existing_list, new_keys):
    """Append keys to the list, preserving the original order and ensuring no duplicates."""
    for key in new_keys:
        if key not in existing_list:
            existing_list.append(key)
    return existing_list


def save_datasets(datasets, output_dir: str):
    # Save datasets to a file
    output_name = os.path.join(output_dir, "datasets.object.gz")
    with gzip.open(output_name, "wb") as file:
        pickle.dump(datasets, file)
    logger.success(f"Datasets successfully saved in {output_name}.")


def create_visualizations_out_dirs(output_dir, dataset_name):
    # Base directory
    output_dir = os.path.join(output_dir, dataset_name)

    # Subdirectories
    subdirs = ["class_raw", "class_clean", "relation_raw", "relation_clean"]
    paths = [os.path.join(output_dir, subdir) for subdir in subdirs]

    # Create directories if they don't exist
    for path in paths:
        os.makedirs(path, exist_ok=True)

    # Unpack and return the paths
    return tuple(paths)
