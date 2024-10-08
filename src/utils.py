# Utility function to save to CSV
import pandas as pd
from loguru import logger


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
