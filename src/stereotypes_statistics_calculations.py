import numpy as np
import pandas as pd
from loguru import logger


# Unified function to calculate statistics for both class and relation stereotypes
def calculate_stereotype_metrics(models, stereotype_type: str, filter_type: str) -> dict:
    """
    Calculate statistics for class or relation stereotypes based on the provided type and filter.

    :param models: List of models.
    :param stereotype_type: 'class' or 'relation', which type of stereotype to calculate for.
    :param filter_type: 'gross' or 'net', determines how to filter the data.
    :return: Dictionary of calculated statistics.
    """
    # Step 1: Extract the data (either class_stereotypes or relation_stereotypes)
    data = extract_stereotype_data(models, stereotype_type, filter_type)

    # Step 2: Calculate frequency and other metrics
    frequency_analysis = calculate_frequency_analysis(data)
    diversity_measures = calculate_diversity_measures(data)
    central_tendency = calculate_central_tendency(data)
    coverage = calculate_coverage(data)

    # Combine all statistics into a single dictionary
    statistics = {**frequency_analysis, **diversity_measures, **central_tendency, **coverage}

    logger.success(f"{stereotype_type.capitalize()} stereotype statistics ({filter_type}) calculated successfully.")

    return statistics


# Extract stereotype data (works for both class and relation)
def extract_stereotype_data(models, stereotype_type: str, filter_type: str) -> pd.DataFrame:
    """
    Extract data for class or relation stereotypes.

    :param models: List of models.
    :param stereotype_type: 'class' or 'relation' to specify which stereotypes to extract.
    :param filter_type: 'gross' or 'net', applies filters if necessary.
    :return: DataFrame with extracted stereotype data.
    """

    # Collect data from each model based on stereotype type
    if stereotype_type == 'class':
        data = [model.class_stereotypes for model in models]
    elif stereotype_type == 'relation':
        data = [model.relation_stereotypes for model in models]

    df = pd.DataFrame(data)

    # Apply filtering if required (filter_type == 'net')
    if filter_type == 'net':
        df = df[df != 'irrelevant_stereotype']  # Example: filter out irrelevant stereotypes

    return df


# Function to calculate frequency analysis
def calculate_frequency_analysis(data: pd.DataFrame) -> dict:
    """
    Calculate frequency-related metrics from stereotype data.

    :param data: DataFrame containing stereotype data.
    :return: Dictionary of frequency-related metrics.
    """
    total_frequency = data.sum(axis=0)
    group_frequency = (data != 0).sum(axis=0)
    ubiquity_index = group_frequency / len(data)

    return {'Total Frequency': total_frequency, 'Group Frequency': group_frequency, 'Ubiquity Index': ubiquity_index, }


# Function to calculate diversity measures (e.g., Shannon, Gini, Simpson)
def calculate_diversity_measures(data: pd.DataFrame) -> dict:
    """
    Calculate diversity and distribution metrics (e.g., Shannon Entropy, Gini Coefficient, Simpson Index).

    :param data: DataFrame containing stereotype data.
    :return: Dictionary of diversity-related metrics.
    """

    # Shannon Entropy
    shannon_entropy = data.apply(lambda x: -np.sum((x / x.sum()) * np.log2(x / x.sum() + 1e-9)) if x.sum() > 0 else 0)

    # Gini Coefficient: Prevent division by zero when x.sum() is 0
    gini_coefficient = data.apply(
        lambda x: (2.0 * np.sum(np.arange(1, len(x) + 1) * np.sort(x)) - (len(x) + 1) * x.sum()) / (
                len(x) * x.sum()) if x.sum() != 0 else 0)

    # Simpson Index
    simpson_index = data.apply(lambda x: np.sum((x / x.sum()) ** 2) if x.sum() > 0 else 0)

    return {'Shannon Entropy': shannon_entropy, 'Gini Coefficient': gini_coefficient, 'Simpson Index': simpson_index, }


# Function to calculate central tendency and dispersion measures
def calculate_central_tendency(data: pd.DataFrame) -> dict:
    """
    Calculate central tendency (mean, median, mode) and dispersion (std, variance, skewness).

    :param data: DataFrame containing stereotype data.
    :return: Dictionary of central tendency and dispersion-related metrics.
    """
    mean = data.mean(axis=0)
    median = data.median(axis=0)
    mode = data.mode(axis=0).iloc[0]
    std = data.std(axis=0)
    variance = data.var(axis=0)

    return {'Mean': mean, 'Median': median, 'Mode': mode, 'Standard Deviation': std, 'Variance': variance, }


# Function to calculate coverage metrics
def calculate_coverage(data: pd.DataFrame) -> dict:
    """
    Calculate coverage of top percentages of stereotypes.

    :param data: DataFrame containing stereotype data.
    :return: Dictionary of coverage metrics.
    """
    total_occurrences = data.sum(axis=0).sum()
    coverage_list = []

    for pct in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
        top_k = data.sum(axis=0).nlargest(int(len(data.columns) * pct))
        coverage_list.append(
            {'Percentage': f'Top {int(pct * 100)}% Coverage', 'Coverage': top_k.sum() / total_occurrences})

    return {'Coverage': pd.DataFrame(coverage_list)}
