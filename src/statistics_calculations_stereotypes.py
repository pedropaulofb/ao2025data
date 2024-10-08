import os
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import entropy

# Unified function to calculate statistics for both class and relation stereotypes
def calculate_stereotype_metrics(models, stereotype_type: str, filter_type: bool) -> dict:
    """
    Calculate statistics for class or relation stereotypes based on the provided type and filter.

    :param models: List of models.
    :param stereotype_type: 'class' or 'relation', which type of stereotype to calculate for.
    :param filter_type: boolean that, when true, remove 'other' and 'none' columns.
    :return: Dictionary of calculated statistics.
    """
    # Step 1: Extract the data (either class_stereotypes or relation_stereotypes)
    data = extract_stereotype_data(models, stereotype_type, filter_type)

    # Step 2: Calculate frequency analysis
    frequency_analysis_df = calculate_frequency_analysis(data)

    # Step 3: Calculate rank-frequency distribution
    rank_frequency_df = calculate_rank_frequency_distribution(data.sum(axis=0))

    # Step 4: Calculate group-wise rank-frequency distribution
    rank_groupwise_frequency_df = calculate_groupwise_rank_frequency_distribution(data)

    # Step 5: Calculate diversity measures
    diversity_measures_df = calculate_diversity_measures(data)

    # Step 6: Calculate central tendency and dispersion
    central_tendency_df = calculate_central_tendency(data)

    # Step 7: Calculate coverage metrics (occurrence-wise)
    coverage_df = calculate_coverage(data)

    # Step 8: Calculate similarity measures (Jaccard and Dice)
    similarity_measures_df = calculate_similarity_measures(data)

    # Step 9: Calculate correlation and dependency (Spearman, Mutual Information)
    spearman_correlation_df = calculate_spearman_correlation(data)
    mutual_info_df = calculate_mutual_information(data)

    # Combine all statistics into a dictionary
    statistics = {
        'frequency_analysis': frequency_analysis_df,
        'rank_frequency_distribution': rank_frequency_df,
        'rank_groupwise_frequency_distribution': rank_groupwise_frequency_df,
        'diversity_measures': diversity_measures_df,
        'central_tendency_dispersion': central_tendency_df,
        'coverage_metrics': coverage_df,
        'similarity_measures': similarity_measures_df,
        'spearman_correlation': spearman_correlation_df,
        'mutual_information': mutual_info_df,
    }

    return statistics

# Function to calculate frequency analysis
def calculate_frequency_analysis(data: pd.DataFrame) -> pd.DataFrame:
    group_num = len(data)
    total_frequency = data.sum(axis=0)
    total_frequency_per_group = total_frequency / group_num
    group_frequency = (data != 0).sum(axis=0)
    ubiquity_index = group_frequency / group_num
    global_relative_frequency_occurrence = total_frequency / total_frequency.sum()

    # Correct Calculation for Global Relative Frequency (Group-wise)
    total_groups = group_frequency.sum()
    global_relative_frequency_groupwise = group_frequency / total_groups

    # Combine into a single DataFrame
    frequency_analysis_df = pd.DataFrame({
        'Stereotype': total_frequency.index,
        'Total Frequency': total_frequency.values,
        'Total Frequency per Group': total_frequency_per_group,
        'Group Frequency': group_frequency.values,
        'Ubiquity Index': ubiquity_index,
        'Global Relative Frequency (Occurrence-wise)': global_relative_frequency_occurrence.values,
        'Global Relative Frequency (Group-wise)': global_relative_frequency_groupwise.values
    })

    return frequency_analysis_df


# Function to calculate rank-frequency distribution
def calculate_rank_frequency_distribution(data: pd.Series) -> pd.DataFrame:
    total_frequency_sorted = data.sort_values(ascending=False)
    total_occurrences = total_frequency_sorted.sum()

    rank_frequency_df = pd.DataFrame({
        'Stereotype': total_frequency_sorted.index,
        'Frequency': total_frequency_sorted.values,
        'Rank': range(1, len(total_frequency_sorted) + 1),
        'Cumulative Frequency': total_frequency_sorted.cumsum(),
        'Cumulative Percentage': (total_frequency_sorted.cumsum() / total_occurrences) * 100
    })

    return rank_frequency_df


# Function to calculate group-wise rank-frequency distribution
def calculate_groupwise_rank_frequency_distribution(data: pd.DataFrame) -> pd.DataFrame:
    group_frequency = (data != 0).sum(axis=0)
    group_frequency_sorted = group_frequency.sort_values(ascending=False)
    total_group_occurrences = group_frequency_sorted.sum()

    rank_groupwise_frequency_df = pd.DataFrame({
        'Stereotype': group_frequency_sorted.index,
        'Group-wise Frequency': group_frequency_sorted.values,
        'Group-wise Rank': range(1, len(group_frequency_sorted) + 1),
        'Cumulative Group-wise Frequency': group_frequency_sorted.cumsum(),
        'Group-wise Cumulative Percentage': (group_frequency_sorted.cumsum() / total_group_occurrences) * 100
    })

    return rank_groupwise_frequency_df


# Function to calculate diversity measures (Shannon, Gini, Simpson)
def calculate_diversity_measures(data: pd.DataFrame) -> pd.DataFrame:
    shannon_entropy = data.apply(
        lambda x: max(0, -np.sum((x / x.sum()) * np.log2(x / x.sum() + 1e-9)) if x.sum() > 0 else 0)
    )
    gini_coefficient = data.apply(
        lambda x: (2.0 * np.sum(np.arange(1, len(x) + 1) * np.sort(x)) - (len(x) + 1) * x.sum()) / (len(x) * x.sum()) if x.sum() != 0 else 0
    )
    simpson_index = data.apply(lambda x: np.sum((x / x.sum()) ** 2) if x.sum() > 0 else 0)

    diversity_measures_df = pd.DataFrame({
        'Stereotype': data.columns,
        'Shannon Entropy': shannon_entropy.values,
        'Gini Coefficient': gini_coefficient.values,
        'Simpson Index': simpson_index.values
    })

    return diversity_measures_df


# Function to calculate central tendency and dispersion measures with additional metrics
def calculate_central_tendency(data: pd.DataFrame) -> pd.DataFrame:
    mean = data.mean(axis=0)
    median = data.median(axis=0)
    mode = data.mode(axis=0).iloc[0] if not data.mode(axis=0).empty else np.nan
    std = data.std(axis=0)
    variance = data.var(axis=0)
    skewness = data.skew(axis=0)
    kurtosis = data.kurt(axis=0)
    q1 = data.quantile(0.25, axis=0)
    q3 = data.quantile(0.75, axis=0)
    iqr = q3 - q1
    max_value = data.max(axis=0)
    min_value = data.min(axis=0)
    range_value = max_value - min_value

    # Handling non-zero values
    non_zero_data = data.apply(lambda x: x[x > 0], axis=0)
    min_non_zero = non_zero_data.min(axis=0)
    range_non_zero = max_value - min_non_zero
    total = data.sum(axis=0)

    central_tendency_df = pd.DataFrame({
        'Stereotype': data.columns,
        'Total': total.values,
        'Mean': mean.values,
        'Median': median.values,
        'Mode': mode.values,
        'Standard Deviation': std.values,
        'Variance': variance.values,
        'Skewness': skewness.values,
        'Kurtosis': kurtosis.values,
        'Q1': q1.values,
        'Q3': q3.values,
        'IQR': iqr.values,
        'Min': min_value.values,
        'Max': max_value.values,
        'Range': range_value.values,
        'Min Non-Zero': min_non_zero.values,
        'Range Non-Zero': range_non_zero.values
    })

    return central_tendency_df



# Function to calculate coverage metrics (occurrence-wise and group-wise)
# Merged function to calculate both occurrence-wise and group-wise coverage metrics
def calculate_coverage(data: pd.DataFrame) -> pd.DataFrame:
    # Occurrence-wise data
    total_occurrences = data.sum(axis=0).sum()

    # Group-wise data
    group_frequency = (data != 0).sum(axis=0)
    total_groupwise_occurrences = group_frequency.sum()

    # Define percentages to calculate
    percentages = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

    # List to store coverage results
    combined_coverage_list = []

    for pct in percentages:
        # Occurrence-wise top k stereotypes
        top_k_occurrence = data.sum(axis=0).nlargest(int(len(data.columns) * pct))
        coverage_occurrence = top_k_occurrence.sum() / total_occurrences

        # Group-wise top k stereotypes
        top_k_groupwise = group_frequency.nlargest(int(len(group_frequency) * pct))
        coverage_groupwise = top_k_groupwise.sum() / total_groupwise_occurrences

        # Append combined result for each percentage
        combined_coverage_list.append({
            'Percentage': pct * 100,
            'Top k Stereotypes (Occurrence-wise)': len(top_k_occurrence),
            'Coverage (Occurrence-wise)': coverage_occurrence,
            'Top k Stereotypes (Group-wise)': len(top_k_groupwise),
            'Coverage (Group-wise)': coverage_groupwise
        })

    # Convert the list to a DataFrame
    combined_coverage_df = pd.DataFrame(combined_coverage_list)

    return combined_coverage_df


# Function to calculate similarity measures (Jaccard and Dice)
def calculate_similarity_measures(data: pd.DataFrame) -> pd.DataFrame:
    jaccard_similarity = {}
    dice_similarity = {}

    stereotypes = data.columns

    for (stereotype1, stereotype2) in combinations(stereotypes, 2):
        set1 = set(data[data[stereotype1] > 0].index)
        set2 = set(data[data[stereotype2] > 0].index)

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        jaccard_similarity[(stereotype1, stereotype2)] = intersection / union if union != 0 else 0
        dice_similarity[(stereotype1, stereotype2)] = (2 * intersection) / (len(set1) + len(set2)) if (len(set1) + len(set2)) != 0 else 0

    similarity_measures_df = pd.DataFrame({
        'Stereotype Pair': list(jaccard_similarity.keys()),
        'Jaccard Similarity': list(jaccard_similarity.values()),
        'Dice Coefficient': list(dice_similarity.values())
    })

    return similarity_measures_df


# Function to calculate Spearman Correlation
def calculate_spearman_correlation(data: pd.DataFrame, extra_calculation:bool=True) -> pd.DataFrame:
    spearman_correlation = data.corr(method='spearman')
    spearman_correlation.index.name = 'Stereotype'
    spearman_correlation.reset_index(inplace=True)

    return spearman_correlation


# Function to calculate Mutual Information
def calculate_mutual_information(data: pd.DataFrame) -> pd.DataFrame:
    stereotypes = data.columns
    mutual_info = pd.DataFrame(index=stereotypes, columns=stereotypes, dtype=float)

    for stereotype1, stereotype2 in combinations(stereotypes, 2):
        mi = mutual_info_score(data[stereotype1], data[stereotype2])
        mutual_info.loc[stereotype1, stereotype2] = mi
        mutual_info.loc[stereotype2, stereotype1] = mi

    for stereotype in stereotypes:
        entropy_value = entropy(data[stereotype])
        mutual_info.loc[stereotype, stereotype] = entropy_value

    mutual_info.index.name = 'Stereotype'
    mutual_info.reset_index(inplace=True)

    return mutual_info


# Function to extract stereotype data (works for both class and relation)
def extract_stereotype_data(models, stereotype_type: str, filter_type: bool) -> pd.DataFrame:
    if stereotype_type == 'class':
        data = [model.class_stereotypes for model in models]
    elif stereotype_type == 'relation':
        data = [model.relation_stereotypes for model in models]

    df = pd.DataFrame(data)

    # Remove 'other' and 'none' columns if filter_type is True
    if filter_type:
        df = df.drop(columns=["other", "none"], errors='ignore')

    return df

