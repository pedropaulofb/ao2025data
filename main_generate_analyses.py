import numpy as np
import pandas as pd
from loguru import logger
from itertools import combinations
from sklearn.metrics import mutual_info_score


def save_to_csv(dataframe, filepath, message):
    """
    Utility function to save a DataFrame to a CSV file with error handling.
    """
    try:
        dataframe.to_csv(filepath, index=False)
        logger.success(message)
    except Exception as e:
        logger.error(f"Failed to save {filepath}: {e}")


def main() -> None:
    """
    Main function to generate statistical analysis of the consolidated data.
    """

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('./outputs/consolidated_all.csv')

    # Validate the input data
    if df.empty:
        logger.error("Input CSV file is empty. Exiting the program.")
        return

    if len(df.columns) < 2:
        logger.error("Input CSV file does not have the expected columns. Exiting the program.")
        return

    # Extract the data to be analyzed (excluding the first column)
    data = df.iloc[:, 1:]

    # 1. Frequency Analysis

    # Total Frequency of Each Construct
    total_frequency = data.sum(axis=0)

    # Group Frequency of Each Construct
    group_frequency = data.astype(bool).sum(axis=0)

    # Global Relative Frequency (Occurrence-wise)
    global_relative_frequency = total_frequency / total_frequency.sum()

    # Global Relative Frequency (Group-wise)
    global_relative_frequency_groupwise = group_frequency / len(df)

    # Combine all frequency-related data into a single DataFrame
    frequency_analysis_df = pd.DataFrame({'Construct': total_frequency.index, 'Total Frequency': total_frequency.values,
                                          'Group Frequency': group_frequency.values,
                                          'Global Relative Frequency (Occurrence-wise)': global_relative_frequency.values,
                                          'Global Relative Frequency (Group-wise)': global_relative_frequency_groupwise.values})

    # Save the consolidated frequency analysis data to a single CSV file
    save_to_csv(frequency_analysis_df, './outputs/analyses/frequency_analysis.csv',
                "Frequency Analysis (Total, Group, Relative Frequencies) saved to CSV.")

    # 2. Diversity and Distribution Measures

    # Define a function to calculate Shannon Entropy for each construct
    def calculate_shannon_entropy(series):
        probabilities = series / series.sum()  # Calculate probabilities
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Entropy calculation
        return max(0.0, entropy)  # Return 0.0 instead of -0.0 if entropy is negative

    # Apply Shannon Entropy function across the dataframe
    shannon_entropy = data.apply(calculate_shannon_entropy)

    # Define a function to calculate Gini Coefficient for each construct
    def calculate_gini_coefficient(series):
        sorted_series = np.sort(series)  # Sort the values
        n = len(series)
        cumulative = np.cumsum(sorted_series)
        if cumulative[-1] == 0:  # Check for division by zero
            return 0  # Return zero if total is zero (uniform distribution)
        gini_index = (2.0 * np.sum((np.arange(1, n + 1) * sorted_series)) - (n + 1) * cumulative[-1]) / (
                    n * cumulative[-1])
        return gini_index

    # Apply Gini Coefficient function across the dataframe
    gini_coefficient = data.apply(calculate_gini_coefficient)

    # Define a function to calculate Simpson's Index for each construct
    def calculate_simpson_index(series):
        total = series.sum()
        if total == 0:
            return 0  # Handle case where total is zero
        probabilities = series / total
        return np.sum(probabilities ** 2)

    # Apply Simpson's Index function across the dataframe
    simpson_index = data.apply(calculate_simpson_index)

    # Combine all diversity and distribution measures into a single DataFrame
    diversity_measures_df = pd.DataFrame({'Construct': df.columns[1:], 'Shannon Entropy': shannon_entropy.values,
                                          'Gini Coefficient': gini_coefficient.values,
                                          'Simpson Index': simpson_index.values})

    # Save the consolidated diversity and distribution measures data to a single CSV file
    save_to_csv(diversity_measures_df, './outputs/analyses/diversity_measures.csv',
                "Diversity and Distribution Measures (Shannon Entropy, Gini Coefficient, Simpson's Index) saved to CSV.")

    # 3. Central Tendency and Dispersion

    # Calculating all statistics
    mean_counts = data.mean(axis=0)
    median_counts = data.median(axis=0)
    mode_counts = data.mode(axis=0).iloc[0]  # Mode may return multiple values, take the first mode
    std_counts = data.std(axis=0)
    variance_counts = data.var(axis=0)
    skewness_counts = data.skew(axis=0)
    kurtosis_counts = data.kurt(axis=0)

    # Calculate the 25th percentile (Q1) and 75th percentile (Q3)
    q1_counts = data.quantile(0.25, axis=0)  # 25th percentile
    q3_counts = data.quantile(0.75, axis=0)  # 75th percentile

    # Calculate the Interquartile Range (IQR)
    iqr_counts = q3_counts - q1_counts

    # Combine all central tendency and dispersion statistics into a single DataFrame
    central_tendency_dispersion_df = pd.DataFrame(
        {'Construct': mean_counts.index, 'Mean': mean_counts.values, 'Median': median_counts.values,
         'Mode': mode_counts.values, 'Standard Deviation': std_counts.values, 'Variance': variance_counts.values,
         'Skewness': skewness_counts.values, 'Kurtosis': kurtosis_counts.values,
         '25th Percentile (Q1)': q1_counts.values, '75th Percentile (Q3)': q3_counts.values,
         'Interquartile Range (IQR)': iqr_counts.values})

    # Save the consolidated central tendency and dispersion data to a single CSV file
    save_to_csv(central_tendency_dispersion_df, './outputs/analyses/central_tendency_dispersion.csv',
                "Central Tendency and Dispersion Measures (Mean, Median, Mode, Standard Deviation, Variance, Skewness, Kurtosis, Q1, Q3, IQR) saved to CSV.")

    # 4. Similarity Measures

    # Initialize dictionaries to store the similarity values
    jaccard_similarity = {}
    dice_similarity = {}

    # List of constructs to compare
    constructs = df.columns[1:]

    # Calculate Jaccard Similarity and Dice Coefficient for each pair of constructs
    for (construct1, construct2) in combinations(constructs, 2):
        set1 = set(df[df[construct1] > 0].index)
        set2 = set(df[df[construct2] > 0].index)

        # Calculate the intersection and union of the sets
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        # Jaccard Similarity calculation
        jaccard_similarity[(construct1, construct2)] = intersection / union if union != 0 else 0

        # Dice Coefficient calculation
        dice_similarity[(construct1, construct2)] = (2 * intersection) / (len(set1) + len(set2)) if (len(set1) + len(
            set2)) != 0 else 0

    # Convert the similarity dictionaries to a DataFrame
    similarity_measures_df = pd.DataFrame(
        {'Construct Pair': list(jaccard_similarity.keys()), 'Jaccard Similarity': list(jaccard_similarity.values()),
         'Dice Coefficient': list(dice_similarity.values())})

    # Save the combined similarity measures data to a single CSV file
    save_to_csv(similarity_measures_df, './outputs/analyses/similarity_measures.csv',
                "Similarity Measures (Jaccard Similarity and Dice Coefficient) saved to CSV.")

    # 5. Coverage and Ubiquity Measures

    # 5.1. Coverage of Top Percentage of Constructs
    percentages = [0.10, 0.20, 0.30, 0.40, 0.50]  # Define percentages
    total_constructs = len(total_frequency)  # Total number of constructs
    total_occurrences = total_frequency.sum()  # Total occurrences of all constructs

    # List to store coverage results
    coverage_list = []

    for pct in percentages:
        k = int(total_constructs * pct)  # Calculate number of top constructs for the given percentage
        top_k_constructs = total_frequency.sort_values(ascending=False).head(k)  # Get top k constructs by frequency
        coverage_top_k = top_k_constructs.sum() / total_occurrences  # Calculate coverage
        coverage_list.append({'Percentage': pct * 100, 'Top k Constructs': k, 'Coverage': coverage_top_k})

    # Convert the list to a DataFrame
    coverage_df = pd.DataFrame(coverage_list)

    # Save the coverage data to a CSV file
    save_to_csv(coverage_df, './outputs/analyses/coverage_percentage.csv',
                "Coverage of Top Percentage of Constructs saved to CSV.")

    # 5.2. Ubiquity Index
    ubiquity_index = group_frequency / len(df)  # Fraction of groups in which each construct appears
    ubiquity_index_df = ubiquity_index.reset_index()
    ubiquity_index_df.columns = ['Construct', 'Ubiquity Index']
    save_to_csv(ubiquity_index_df, './outputs/analyses/ubiquity_index.csv', "Ubiquity Index saved to CSV.")

    # 6. Rank-Frequency Distribution with Cumulative Frequency and Percentage

    # Calculate the total frequency of each construct (already computed earlier)
    total_frequency_sorted = total_frequency.sort_values(ascending=False)

    # Calculate the total number of occurrences
    total_occurrences = total_frequency_sorted.sum()

    # Create a DataFrame for rank and frequency
    rank_frequency_df = pd.DataFrame(
        {'Construct': total_frequency_sorted.index, 'Frequency': total_frequency_sorted.values,
            'Rank': range(1, len(total_frequency_sorted) + 1)})

    # Calculate the cumulative frequency
    rank_frequency_df['Cumulative Frequency'] = rank_frequency_df['Frequency'].cumsum()

    # Calculate the cumulative percentage
    rank_frequency_df['Cumulative Percentage'] = (rank_frequency_df['Cumulative Frequency'] / total_occurrences) * 100

    # Save the rank-frequency, cumulative frequency, and cumulative percentage data to a single CSV file
    save_to_csv(rank_frequency_df, './outputs/analyses/rank_frequency_distribution.csv',
                "Rank-Frequency, Cumulative Frequency, and Cumulative Percentage data saved to CSV.")

    # 7. Correlation and Dependency Measures

    # Spearman Correlation Coefficient - Used for other distributions
    spearman_correlation = data.corr(method='spearman')

    # Add 'Construct' as the first column header
    spearman_correlation.index.name = 'Construct'

    # Save Spearman Correlation Coefficient to its own CSV file
    save_to_csv(spearman_correlation, './outputs/analyses/spearman_correlation.csv',
                "Spearman Correlation Coefficient saved to CSV.")

    # Mutual Information - Capture non-linear dependencies

    # Initialize a DataFrame to store Mutual Information
    mutual_info = pd.DataFrame(index=constructs, columns=constructs, dtype=float)

    # Compute Mutual Information for all pairs of constructs
    for construct1, construct2 in combinations(constructs, 2):
        mi = mutual_info_score(df[construct1], df[construct2])
        mutual_info.loc[construct1, construct2] = mi
        mutual_info.loc[construct2, construct1] = mi  # Symmetric matrix

    # Add 'Construct' as the first column header
    mutual_info.index.name = 'Construct'

    # Save Mutual Information matrix to its own CSV file
    save_to_csv(mutual_info, './outputs/analyses/mutual_information.csv', "Mutual Information saved to CSV.")


if __name__ == "__main__":
    main()
