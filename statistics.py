import pandas as pd
import numpy as np
from loguru import logger

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('./outputs/models_stats.csv')

# 1. Element Frequency Analysis

# Total Frequency of Each Element
total_frequency = df.iloc[:, 1:].sum(axis=0)
total_frequency_df = total_frequency.reset_index()
total_frequency_df.columns = ['Element', 'Total Frequency']
total_frequency_df.to_csv('./outputs/analyses/total_frequency.csv', index=False)
logger.success("Total Frequency of Each Element saved to CSV.")

# Group Frequency of Each Element
group_frequency = df.iloc[:, 1:].astype(bool).sum(axis=0)
group_frequency_df = group_frequency.reset_index()
group_frequency_df.columns = ['Element', 'Group Frequency']
group_frequency_df.to_csv('./outputs/analyses/group_frequency.csv', index=False)
logger.success("Group Frequency of Each Element saved to CSV.")

# 2. Relative Frequency

# Relative Frequency per Group
relative_frequency_per_group = df.iloc[:, 1:].div(df.iloc[:, 1:].sum(axis=1), axis=0)
relative_frequency_per_group.to_csv('./outputs/analyses/relative_frequency_per_group.csv', index=False)
logger.success("Relative Frequency per Group saved to CSV.")

# Global Relative Frequency (Occurrence-wise)
global_relative_frequency = total_frequency / total_frequency.sum()
global_relative_frequency_df = global_relative_frequency.reset_index()
global_relative_frequency_df.columns = ['Element', 'Global Relative Frequency (Occurrence-wise)']
global_relative_frequency_df.to_csv('./outputs/analyses/global_relative_frequency_occurrence-wise.csv', index=False)
logger.success("Global Relative Frequency (Occurrence-wise) saved to CSV.")

# Global Relative Frequency (Group-wise)
global_relative_frequency_groupwise = group_frequency / len(df)
global_relative_frequency_groupwise_df = global_relative_frequency_groupwise.reset_index()
global_relative_frequency_groupwise_df.columns = ['Element', 'Global Relative Frequency (Group-wise)']
global_relative_frequency_groupwise_df.to_csv('./outputs/analyses/global_relative_frequency_group-wise.csv', index=False)
logger.success("Global Relative Frequency (Group-wise) saved to CSV.")

# 3. Shannon Entropy

# Define a function to calculate Shannon Entropy for each element
def calculate_shannon_entropy(series):
    probabilities = series / series.sum()  # Calculate probabilities
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Entropy calculation
    return max(0.0, entropy)  # Return 0.0 instead of -0.0 if entropy is negative

# Apply Shannon Entropy function across the dataframe
shannon_entropy = df.iloc[:, 1:].apply(calculate_shannon_entropy)
shannon_entropy_df = shannon_entropy.reset_index()
shannon_entropy_df.columns = ['Element', 'Shannon Entropy']
shannon_entropy_df.to_csv('./outputs/analyses/shannon_entropy.csv', index=False)
logger.success("Shannon Entropy saved to CSV.")

# 4. Gini Coefficient

# Define a function to calculate Gini Coefficient for each element
def calculate_gini_coefficient(series):
    sorted_series = np.sort(series)  # Sort the values
    n = len(series)
    cumulative = np.cumsum(sorted_series)
    if cumulative[-1] == 0:  # Check for division by zero
        return 0  # Return zero if total is zero (uniform distribution)
    gini_index = (2.0 * np.sum((np.arange(1, n + 1) * sorted_series)) - (n + 1) * cumulative[-1]) / (n * cumulative[-1])
    return gini_index

# Apply Gini Coefficient function across the dataframe
gini_coefficient = df.iloc[:, 1:].apply(calculate_gini_coefficient)
gini_coefficient_df = gini_coefficient.reset_index()
gini_coefficient_df.columns = ['Element', 'Gini Coefficient']
gini_coefficient_df.to_csv('./outputs/analyses/gini_coefficient.csv', index=False)
logger.success("Gini Coefficient saved to CSV.")

# 5. Simpson's Index

# Define a function to calculate Simpson's Index for each element
def calculate_simpson_index(series):
    total = series.sum()
    if total == 0:
        return 0  # Handle case where total is zero
    probabilities = series / total
    return np.sum(probabilities ** 2)

# Apply Simpson's Index function across the dataframe
simpson_index = df.iloc[:, 1:].apply(calculate_simpson_index)
simpson_index_df = simpson_index.reset_index()
simpson_index_df.columns = ['Element', 'Simpson Index']
simpson_index_df.to_csv('./outputs/analyses/simpson_index.csv', index=False)
logger.success("Simpson's Index saved to CSV.")


# 6. Jaccard Similarity Index

# Compute the Jaccard Similarity Index between every pair of elements
from itertools import combinations

jaccard_similarity = {}
elements = df.columns[1:]

# Calculate Jaccard Similarity for each pair of elements
for (element1, element2) in combinations(elements, 2):
    set1 = set(df[df[element1] > 0].index)
    set2 = set(df[df[element2] > 0].index)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    jaccard_similarity[(element1, element2)] = intersection / union if union != 0 else 0

# Convert Jaccard Similarity dictionary to a DataFrame and save it
jaccard_similarity_df = pd.DataFrame(list(jaccard_similarity.items()), columns=['Element Pair', 'Jaccard Similarity'])
jaccard_similarity_df.to_csv('./outputs/analyses/jaccard_similarity_index.csv', index=False)
logger.success("Jaccard Similarity Index saved to CSV.")

# 7. Central Tendency and Dispersion

# Calculating all statistics
mean_counts = df.iloc[:, 1:].mean(axis=0)
median_counts = df.iloc[:, 1:].median(axis=0)
mode_counts = df.iloc[:, 1:].mode(axis=0).iloc[0]  # Mode may return multiple values, take the first mode
std_counts = df.iloc[:, 1:].std(axis=0)
variance_counts = df.iloc[:, 1:].var(axis=0)
skewness_counts = df.iloc[:, 1:].skew(axis=0)
kurtosis_counts = df.iloc[:, 1:].kurt(axis=0)

# Calculate the 25th percentile (Q1) and 75th percentile (Q3)
q1_counts = df.iloc[:, 1:].quantile(0.25, axis=0)  # 25th percentile
q3_counts = df.iloc[:, 1:].quantile(0.75, axis=0)  # 75th percentile

# Calculate the Interquartile Range (IQR)
iqr_counts = q3_counts - q1_counts

# Combine all statistics into a single DataFrame
statistics_df = pd.DataFrame({
    'Element': mean_counts.index,
    'Mean': mean_counts.values,
    'Median': median_counts.values,
    'Mode': mode_counts.values,
    'Standard Deviation': std_counts.values,
    'Variance': variance_counts.values,
    'Skewness': skewness_counts.values,
    'Kurtosis': kurtosis_counts.values,
    '25th Percentile (Q1)': q1_counts.values,
    '75th Percentile (Q3)': q3_counts.values,
    'Interquartile Range (IQR)': iqr_counts.values
})

# Save the consolidated statistics to a CSV file
statistics_df.to_csv('./outputs/analyses/consolidated_statistics.csv', index=False)
logger.success("Consolidated statistics (Mean, Median, Mode, Standard Deviation, Variance, Skewness, Kurtosis, Q1, Q3, IQR) saved to CSV.")



# 8. Rank-Frequency Distribution

# Calculate the total frequency of each element (already computed earlier)
total_frequency_sorted = total_frequency.sort_values(ascending=False)

# Create a DataFrame for rank and frequency
rank_frequency_df = pd.DataFrame({
    'Element': total_frequency_sorted.index,
    'Frequency': total_frequency_sorted.values,
    'Rank': range(1, len(total_frequency_sorted) + 1)
})

# Save the rank-frequency data to a CSV file
rank_frequency_df.to_csv('./outputs/analyses/rank_frequency.csv', index=False)
logger.success("Rank-Frequency data saved to CSV.")

# 9. Coverage and Ubiquity Measures

# Coverage of Top Percentage of Elements
percentages = [0.10, 0.20, 0.30, 0.40, 0.50]  # Define percentages
total_elements = len(total_frequency)  # Total number of elements
total_occurrences = total_frequency.sum()  # Total occurrences of all elements

# List to store coverage results
coverage_list = []

for pct in percentages:
    k = int(total_elements * pct)  # Calculate number of top elements for the given percentage
    top_k_elements = total_frequency.sort_values(ascending=False).head(k)  # Get top k elements by frequency
    coverage_top_k = top_k_elements.sum() / total_occurrences  # Calculate coverage
    coverage_list.append({'Percentage': pct * 100, 'Top k Elements': k, 'Coverage': coverage_top_k})

# Convert the list to a DataFrame
coverage_df = pd.DataFrame(coverage_list)

# Save the coverage data to a CSV file
coverage_df.to_csv('./outputs/analyses/coverage_percentage.csv', index=False)
logger.success("Coverage of Top Percentage of Elements saved to CSV.")

# Ubiquity Index
ubiquity_index = group_frequency / len(df)  # Fraction of groups in which each element appears
ubiquity_index_df = ubiquity_index.reset_index()
ubiquity_index_df.columns = ['Element', 'Ubiquity Index']
ubiquity_index_df.to_csv('./outputs/analyses/ubiquity_index.csv', index=False)
logger.success("Ubiquity Index saved to CSV.")


# 10. Correlation Measures

# Pearson Correlation Coefficient - Used for normal distributions
# pearson_correlation = df.iloc[:, 1:].corr(method='pearson')
# pearson_correlation.to_csv('./outputs/analyses/pearson_correlation.csv', index=False)
# logger.success("Pearson Correlation Coefficient saved to CSV.")

# Spearman Correlation Coefficient - Used for other distributions
spearman_correlation = df.iloc[:, 1:].corr(method='spearman')
spearman_correlation.to_csv('./outputs/analyses/spearman_correlation.csv', index=False)
logger.success("Spearman Correlation Coefficient saved to CSV.")
