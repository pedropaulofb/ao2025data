import pandas as pd
from loguru import logger


def classify_and_save_spearman_correlations(spearman_correlation: pd.DataFrame, output_filepath: str) -> None:
    """
    Classify Spearman correlation values and save them to a CSV file.

    :param spearman_correlation: DataFrame with Spearman correlation values.
    :param output_filepath: Filepath to save the classified correlations CSV.
    """
    # Melt the DataFrame to get pairwise correlations (excluding self-correlations)
    correlations = spearman_correlation.melt(id_vars=['Stereotype'], var_name='Stereotype2', value_name='Correlation')
    correlations = correlations[correlations['Stereotype'] != correlations['Stereotype2']]  # Exclude diagonal

    # Function to classify correlation based on absolute value
    def classify_correlation(value):
        abs_value = abs(value)
        if abs_value < 0.2:
            return 'very_weak'
        elif 0.2 <= abs_value < 0.4:
            return 'weak'
        elif 0.4 <= abs_value < 0.6:
            return 'moderate'
        elif 0.6 <= abs_value < 0.8:
            return 'strong'
        else:
            return 'very_strong'

    # Add classification and other columns
    correlations['Absolute Value'] = correlations['Correlation'].abs()
    correlations['Sign'] = correlations['Correlation'].apply(lambda x: 'positive' if x > 0 else 'negative')
    correlations['Classification'] = correlations['Correlation'].apply(classify_correlation)

    # Select and rename columns for the final output
    output_df = correlations[['Stereotype', 'Stereotype2', 'Absolute Value', 'Sign','Classification']]

    # Save the result to a CSV file
    output_df.to_csv(output_filepath, index=False)
