import pandas as pd
import numpy as np
from loguru import logger
from scipy.stats import skew, kurtosis

def calculate_stats(data):
    stats = {}
    stats['max'] = np.max(data)
    stats['min'] = np.min(data)

    # Handling for non-zero values
    non_zero_data = data[data > 0]
    if len(non_zero_data) > 0:
        stats['min_non_zero'] = np.min(non_zero_data)
        stats['range_non_zero'] = stats['max'] - stats['min_non_zero']
    else:
        # If there are no non-zero values, handle gracefully
        stats['min_non_zero'] = 0
        stats['range_non_zero'] = 0

    stats['range'] = stats['max'] - stats['min']
    stats['range_non_zero'] = stats['max'] - stats['min_non_zero']
    stats['mean'] = np.mean(data)
    stats['median'] = np.median(data)
    stats['mode'] = pd.Series(data).mode().iloc[0] if not pd.Series(data).mode().empty else np.nan
    stats['std_dev'] = np.std(data)
    stats['variance'] = np.var(data)
    stats['q1'] = np.percentile(data, 25)
    stats['q2'] = np.percentile(data, 50)  # Same as median
    assert stats['q2'] == stats['median'], f"Q2 should be the same as median."
    stats['q3'] = np.percentile(data, 75)
    stats['iqr'] = stats['q3'] - stats['q1']

    try:
        stats['skewness'] = skew(data)
    except Exception as e:
        stats['skewness'] = np.nan
        logger.warning(f"Skewness calculation failed: {e}")

    try:
        stats['kurtosis'] = kurtosis(data)
    except Exception as e:
        stats['kurtosis'] = np.nan
        logger.warning(f"Kurtosis calculation failed: {e}")

    return stats

def calculate_class_and_relation_metrics(df, label):
    # Total number of classes or relations
    total = df.iloc[:, 1:].sum(axis=1)

    # Stereotyped (all but 'none')
    stereotyped = total - df['none']

    # Non-stereotyped (only 'none')
    non_stereotyped = df['none']

    # OntoUML (all but 'none' and 'other')
    ontouml = total - df['none'] - df['other']

    # Non-OntoUML ('none' + 'other')
    non_ontouml = df['none'] + df['other']

    # Add assertions to ensure the totals match
    assert total.sum() == stereotyped.sum() + non_stereotyped.sum(), f"Total {label} must equal stereotyped + non-stereotyped {label}"
    assert total.sum() == ontouml.sum() + non_ontouml.sum(), f"Total {label} must equal OntoUML + non-OntoUML {label}"

    return {
        f'total_{label}': total.sum(),
        f'stereotyped_{label}': stereotyped.sum(),
        f'non_stereotyped_{label}': non_stereotyped.sum(),
        f'ontouml_{label}': ontouml.sum(),
        f'non_ontouml_{label}': non_ontouml.sum()
    }, total, stereotyped, non_stereotyped, ontouml, non_ontouml