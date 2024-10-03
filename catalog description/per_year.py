import pandas as pd
from loguru import logger

from src.common_calculations import calculate_stats, calculate_class_and_relation_metrics

def calculate_metrics_per_year(models_df, classes_df, relations_df, output_file):
    # Get unique years from the models dataframe
    years = models_df['year'].unique()

    # Dictionary to store output data per year
    output_data_per_year = {}

    # Iterate over each year
    for year in years:
        # Filter models for the current year
        models_in_year = models_df[models_df['year'] == year]['model']

        # Filter classes and relations based on the models for the current year
        classes_in_year = classes_df[classes_df['model'].isin(models_in_year)]
        relations_in_year = relations_df[relations_df['model'].isin(models_in_year)]

        # Calculate metrics for classes and relations for this specific year
        class_metrics, class_total, class_stereotyped, class_non_stereotyped, class_ontouml, class_non_ontouml = calculate_class_and_relation_metrics(
            classes_in_year, 'classes')
        relation_metrics, relation_total, relation_stereotyped, relation_non_stereotyped, relation_ontouml, relation_non_ontouml = calculate_class_and_relation_metrics(
            relations_in_year, 'relations')

        # Calculate statistics (max, min, etc.) for each model in this year
        metrics = {
            'class_total': calculate_stats(class_total),
            'class_stereotyped': calculate_stats(class_stereotyped),
            'class_non_stereotyped': calculate_stats(class_non_stereotyped),
            'class_ontouml': calculate_stats(class_ontouml),
            'class_non_ontouml': calculate_stats(class_non_ontouml),
            'relation_total': calculate_stats(relation_total),
            'relation_stereotyped': calculate_stats(relation_stereotyped),
            'relation_non_stereotyped': calculate_stats(relation_non_stereotyped),
            'relation_ontouml': calculate_stats(relation_ontouml),
            'relation_non_ontouml': calculate_stats(relation_non_ontouml)
        }

        # Prepare data for output for the current year
        output_data = {}
        output_data.update(class_metrics)
        output_data.update(relation_metrics)

        for key, stat_dict in metrics.items():
            for stat_name, value in stat_dict.items():
                output_data[f'{key}_{stat_name}'] = value

        # Save data for the current year
        output_data_per_year[year] = output_data

    # Create a DataFrame from the yearly data
    df_output_per_year = pd.DataFrame.from_dict(output_data_per_year, orient='index')

    # Write to CSV
    df_output_per_year.to_csv(output_file, index=True, index_label='year')

    logger.success(f"Statistics successfully saved per year to {output_file}.")
