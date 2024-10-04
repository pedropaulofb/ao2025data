import pandas as pd
from loguru import logger
from ..common_calculations import calculate_stats, calculate_class_and_relation_metrics, calculate_ratios


def calculate_metrics_per_year_no_outliers(models_no_outliers_file, models_years_file, classes_df, relations_df, output_file):
    # Load the non-outliers file to get the list of non-outlier models
    models_no_outliers_df = pd.read_csv(models_no_outliers_file)

    # Load the models_years_file to get the year information for each model
    models_years_df = pd.read_csv(models_years_file)

    # Merge the non-outliers with the year data to obtain the years for the non-outlier models
    models_filtered_with_years = pd.merge(models_no_outliers_df[['model']], models_years_df, on='model')

    # Get unique years from the merged DataFrame
    years = models_filtered_with_years['year'].unique()

    # Dictionary to store output data per year
    output_data_per_year = {}

    # Iterate over each year
    for year in years:
        # Filter models for the current year
        models_in_year = models_filtered_with_years[models_filtered_with_years['year'] == year]['model']

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

        # Extra Calculation: Different number of OntoUML stereotypes (excluding 'none' and 'other')
        def count_unique_stereotypes(df, label):
            stereotype_columns = df.columns.difference(['model', 'none', 'other'])
            unique_stereotypes_per_model = df[stereotype_columns].astype(bool).sum(axis=1)
            total_unique_stereotypes_year = (df[stereotype_columns] > 0).any().sum()
            avg_unique_stereotypes_per_model = unique_stereotypes_per_model.mean()
            return total_unique_stereotypes_year, avg_unique_stereotypes_per_model

        # For classes
        total_unique_class_stereotypes, avg_unique_class_stereotypes_per_model = count_unique_stereotypes(
            classes_in_year, 'classes')

        # For relations
        total_unique_relation_stereotypes, avg_unique_relation_stereotypes_per_model = count_unique_stereotypes(
            relations_in_year, 'relations')

        # Add new metrics for unique OntoUML stereotypes
        metrics['total_unique_class_stereotypes'] = total_unique_class_stereotypes
        metrics['avg_unique_class_stereotypes_per_model'] = avg_unique_class_stereotypes_per_model
        metrics['total_unique_relation_stereotypes'] = total_unique_relation_stereotypes
        metrics['avg_unique_relation_stereotypes_per_model'] = avg_unique_relation_stereotypes_per_model

        # Calculate ratios for the current year (with additional parameters)
        ratios = calculate_ratios(class_metrics['total_classes'], relation_metrics['total_relations'],
                                  class_metrics['stereotyped_classes'], relation_metrics['stereotyped_relations'],
                                  class_metrics['non_stereotyped_classes'],
                                  relation_metrics['non_stereotyped_relations'],
                                  class_metrics['ontouml_classes'], relation_metrics['ontouml_relations'],
                                  class_metrics['non_ontouml_classes'], relation_metrics['non_ontouml_relations'])

        # Prepare data for output for the current year
        output_data = {}
        output_data.update(class_metrics)
        output_data.update(relation_metrics)
        output_data.update(ratios)

        for key, stat_dict in metrics.items():
            if isinstance(stat_dict, dict):
                for stat_name, value in stat_dict.items():
                    output_data[f'{key}_{stat_name}'] = value
            else:
                output_data[key] = stat_dict

        # Save data for the current year
        output_data_per_year[year] = output_data

    # Create a DataFrame from the yearly data
    df_output_per_year = pd.DataFrame.from_dict(output_data_per_year, orient='index')

    # Write to CSV
    df_output_per_year.to_csv(output_file, index=True, index_label='year')

    logger.success(f"Statistics per year (no outliers) successfully saved to {output_file}.")
