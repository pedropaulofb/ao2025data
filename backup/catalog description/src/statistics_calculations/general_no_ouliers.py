import pandas as pd
from loguru import logger
from ..common_calculations import calculate_stats, calculate_class_and_relation_metrics, calculate_ratios


def calculate_metrics_general_no_outliers(models_no_outliers_file, classes_df, relations_df, output_file):
    # Load the non-outliers file to get the list of non-outlier models
    models_no_outliers_df = pd.read_csv(models_no_outliers_file)

    # Filter the classes_df and relations_df based on the models that are not outliers
    non_outlier_models = models_no_outliers_df['model'].unique()

    classes_df_filtered = classes_df[classes_df['model'].isin(non_outlier_models)]
    relations_df_filtered = relations_df[relations_df['model'].isin(non_outlier_models)]

    # Calculate global metrics for classes and relations using the filtered data
    class_metrics, class_total, class_stereotyped, class_non_stereotyped, class_ontouml, class_non_ontouml = calculate_class_and_relation_metrics(
        classes_df_filtered, 'classes')
    relation_metrics, relation_total, relation_stereotyped, relation_non_stereotyped, relation_ontouml, relation_non_ontouml = calculate_class_and_relation_metrics(
        relations_df_filtered, 'relations')

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

    # Calculate ratios for the filtered dataset (no outliers)
    ratios = calculate_ratios(class_metrics['total_classes'], relation_metrics['total_relations'],
                              class_metrics['stereotyped_classes'], relation_metrics['stereotyped_relations'],
                              class_metrics['non_stereotyped_classes'], relation_metrics['non_stereotyped_relations'],
                              class_metrics['ontouml_classes'], relation_metrics['ontouml_relations'],
                              class_metrics['non_ontouml_classes'], relation_metrics['non_ontouml_relations'])

    # Prepare data for output
    output_data = {}
    output_data.update(class_metrics)
    output_data.update(relation_metrics)
    output_data.update(ratios)

    for key, stat_dict in metrics.items():
        for stat_name, value in stat_dict.items():
            output_data[f'{key}_{stat_name}'] = value

    # Write to CSV
    df_output = pd.DataFrame([output_data])
    df_output.to_csv(output_file, index=False)

    logger.success(f"Statistics successfully saved to {output_file}.")
