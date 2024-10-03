import pandas as pd
from loguru import logger
from src.common_calculations import calculate_stats, calculate_class_and_relation_metrics





def calculate_metrics_general(models_df, classes_df, relations_df, output_file):
    # Calculate global metrics for classes and relations
    class_metrics, class_total, class_stereotyped, class_non_stereotyped, class_ontouml, class_non_ontouml = calculate_class_and_relation_metrics(
        classes_df, 'classes')
    relation_metrics, relation_total, relation_stereotyped, relation_non_stereotyped, relation_ontouml, relation_non_ontouml = calculate_class_and_relation_metrics(
        relations_df, 'relations')

    # Calculate per-model statistics
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

    # Prepare data for output
    output_data = {}
    output_data.update(class_metrics)
    output_data.update(relation_metrics)

    for key, stat_dict in metrics.items():
        for stat_name, value in stat_dict.items():
            output_data[f'{key}_{stat_name}'] = value

    # Write to CSV
    df_output = pd.DataFrame([output_data])
    df_output.to_csv(output_file, index=False)

    logger.success(f"Statistics successfully saved to {output_file}.")