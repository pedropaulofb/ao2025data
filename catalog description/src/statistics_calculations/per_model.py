import pandas as pd
from loguru import logger
from ..common_calculations import calculate_class_and_relation_metrics, calculate_ratios


def calculate_metrics_per_model(models_df, classes_df, relations_df, output_file):
    # Dictionary to store output data per model
    output_data_per_model = {}

    # Iterate over each model
    for model in models_df['model']:
        # Filter classes and relations for the current model
        classes_in_model = classes_df[classes_df['model'] == model]
        relations_in_model = relations_df[relations_df['model'] == model]

        # Calculate metrics for classes and relations for this specific model
        class_metrics, class_total, class_stereotyped, class_non_stereotyped, class_ontouml, class_non_ontouml = calculate_class_and_relation_metrics(
            classes_in_model, 'classes')
        relation_metrics, relation_total, relation_stereotyped, relation_non_stereotyped, relation_ontouml, relation_non_ontouml = calculate_class_and_relation_metrics(
            relations_in_model, 'relations')

        # Extra Calculation: Different number of OntoUML stereotypes (excluding 'none' and 'other')
        def count_unique_stereotypes(df, label):
            stereotype_columns = df.columns.difference(['model', 'none', 'other'])
            unique_stereotypes = (df[stereotype_columns] > 0).any().sum()  # Count of unique stereotypes
            return unique_stereotypes

        # For classes
        unique_class_stereotypes = count_unique_stereotypes(classes_in_model, 'classes')

        # For relations
        unique_relation_stereotypes = count_unique_stereotypes(relations_in_model, 'relations')

        # Calculate ratios for the current model (with additional parameters)
        ratios = calculate_ratios(class_metrics['total_classes'], relation_metrics['total_relations'],
                                  class_metrics['stereotyped_classes'], relation_metrics['stereotyped_relations'],
                                  class_metrics['non_stereotyped_classes'], relation_metrics['non_stereotyped_relations'],
                                  class_metrics['ontouml_classes'], relation_metrics['ontouml_relations'],
                                  class_metrics['non_ontouml_classes'], relation_metrics['non_ontouml_relations'])


        # Prepare data for output for the current model
        output_data = {}
        output_data.update(class_metrics)
        output_data.update(relation_metrics)
        output_data.update(ratios)

        # Add unique stereotypes for this model
        output_data['unique_class_stereotypes'] = unique_class_stereotypes
        output_data['unique_relation_stereotypes'] = unique_relation_stereotypes

        # Save data for the current model
        output_data_per_model[model] = output_data

    # Create a DataFrame from the model data
    df_output_per_model = pd.DataFrame.from_dict(output_data_per_model, orient='index')

    # Write to CSV
    df_output_per_model.to_csv(output_file, index=True, index_label='model')

    logger.success(f"Statistics successfully saved per model to {output_file}.")
