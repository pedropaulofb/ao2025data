import pandas as pd
from loguru import logger


# Function to calculate metrics per model without outliers
def calculate_metrics_per_model_no_outliers(models_df, classes_df, relations_df, output_file):
    # Calculate the IQR for total_classes and total_relations
    Q1_classes = models_df['total_classes'].quantile(0.25)
    Q3_classes = models_df['total_classes'].quantile(0.75)
    IQR_classes = Q3_classes - Q1_classes

    Q1_relations = models_df['total_relations'].quantile(0.25)
    Q3_relations = models_df['total_relations'].quantile(0.75)
    IQR_relations = Q3_relations - Q1_relations

    # Define the bounds for non-outliers for classes and relations
    lower_bound_classes = Q1_classes - 1.5 * IQR_classes
    upper_bound_classes = Q3_classes + 1.5 * IQR_classes

    lower_bound_relations = Q1_relations - 1.5 * IQR_relations
    upper_bound_relations = Q3_relations + 1.5 * IQR_relations

    # Filter out models that are outliers for total_classes or total_relations
    df_no_outliers = models_df[
        (models_df['total_classes'] >= lower_bound_classes) & (models_df['total_classes'] <= upper_bound_classes) &
        (models_df['total_relations'] >= lower_bound_relations) & (models_df['total_relations'] <= upper_bound_relations)
    ]

    # Save the filtered data to a new CSV file
    df_no_outliers.to_csv(output_file, index=False)

    # Check if any outliers were removed
    if len(df_no_outliers) == len(models_df):
        logger.success(f"No outliers found. All models are within the normal range. Number of models: {len(models_df)}.")
    else:
        logger.success(f"Outliers removed. Original number of models: {len(models_df)}. Number of models after removing outliers: {len(df_no_outliers)}.")

