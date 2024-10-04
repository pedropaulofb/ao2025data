import pandas as pd

from src.statistics_calculations.general import calculate_metrics_general
from src.statistics_calculations.per_year import calculate_metrics_per_year
from src.statistics_calculations.per_model import calculate_metrics_per_model
from src.statistics_calculations.per_model_no_outliers import calculate_metrics_per_model_no_outliers
from src.statistics_calculations.general_no_ouliers import calculate_metrics_general_no_outliers
from src.statistics_calculations.per_year_no_outliers import calculate_metrics_per_year_no_outliers


# Load the CSV files
def load_data(models_file, classes_file, relations_file):
    models_df = pd.read_csv(models_file, header=0)
    classes_df = pd.read_csv(classes_file, header=0)
    relations_df = pd.read_csv(relations_file, header=0)

    # Remove the 'model' header row if it exists as a data row
    models_df = models_df[models_df['model'] != 'model']
    classes_df = classes_df[classes_df['model'] != 'model']
    relations_df = relations_df[relations_df['model'] != 'model']

    return models_df, classes_df, relations_df


if __name__ == "__main__":
    # Load original data
    models_file = '../outputs/ontouml_no_classroom_years.csv'
    classes_file = '../outputs/consolidated_data/cs_ontouml_no_classroom.csv'
    relations_file = '../outputs/consolidated_data/rs_ontouml_no_classroom.csv'

    output_file_general = 'statistics/catalog_statistics_general.csv'
    output_file_per_model = 'statistics/catalog_statistics_per_model.csv'
    output_file_per_year = 'statistics/catalog_statistics_per_year.csv'

    models_df, classes_df, relations_df = load_data(models_file, classes_file, relations_file)

    # Calculate general metrics
    calculate_metrics_general(models_df, classes_df, relations_df, output_file_general)
    # Calculate per year
    calculate_metrics_per_year(models_df, classes_df, relations_df, output_file_per_year)
    # Calculate per model
    calculate_metrics_per_model(models_df, classes_df, relations_df, output_file_per_model)

    # Calculating without outliers
    models_file = 'statistics/catalog_statistics_per_model_no_outliers.csv'  # THIS FILE CONTAINS 'total_classes' column
    models_file_years = '../outputs/ontouml_no_classroom_years.csv'
    output_file_per_model_no_outliers = 'statistics/catalog_statistics_per_model_no_outliers.csv'
    output_file_general_no_outliers = 'statistics/catalog_statistics_general_no_outliers.csv'
    output_file_per_year_no_outliers = 'statistics/catalog_statistics_per_year_no_outliers.csv'

    # Reload data from the statistics file that contains 'total_classes'
    models_df, classes_df, relations_df = load_data(models_file, classes_file, relations_file)

    # Calculate per model without outliers
    calculate_metrics_per_model_no_outliers(models_df, classes_df, relations_df, output_file_per_model_no_outliers)

    # Calculate general metrics without outliers
    calculate_metrics_general_no_outliers(output_file_per_model_no_outliers, classes_df, relations_df,
                                          output_file_general_no_outliers)

    # Calculate per year without outliers
    calculate_metrics_per_year_no_outliers(output_file_per_model_no_outliers, models_file_years, classes_df,
                                           relations_df, output_file_per_year_no_outliers)
