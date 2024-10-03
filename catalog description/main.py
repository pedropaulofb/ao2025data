import pandas as pd

from general import calculate_metrics_general
from per_year import calculate_metrics_per_year

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
    models_file = '../outputs/ontouml_no_classroom_years.csv'
    classes_file = '../outputs/consolidated_data/cs_ontouml_no_classroom.csv'
    relations_file = '../outputs/consolidated_data/rs_ontouml_no_classroom.csv'

    output_file_general = './stats/catalog_statistics_general.csv'
    output_file_per_model = './stats/catalog_statistics_per_model.csv'
    output_file_per_year = './stats/catalog_statistics_per_year.csv'

    models_df, classes_df, relations_df = load_data(models_file, classes_file, relations_file)

    calculate_metrics_general(models_df, classes_df, relations_df, output_file_general)
    # calculate_metrics_per_model(classes_df, relations_df, output_file_per_model)
    calculate_metrics_per_year(models_df, classes_df, relations_df, output_file_per_year)