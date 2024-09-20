import os.path

import pandas as pd
from loguru import logger


# Function to generate the first CSV file: Percent occurrence of each construct per year
def generate_yearly_statistics(constructs_file, years_file, output_file, clean):
    # Load the CSV files into DataFrames
    constructs_df = pd.read_csv(constructs_file)
    years_df = pd.read_csv(years_file)

    # Merge both DataFrames on the 'model' column
    merged_df = pd.merge(constructs_df, years_df, on='model')

    # Validate the input data
    if merged_df.empty:
        logger.error("Input CSV file is empty. Exiting the program.")
        return

    if clean:
        merged_df = merged_df.drop(columns=["other", "none"])

    if len(merged_df.columns) < 2:
        logger.error("Input CSV file does not have the expected columns. Exiting the program.")
        return

    # Drop the 'model' column since it is not needed for calculations
    merged_df.drop(columns=['model'], inplace=True)

    # Group by 'year' and sum occurrences for each year
    yearly_data = merged_df.groupby('year').sum()

    # Calculate the percentage per year so that the sum of all constructs per year equals 100%
    yearly_relative_frequencies = yearly_data.div(yearly_data.sum(axis=1), axis=0)

    # Save to CSV with 5 decimal places
    yearly_relative_frequencies.to_csv(output_file, float_format='%.7f')
    logger.success(f"Yearly Relative Frequency saved to {output_file}.")


# Function to generate the second CSV file: Percent occurrence across all years (but broken down by year)
def generate_overall_statistics(constructs_file, years_file, output_file, clean: bool):
    # Load the CSV files into DataFrames
    constructs_df = pd.read_csv(constructs_file)
    years_df = pd.read_csv(years_file)

    # Merge both DataFrames on the 'model' column
    merged_df = pd.merge(constructs_df, years_df, on='model')

    # Validate the input data
    if merged_df.empty:
        logger.error("Input CSV file is empty. Exiting the program.")
        return

    if clean:
        merged_df = merged_df.drop(columns=["other", "none"])

    if len(merged_df.columns) < 2:
        logger.error("Input CSV file does not have the expected columns. Exiting the program.")
        return

    # Drop the 'model' column since it is not needed for calculations
    merged_df.drop(columns=['model'], inplace=True)

    # Group by 'year' and sum occurrences for each year
    yearly_data = merged_df.groupby('year').sum()

    # Calculate the percentage of each construct per year
    overall_relative_frequencies = yearly_data.div(yearly_data.sum().sum())

    # Add 'year' back as the index (if it is dropped during grouping)
    overall_relative_frequencies['year'] = overall_relative_frequencies.index

    # Ensure the 'year' column is first by reordering the columns
    cols = ['year'] + [col for col in overall_relative_frequencies.columns if col != 'year']
    overall_relative_frequencies = overall_relative_frequencies[cols]

    # Save to CSV with 5 decimal places
    overall_relative_frequencies.to_csv(output_file, float_format='%.7f', index=False)
    logger.success(f"Overall Relative Frequency saved to {output_file}.")


if __name__ == "__main__":
    input_file_constructs = ["./outputs/consolidated_data/cs_ontouml_no_classroom.csv",
                             "./outputs/consolidated_data/rs_ontouml_no_classroom.csv"]
    input_file_years = "./outputs/ontouml_no_classroom_years.csv"

    for input_file_construct in input_file_constructs:
        analysis = os.path.splitext(os.path.basename(input_file_construct))[0]

        # Generate the first file: Yearly percentage occurrences
        generate_yearly_statistics(input_file_construct, input_file_years,
                                   os.path.join("./outputs/statistics", analysis + "_t", "yearly_statistics.csv"), True)
        generate_yearly_statistics(input_file_construct, input_file_years,
                                   os.path.join("./outputs/statistics", analysis + "_f", "yearly_statistics.csv"),
                                   False)
        # Generate the second file: Overall percentage occurrences
        generate_overall_statistics(input_file_construct, input_file_years,
                                    os.path.join("./outputs/statistics", analysis + "_t", "overall_statistics.csv"),
                                    True)
        generate_overall_statistics(input_file_construct, input_file_years,
                                    os.path.join("./outputs/statistics", analysis + "_f", "overall_statistics.csv"),
                                    False)
