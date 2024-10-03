import os.path
import pandas as pd
from loguru import logger

# Function to generate the first CSV file: Percent occurrence of each stereotype per year
def generate_yearly_statistics(stereotypes_file, years_file, output_file, clean):
    # Load the CSV files into DataFrames
    stereotypes_df = pd.read_csv(stereotypes_file)
    years_df = pd.read_csv(years_file)

    # Merge both DataFrames on the 'model' column
    merged_df = pd.merge(stereotypes_df, years_df, on='model')

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

    # Calculate the percentage per year so that the sum of all stereotypes per year equals 100%
    yearly_relative_frequencies = yearly_data.div(yearly_data.sum(axis=1), axis=0)

    # Save to CSV with 5 decimal places
    output_file = os.path.normpath(output_file)
    yearly_relative_frequencies.to_csv(output_file, float_format='%.7f')
    logger.success(f"Yearly Relative Frequency (Occurrence-wise) saved to {output_file}.")

# Function to generate the second CSV file: Percent occurrence across all years (but broken down by year)
def generate_overall_statistics(stereotypes_file, years_file, output_file, clean: bool):
    # Load the CSV files into DataFrames
    stereotypes_df = pd.read_csv(stereotypes_file)
    years_df = pd.read_csv(years_file)

    # Merge both DataFrames on the 'model' column
    merged_df = pd.merge(stereotypes_df, years_df, on='model')

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

    # Calculate the percentage of each stereotype per year
    overall_relative_frequencies = yearly_data.div(yearly_data.sum().sum())

    # Add 'year' back as the index (if it is dropped during grouping)
    overall_relative_frequencies['year'] = overall_relative_frequencies.index

    # Ensure the 'year' column is first by reordering the columns
    cols = ['year'] + [col for col in overall_relative_frequencies.columns if col != 'year']
    overall_relative_frequencies = overall_relative_frequencies[cols]

    # Save to CSV with 5 decimal places
    output_file = os.path.normpath(output_file)
    overall_relative_frequencies.to_csv(output_file, float_format='%.7f', index=False)
    logger.success(f"Overall Relative Frequency (Occurrence-wise) saved to {output_file}.")

# Function to generate yearly-normalized model-wise frequency
def generate_yearly_modelwise_statistics(stereotypes_file, years_file, output_file, clean):
    # Load the CSV files into DataFrames
    stereotypes_df = pd.read_csv(stereotypes_file)
    years_df = pd.read_csv(years_file)

    # Merge both DataFrames on the 'model' column
    merged_df = pd.merge(stereotypes_df, years_df, on='model')

    # Validate the input data
    if merged_df.empty:
        logger.error("Input CSV file is empty. Exiting the program.")
        return

    # Drop columns if clean is True
    if clean:
        merged_df = merged_df.drop(columns=["other", "none"])

    # Drop the 'model' column since it's not needed for calculation
    merged_df.drop(columns=['model'], inplace=True)

    # Convert non-zero values to 1 (indicating the presence of a stereotype in a model)
    modelwise_df = merged_df.iloc[:, :-1].apply(lambda x: x.gt(0).astype(int))

    # Group by 'year' and count the number of models containing each stereotype
    yearly_modelwise_data = modelwise_df.groupby(merged_df['year']).sum()

    # Calculate the yearly normalized frequency (relative frequency within each year)
    yearly_normalized = yearly_modelwise_data.div(yearly_modelwise_data.sum(axis=1), axis=0)

    # Save to CSV with 5 decimal places
    yearly_normalized.to_csv(output_file, float_format='%.5f')
    logger.success(f"Yearly Normalized Model-wise Frequency saved to {output_file}.")


# Function to generate overall-normalized model-wise frequency
def generate_overall_modelwise_statistics(stereotypes_file, years_file, output_file, clean):
    # Load the CSV files into DataFrames
    stereotypes_df = pd.read_csv(stereotypes_file)
    years_df = pd.read_csv(years_file)

    # Merge both DataFrames on the 'model' column
    merged_df = pd.merge(stereotypes_df, years_df, on='model')

    # Validate the input data
    if merged_df.empty:
        logger.error("Input CSV file is empty. Exiting the program.")
        return

    # Drop columns if clean is True
    if clean:
        merged_df = merged_df.drop(columns=["other", "none"])

    # Drop the 'model' column since it's not needed for calculation
    merged_df.drop(columns=['model'], inplace=True)

    # Convert non-zero values to 1 (indicating the presence of a stereotype in a model)
    modelwise_df = merged_df.iloc[:, :-1].apply(lambda x: x.gt(0).astype(int))

    # Group by 'year' and sum the values
    yearly_modelwise_data = modelwise_df.groupby(merged_df['year']).sum()

    # Calculate the overall normalized frequency (relative frequency across all years)
    overall_normalized = yearly_modelwise_data.div(yearly_modelwise_data.sum().sum())

    # Save to CSV with 5 decimal places
    overall_normalized.to_csv(output_file, float_format='%.5f')
    logger.success(f"Overall Normalized Model-wise Frequency saved to {output_file}.")


if __name__ == "__main__":
    input_file_stereotypes = ["./outputs/consolidated_data/cs_ontouml_no_classroom_after_2015.csv",
                              "./outputs/consolidated_data/rs_ontouml_no_classroom_after_2015.csv"]
    input_file_years = "./outputs/ontouml_no_classroom_years.csv"

    for input_file_stereotype in input_file_stereotypes:
        analysis = os.path.splitext(os.path.basename(input_file_stereotype))[0]

        # Generate the first file: Yearly percentage occurrences (Occurrence-wise)
        generate_yearly_statistics(input_file_stereotype, input_file_years,
                                   os.path.join("./outputs/statistics", analysis + "_t", "temporal_yearly_stats.csv"),
                                   True)
        generate_yearly_statistics(input_file_stereotype, input_file_years,
                                   os.path.join("./outputs/statistics", analysis + "_f", "temporal_yearly_stats.csv"),
                                   False)

        # Generate the second file: Overall percentage occurrences (Occurrence-wise)
        generate_overall_statistics(input_file_stereotype, input_file_years,
                                    os.path.join("./outputs/statistics", analysis + "_t", "temporal_overall_stats.csv"),
                                    True)
        generate_overall_statistics(input_file_stereotype, input_file_years,
                                    os.path.join("./outputs/statistics", analysis + "_f", "temporal_overall_stats.csv"),
                                    False)

        # Generate the third file: Yearly percentage of models (Model-wise)
        generate_yearly_modelwise_statistics(input_file_stereotype, input_file_years,
                                             os.path.join("./outputs/statistics", analysis + "_t", "temporal_yearly_modelwise_stats.csv"),
                                             True)
        generate_yearly_modelwise_statistics(input_file_stereotype, input_file_years,
                                             os.path.join("./outputs/statistics", analysis + "_f", "temporal_yearly_modelwise_stats.csv"),
                                             False)

        # Generate the fourth file: Overall percentage of models (Model-wise)
        generate_overall_modelwise_statistics(input_file_stereotype, input_file_years,
                                              os.path.join("./outputs/statistics", analysis + "_t", "temporal_overall_modelwise_stats.csv"),
                                              True)
        generate_overall_modelwise_statistics(input_file_stereotype, input_file_years,
                                              os.path.join("./outputs/statistics", analysis + "_f", "temporal_overall_modelwise_stats.csv"),
                                              False)
