import csv
import math
import os

import numpy as np
import pandas as pd
from loguru import logger

from src import ModelData
from src.calculations.statistics_calculations_datasets import calculate_class_and_relation_metrics, calculate_stats, \
    calculate_ratios
from src.calculations.statistics_calculations_stereotypes import calculate_stereotype_metrics
from src.calculations.statistics_calculations_stereotypes_extra import classify_and_save_spearman_correlations, \
    calculate_quadrants_and_save
from src.utils import append_unique_preserving_order, save_to_csv


class Dataset():
    def __init__(self, name: str, models: list[ModelData]) -> None:

        self.name: str = name
        self.models: list[ModelData] = models

        self.num_models: int = len(models)
        self.num_classes: int = -1
        self.num_relations: int = -1

        self.statistics = {}

        self.years_stereotypes_data = {}

        self.class_statistics_raw = {}
        self.class_statistics_clean = {}

        self.relation_statistics_raw = {}
        self.relation_statistics_clean = {}

        self.combined_statistics_raw = {}
        self.combined_statistics_clean = {}

    def save_dataset_general_data_csv(self, output_dir: str) -> None:
        output_dir = os.path.join(output_dir, self.name)

        # Create folder if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        # Prepare the data for the CSV
        data = [[model.name, model.year, model.total_class_number, model.total_relation_number] for model in
                self.models]
        df = pd.DataFrame(data, columns=["model", "year", "total_class_number", "total_relation_number"])

        # Save to CSV using the common utility function
        filepath = os.path.join(output_dir, f'{self.name}_data.csv')
        save_to_csv(df, filepath, f"General data for dataset '{self.name}' successfully saved to {filepath}.")

    def save_dataset_class_data_csv(self, output_dir: str) -> None:
        output_dir = os.path.join(output_dir, self.name)

        # Create folder if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        # Extract all the stereotypes from the first model
        stereotypes = list(self.models[0].class_stereotypes.keys())

        # Create a DataFrame for class data
        data = [[model.name] + [model.class_stereotypes[st] for st in stereotypes] for model in self.models]
        df = pd.DataFrame(data, columns=["model"] + stereotypes)
        self.data_class = df

        # Save to CSV using the common utility function
        filepath = os.path.join(output_dir, f'{self.name}_class_data.csv')
        save_to_csv(df, filepath, f"Class data for dataset '{self.name}' successfully saved to {filepath}.")

    def save_dataset_relation_data_csv(self, output_dir: str) -> None:
        output_dir = os.path.join(output_dir, self.name)

        # Create folder if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        # Extract all the relation stereotypes from the first model
        stereotypes = list(self.models[0].relation_stereotypes.keys())

        # Create a DataFrame for relation data
        data = [[model.name] + [model.relation_stereotypes[st] for st in stereotypes] for model in self.models]
        df = pd.DataFrame(data, columns=["model"] + stereotypes)
        self.data_relation = df

        # Save to CSV using the common utility function
        filepath = os.path.join(output_dir, f'{self.name}_relation_data.csv')
        save_to_csv(df, filepath, f"Relation data for dataset '{self.name}' successfully saved to {filepath}.")

    def calculate_dataset_statistics(self) -> None:
        """Calculates statistics and metrics for the dataset and stores them in self.statistics."""

        # Step 1: Prepare the data for class and relation metrics
        class_data = self._create_dataframe_for_stereotypes("class_stereotypes")
        relation_data = self._create_dataframe_for_stereotypes("relation_stereotypes")

        # Step 2: Calculate class and relation metrics
        class_metrics, class_total, class_stereotyped, class_non_stereotyped, class_ontouml, class_non_ontouml = calculate_class_and_relation_metrics(
            class_data, 'classes')
        relation_metrics, relation_total, relation_stereotyped, relation_non_stereotyped, relation_ontouml, relation_non_ontouml = calculate_class_and_relation_metrics(
            relation_data, 'relations')

        # Step 3: Calculate statistics for the dataset
        metrics = {'class_total': calculate_stats(class_total), 'class_stereotyped': calculate_stats(class_stereotyped),
                   'class_non_stereotyped': calculate_stats(class_non_stereotyped),
                   'class_ontouml': calculate_stats(class_ontouml),
                   'class_non_ontouml': calculate_stats(class_non_ontouml),
                   'relation_total': calculate_stats(relation_total),
                   'relation_stereotyped': calculate_stats(relation_stereotyped),
                   'relation_non_stereotyped': calculate_stats(relation_non_stereotyped),
                   'relation_ontouml': calculate_stats(relation_ontouml),
                   'relation_non_ontouml': calculate_stats(relation_non_ontouml)}

        # Step 4: Calculate ratios
        ratios = calculate_ratios(class_metrics['total_classes'], relation_metrics['total_relations'],
                                  class_metrics['stereotyped_classes'], relation_metrics['stereotyped_relations'],
                                  class_metrics['non_stereotyped_classes'],
                                  relation_metrics['non_stereotyped_relations'], class_metrics['ontouml_classes'],
                                  relation_metrics['ontouml_relations'], class_metrics['non_ontouml_classes'],
                                  relation_metrics['non_ontouml_relations'])

        # Step 5: Store the results in the statistics dictionary
        self.statistics.update(class_metrics)
        self.statistics.update(relation_metrics)
        self.statistics.update(ratios)

        # Step 6: Store all calculated statistics in self.statistics
        for key, stat_dict in metrics.items():
            for stat_name, value in stat_dict.items():
                self.statistics[f'{key}_{stat_name}'] = value

        self.num_classes = self.statistics["total_classes"]
        self.num_relations = self.statistics["total_relations"]
        assert self.num_classes != -1  # value attributed
        assert self.num_relations != -1  # value attributed

        logger.success(f"Statistics calculated for dataset '{self.name}'.")

    def _create_dataframe_for_stereotypes(self, stereotype_type: str) -> pd.DataFrame:
        """Helper function to create a DataFrame from class or relation stereotypes."""
        if stereotype_type not in ['class_stereotypes', 'relation_stereotypes']:
            raise ValueError("Invalid stereotype_type. Must be 'class_stereotypes' or 'relation_stereotypes'.")

        data = []
        for model in self.models:
            if stereotype_type == "class_stereotypes":
                data.append(model.class_stereotypes)
            else:
                data.append(model.relation_stereotypes)

        df = pd.DataFrame(data)
        df.insert(0, 'model', [model.name for model in self.models])  # Insert model names as the first column
        return df

    def calculate_models_statistics(self) -> None:
        """
        Calculate the statistics for all models in the dataset.
        """

        # Ensure statistics are calculated for each model
        for model in self.models:
            model.calculate_statistics()

    def save_models_statistics_to_csv(self, output_csv_dir: str) -> None:
        """
        Save statistics from a list of models to a CSV file dynamically.
        """
        # Use a list to preserve insertion order and avoid duplicates
        all_keys = []

        # Collect all the unique statistics keys from the models in this dataset
        for model in self.models:
            # Add logging to check if model.statistics exists and is a dictionary
            if isinstance(model.statistics, dict):
                all_keys = append_unique_preserving_order(all_keys, model.statistics.keys())
            else:
                logger.error(f"Model '{model.name}' does not have a valid statistics dictionary.")

        # Define the output directory
        output_dir = os.path.join(output_csv_dir, self.name)

        # Create the folder if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{self.name}_models_statistics.csv")

        # Open the CSV file for writing
        with open(output_path, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write the header (model, dynamic keys)
            writer.writerow(['model'] + all_keys)

            # Write the statistics for each model
            for model in self.models:
                row = [model.name]  # Start the row with the model name

                # Check for each key, retrieve the value, handle NaN if applicable
                for key in all_keys:
                    value = model.statistics.get(key, 'N/A')  # Get the value or default to 'N/A'

                    # Check if the value is a number and if it's NaN
                    if isinstance(value, (int, float)) and math.isnan(value):
                        row.append('N/A')
                    else:
                        row.append(value)

                # Write the row to the CSV file
                writer.writerow(row)

        logger.success(f"Statistics for models in dataset '{self.name}' successfully saved in {output_path}.")

    def calculate_stereotype_statistics(self) -> None:
        """
        Calculate stereotype statistics for class and relation stereotypes, both raw and clean,
        and store the results in the corresponding dictionaries.
        """
        # Step 1: Calculate raw statistics (without cleaning 'none' and 'other') for class and relation stereotypes
        self.class_statistics_raw = calculate_stereotype_metrics(self.models, 'class', filter_type=False)
        self.relation_statistics_raw = calculate_stereotype_metrics(self.models, 'relation', filter_type=False)
        self.combined_statistics_raw = calculate_stereotype_metrics(self.models, 'combined', filter_type=False)

        # Step 2: Calculate clean statistics (with filtering 'none' and 'other') for class and relation stereotypes
        self.class_statistics_clean = calculate_stereotype_metrics(self.models, 'class', filter_type=True)
        self.relation_statistics_clean = calculate_stereotype_metrics(self.models, 'relation', filter_type=True)
        self.combined_statistics_clean = calculate_stereotype_metrics(self.models, 'combined', filter_type=True)

        logger.success(f"Stereotype statistics calculated for dataset '{self.name}'.")

    def save_stereotype_statistics(self, output_dir: str) -> None:
        """
        Save all stereotype statistics (class/relation, raw/clean) to separate CSV files in different folders.
        :param output_dir: Directory where the CSV files will be saved.
        """
        # Define subdirectories for class/relation and raw/clean data
        subdirs = {'class_raw': self.class_statistics_raw, 'relation_raw': self.relation_statistics_raw,
                   'class_clean': self.class_statistics_clean, 'relation_clean': self.relation_statistics_clean,
                   'combined_raw': self.combined_statistics_raw, 'combined_clean': self.combined_statistics_clean}

        # Create the output directories and save the statistics
        for subdir, statistics in subdirs.items():
            # Create the specific folder (e.g., class_raw, relation_raw, etc.)
            output_subdir = os.path.join(output_dir, self.name, subdir)

            os.makedirs(output_subdir, exist_ok=True)

            # Save the statistics to CSV files
            for stat_name, dataframe in statistics.items():
                stat_name_cleaned = stat_name.lower().replace(" ", "_")
                filepath = os.path.join(output_subdir, f"{stat_name_cleaned}.csv")
                save_to_csv(dataframe, filepath,
                            f"Dataset {self.name}, case '{subdir}', statistic '{stat_name}' saved successfully in '{filepath}'.")

    def classify_and_save_spearman_correlation(self, output_dir: str) -> None:
        """
        Classify the Spearman correlations for both class and relation stereotypes (raw and clean)
        and save the results in a CSV file.
        """
        # Define subdirectories for class/relation and raw/clean data
        subdirs = {'class_raw': self.class_statistics_raw, 'relation_raw': self.relation_statistics_raw,
                   'class_clean': self.class_statistics_clean, 'relation_clean': self.relation_statistics_clean,
                   'combined_raw': self.combined_statistics_raw, 'combined_clean': self.combined_statistics_clean}

        # Iterate through the statistics to classify and save the Spearman correlation
        for subdir, statistics in subdirs.items():
            output_subdir = os.path.join(output_dir, self.name, subdir)
            os.makedirs(output_dir, exist_ok=True)

            correlations = ['spearman_correlation_occurrence_wise', 'spearman_correlation_model_wise']
            # Access the Spearman correlation result
            for correlation in correlations:
                spearman_correlation = statistics[correlation]

                # Generate file path for saving
                filepath = os.path.join(output_subdir, f'{correlation}_classified.csv')

                # Call the classification and save function
                classify_and_save_spearman_correlations(spearman_correlation, filepath)
                logger.success(
                    f"Dataset {self.name}, case '{subdir}', {correlation} classified and saved successfully in '{filepath}'.")

    def classify_and_save_total_correlation(self, output_dir: str) -> None:
        """
        Classifies and saves the total correlations for class/relation (raw/clean)
        stereotypes into two separate CSV files: one for occurrence-wise and one for model-wise.

        :param output_dir: Directory where the CSV files will be saved.
        """
        # Define subdirectories for class/relation and raw/clean data
        subdirs = {'class_raw': self.class_statistics_raw, 'relation_raw': self.relation_statistics_raw,
                   'class_clean': self.class_statistics_clean, 'relation_clean': self.relation_statistics_clean,
                   'combined_raw': self.combined_statistics_raw, 'combined_clean': self.combined_statistics_clean}

        for subdir, statistics in subdirs.items():
            # Extract the Spearman correlation data for occurrence-wise and model-wise
            spearman_occurrence = statistics['spearman_correlation_occurrence_wise'].set_index('Stereotype')
            spearman_model = statistics['spearman_correlation_model_wise'].set_index('Stereotype')

            # Step 1: Calculate total correlations and ranks for occurrence-wise
            total_corr_occurrence = spearman_occurrence.abs().sum(axis=1) - 1  # Subtract 1 to exclude self-correlation
            total_corr_occurrence_rank = total_corr_occurrence.rank(ascending=False, method='min')

            # Step 2: Create a DataFrame for occurrence-wise results
            total_corr_occurrence_df = pd.DataFrame(
                {'stereotype': total_corr_occurrence.index, 'total_correlation': total_corr_occurrence.values,
                 'rank': total_corr_occurrence_rank.astype(int).values})

            # Step 3: Calculate total correlations and ranks for model-wise
            total_corr_model = spearman_model.abs().sum(axis=1) - 1  # Subtract 1 to exclude self-correlation
            total_corr_model_rank = total_corr_model.rank(ascending=False, method='min')

            # Step 4: Create a DataFrame for model-wise results
            total_corr_model_df = pd.DataFrame(
                {'stereotype': total_corr_model.index, 'total_correlation': total_corr_model.values,
                 'rank': total_corr_model_rank.astype(int).values})

            # Step 5: Define the output directories and file paths
            output_subdir = os.path.join(output_dir, self.name, subdir)
            os.makedirs(output_dir, exist_ok=True)

            occurrence_filepath = os.path.join(output_subdir, 'spearman_correlation_total_occurrence_wise.csv')
            model_filepath = os.path.join(output_subdir, 'spearman_correlation_total_model_wise.csv')

            # Step 6: Save the occurrence-wise and model-wise DataFrames to separate CSV files
            total_corr_occurrence_df.to_csv(occurrence_filepath, index=False)
            logger.success(
                f"Dataset '{self.name}', case '{subdir}': total occurrence-wise correlation saved successfully in '{occurrence_filepath}'.")

            total_corr_model_df.to_csv(model_filepath, index=False)
            logger.success(
                f"Dataset '{self.name}', case '{subdir}': total model-wise correlation saved successfully in '{model_filepath}'.")

    # Existing code...

    def classify_and_save_geometric_mean_correlation(self, output_dir: str) -> None:
        # Define subdirectories for class/relation and raw/clean data
        subdirs = {'class_raw': self.class_statistics_raw, 'relation_raw': self.relation_statistics_raw,
                   'class_clean': self.class_statistics_clean, 'relation_clean': self.relation_statistics_clean,
                   'combined_raw': self.combined_statistics_raw, 'combined_clean': self.combined_statistics_clean}

        for subdir, statistics in subdirs.items():
            # Extract the Spearman correlation data for occurrence-wise and model-wise
            spearman_occurrence = statistics['spearman_correlation_occurrence_wise'].set_index('Stereotype')
            spearman_model = statistics['spearman_correlation_model_wise'].set_index('Stereotype')

            # Step 1: Find common stereotypes between occurrence-wise and model-wise correlations
            common_stereotypes = spearman_occurrence.index.intersection(spearman_model.index)

            # Step 2: Filter both correlation data to keep only the common stereotypes
            spearman_occurrence_common = spearman_occurrence.loc[common_stereotypes]
            spearman_model_common = spearman_model.loc[common_stereotypes]

            # Step 3: Calculate geometric mean for each stereotype (row-wise)
            geometric_mean_values = []
            for stereotype in common_stereotypes:
                occurrence_sum = spearman_occurrence_common.loc[
                                     stereotype].abs().sum() - 1  # Subtract 1 to exclude self-correlation
                model_sum = spearman_model_common.loc[
                                stereotype].abs().sum() - 1  # Subtract 1 to exclude self-correlation

                product = occurrence_sum * model_sum

                # Check if the product is non-negative before calculating the square root
                if product >= 0:
                    geometric_mean = math.sqrt(product)
                else:
                    logger.error(f"Negative product encountered for stereotype '{stereotype}': {product}")
                    exit(1)
                    geometric_mean = float('nan')

                geometric_mean_values.append(geometric_mean)

            # Step 4: Create a DataFrame for the geometric mean results
            geometric_mean_df = pd.DataFrame(
                {'stereotype': common_stereotypes, 'total_correlation': geometric_mean_values})

            # Step 5: Calculate rank based on geometric mean (descending order)
            geometric_mean_df['rank'] = geometric_mean_df['total_correlation'].rank(ascending=False,
                                                                                    method='min').astype(int)

            # Step 6: Define the output directory and file path
            output_subdir = os.path.join(output_dir, self.name, subdir)
            os.makedirs(output_dir, exist_ok=True)

            filepath = os.path.join(output_subdir, 'spearman_correlation_total_geometric_mean.csv')

            # Step 7: Save the DataFrame to a CSV file
            geometric_mean_df.to_csv(filepath, index=False)
            logger.success(
                f"Geometric mean correlation data for dataset '{self.name}', case '{subdir}' saved successfully in '{filepath}'.")

    def classify_and_save_geometric_mean_pairwise_correlation(self, output_dir: str) -> None:
        """
        Calculates and saves the geometric mean of the correlation values for each pair of stereotypes.
        This is done for stereotypes that appear in both occurrence-wise and model-wise correlations.
        Saves the result as a matrix where rows and columns are the stereotypes.

        :param output_dir: Directory where the CSV file will be saved.
        """
        # Define subdirectories for class/relation and raw/clean data
        subdirs = {'class_raw': self.class_statistics_raw, 'relation_raw': self.relation_statistics_raw,
                   'class_clean': self.class_statistics_clean, 'relation_clean': self.relation_statistics_clean,
                   'combined_raw': self.combined_statistics_raw, 'combined_clean': self.combined_statistics_clean}

        for subdir, statistics in subdirs.items():
            # Extract the Spearman correlation data for occurrence-wise and model-wise
            spearman_occurrence = statistics['spearman_correlation_occurrence_wise'].set_index('Stereotype')
            spearman_model = statistics['spearman_correlation_model_wise'].set_index('Stereotype')

            # Step 1: Find common stereotypes between occurrence-wise and model-wise correlations
            common_stereotypes = spearman_occurrence.columns.intersection(spearman_model.columns)

            # Step 2: Filter both correlation data to keep only the common stereotypes (rows and columns)
            spearman_occurrence_common = spearman_occurrence.loc[common_stereotypes, common_stereotypes]
            spearman_model_common = spearman_model.loc[common_stereotypes, common_stereotypes]

            # Step 3: Initialize a DataFrame for storing the geometric mean correlations
            geometric_mean_matrix = pd.DataFrame(index=common_stereotypes, columns=common_stereotypes)

            # Step 4: Calculate the geometric mean for each pair of stereotypes and fill the matrix
            for i, stereotype1 in enumerate(common_stereotypes):
                for j, stereotype2 in enumerate(common_stereotypes):
                    if i > j:
                        # Calculate the geometric mean for the upper triangle (i > j)
                        occ_corr = spearman_occurrence_common.loc[stereotype1, stereotype2]
                        model_corr = spearman_model_common.loc[stereotype1, stereotype2]
                        geometric_mean = math.sqrt(abs(occ_corr) * abs(model_corr))

                        # Set both (i, j) and (j, i) to the geometric mean value in the symmetric matrix
                        geometric_mean_matrix.loc[stereotype1, stereotype2] = geometric_mean
                        geometric_mean_matrix.loc[stereotype2, stereotype1] = geometric_mean
                    elif i == j:
                        # Set diagonal elements to 1 (since they represent self-correlation)
                        geometric_mean_matrix.loc[stereotype1, stereotype2] = 1.0

            # Step 5: Define the output directory and file path
            output_subdir = os.path.join(output_dir, self.name, subdir)
            os.makedirs(output_subdir, exist_ok=True)

            filepath = os.path.join(output_subdir, 'spearman_correlation_geometric_mean.csv')

            # Step 6: Save the matrix DataFrame to a CSV file
            geometric_mean_matrix.to_csv(filepath, index_label='Stereotype')
            logger.success(
                f"Geometric mean pairwise correlation matrix for dataset '{self.name}', case '{subdir}' saved successfully in '{filepath}'.")

            # Step 7: Generating classified results
            classified_filepath = os.path.join(output_subdir, 'spearman_correlation_geometric_mean_classified.csv')

            # Reset index and rename the first column to 'Stereotype'
            geometric_mean_matrix_reset = geometric_mean_matrix.reset_index().rename(columns={'index': 'Stereotype'})

            # Call the classification function
            classify_and_save_spearman_correlations(geometric_mean_matrix_reset, classified_filepath)

    def calculate_and_save_quadrants(self, output_dir: str, stats_access: str, x_metric: str, y_metric: str) -> None:
        """
        Calls the calculate_quadrants_and_save function for class and relation raw/clean cases.

        :param output_dir: Directory where the results will be saved.
        :param x_metric: The metric for the x-axis.
        :param y_metric: The metric for the y-axis.
        """
        # Define the four cases: class raw, relation raw, class clean, relation clean
        subdirs = {'class_raw': self.class_statistics_raw, 'relation_raw': self.relation_statistics_raw,
                   'class_clean': self.class_statistics_clean, 'relation_clean': self.relation_statistics_clean,
                   'combined_raw': self.combined_statistics_raw, 'combined_clean': self.combined_statistics_clean}

        for case, statistics in subdirs.items():
            # Check if the statistics for the current case contain the required metrics
            if x_metric in statistics[stats_access].columns and y_metric in statistics[stats_access].columns:
                # Create a DataFrame from the two columns (x_metric and y_metric)
                df = pd.DataFrame(
                    {x_metric: statistics[stats_access][x_metric], y_metric: statistics[stats_access][y_metric]})
                # Define the subdirectory for this case
                case_output_dir = os.path.join(output_dir, self.name, case)
                os.makedirs(case_output_dir, exist_ok=True)

                # Call the calculate_quadrants_and_save function for this case without index_col
                calculate_quadrants_and_save(df, x_metric, y_metric, case_output_dir)
                logger.success(f"Quadrants calculated and saved for '{case}' case in '{case_output_dir}'.")
            else:
                raise ValueError("Metrics not found in dataset's statistics.")

    def calculate_and_save_average_model(self, output_dir: str) -> None:
        """
        Calculate the 'average' model for the dataset in terms of its stereotypes.
        The average model will have an average number of classes and relations, with the class and relation stereotypes
        allocated proportionally based on the dataset's statistics.
        The result is saved to a CSV file for each case (class/relation raw/clean).

        :param output_dir: Directory where the CSV file will be saved.
        """
        # Define subdirectories for class/relation and raw/clean data
        subdirs = {'class_raw': self.class_statistics_raw, 'relation_raw': self.relation_statistics_raw,
                   'class_clean': self.class_statistics_clean, 'relation_clean': self.relation_statistics_clean,
                   'combined_raw': self.combined_statistics_raw, 'combined_clean': self.combined_statistics_clean}

        # Iterate through each case (class/relation and raw/clean)
        for case, statistics in subdirs.items():
            # Step 1: Gather the stereotype data for classes and relations
            if "class_" in case:
                stereotypes = [model.class_stereotypes for model in self.models]
            else:
                stereotypes = [model.relation_stereotypes for model in self.models]

            # Convert to DataFrame for easy manipulation
            df = pd.DataFrame(stereotypes)

            if "_clean" in case:
                df = df.drop(columns=["other", "none"])

            # Step 2: Calculate average total number of classes or relations per model
            if "_clean" in case:
                avg_total = int(round(df.sum(axis=1).mean()))
            else:
                avg_total = round(self.num_classes / self.num_models) if "class_" in case else round(
                    self.num_relations / self.num_models)

            # Step 3: Calculate the average number of each stereotype per model
            avg_stereotypes = df.mean().values

            # Step 4: Normalize the stereotype counts to ensure they sum up to the total average number of classes/relations
            avg_stereotypes = np.floor(avg_stereotypes / avg_stereotypes.sum() * avg_total).astype(int)

            # Step 5: Adjust for rounding issues
            difference = avg_total - avg_stereotypes.sum()

            if difference > 0:
                # Increase the count of the most common stereotype by the difference
                for _ in range(difference):
                    avg_stereotypes[np.argmax(avg_stereotypes)] += 1
            elif difference < 0:
                # Decrease the count of the most common stereotype by the difference
                for _ in range(-difference):
                    avg_stereotypes[np.argmax(avg_stereotypes)] -= 1

            # Step 6: Prepare the data for saving to CSV
            stereotype_labels = df.columns
            avg_model_data = {'Stereotype': stereotype_labels, 'average_quantity': avg_stereotypes}

            # Create DataFrame
            avg_model_df = pd.DataFrame(avg_model_data)

            # Add a rank column, with the highest quantity receiving the lowest rank
            avg_model_df['rank'] = avg_model_df['average_quantity'].rank(ascending=False, method='min').astype(int)

            # Add a percentage column showing the proportion relative to the average number of classes or relations
            avg_model_df['proportion'] = avg_model_df['average_quantity'] / avg_total

            # Step 7: Define output directory, including the case name
            case_output_dir = os.path.join(output_dir, self.name, case)

            # Create folder if it does not exist
            os.makedirs(case_output_dir, exist_ok=True)

            # Step 8: Save the average model data to a CSV file
            avg_model_filepath = os.path.join(case_output_dir, f'average_model.csv')
            avg_model_df.to_csv(avg_model_filepath, index=False)
            logger.success(
                f"Average model for dataset '{self.name}', case '{case}' successfully saved to {avg_model_filepath}.")

            # Step 9: Assert to check that the sum of the stereotypes' rounded average matches the expected value
            expected_avg_total = avg_total

            # Check if the sum of avg_stereotypes matches the expected average
            assert avg_stereotypes.sum() == expected_avg_total, f"Sum of rounded average stereotypes {avg_stereotypes.sum()} does not match expected average {expected_avg_total} for case {case}."

    def calculate_and_save_stereotypes_by_year(self, output_dir: str) -> pd.DataFrame:
        # Create dictionaries to hold the sum of stereotypes per year for class and relation (occurrence-wise and model-wise)
        yearly_data_class_ow = {}
        yearly_data_relation_ow = {}
        yearly_data_class_mw = {}
        yearly_data_relation_mw = {}

        cases = {'class': (self.data_class, yearly_data_class_ow, yearly_data_class_mw),
                 'relation': (self.data_relation, yearly_data_relation_ow, yearly_data_relation_mw)}

        # Loop through each model and accumulate the stereotype counts by year
        for analysis, (content, yearly_data_ow, yearly_data_mw) in cases.items():
            for model in self.models:
                model_year = model.year

                if model_year not in yearly_data_ow:
                    yearly_data_ow[model_year] = np.zeros(len(content.columns) - 1,
                                                          dtype=int)  # Initialize with zeros for occurrence-wise calculation
                if model_year not in yearly_data_mw:
                    yearly_data_mw[model_year] = np.zeros(len(content.columns) - 1,
                                                          dtype=int)  # Initialize with zeros for model-wise calculation

                # Fetch the row for the current model and ensure it matches the expected length
                model_data = content.loc[content['model'] == model.name].iloc[0, 1:].astype(int).values

                if len(model_data) == len(yearly_data_ow[model_year]):
                    # Occurrence-wise: Sum the stereotype counts for each year
                    yearly_data_ow[model_year] += model_data

                    # Model-wise: Check where a stereotype occurs (binary approach)
                    yearly_data_mw[model_year] += (model_data > 0).astype(int)
                else:
                    raise ValueError(f"Mismatch in number of columns for model {model.name} in {analysis} analysis.")

            # Convert the dictionaries to DataFrames (both occurrence-wise and model-wise)
            stereotypes = content.columns[1:]  # Get the stereotype names
            df_yearly_ow = pd.DataFrame.from_dict(yearly_data_ow, orient='index', columns=stereotypes)
            df_yearly_mw = pd.DataFrame.from_dict(yearly_data_mw, orient='index', columns=stereotypes)

            # Set the 'year' as the index
            df_yearly_ow.index.name = 'year'
            df_yearly_mw.index.name = 'year'

            # Sort the DataFrames by the 'year' index in ascending order
            df_yearly_ow = df_yearly_ow.sort_index(ascending=True)
            df_yearly_mw = df_yearly_mw.sort_index(ascending=True)

            # Store the occurrence-wise and model-wise results in years_stereotypes_data
            self.years_stereotypes_data[f'{analysis}_ow'] = df_yearly_ow
            self.years_stereotypes_data[f'{analysis}_mw'] = df_yearly_mw

            # Normalize both occurrence-wise and model-wise results
            self._normalize_stereotypes_overall(f'{analysis}_ow')
            self._normalize_stereotypes_yearly(f'{analysis}_ow')
            self._normalize_stereotypes_overall(f'{analysis}_mw')
            self._normalize_stereotypes_yearly(f'{analysis}_mw')

        # Define the keys for saving the data
        keys = ['class_ow', 'relation_ow', 'class_mw', 'relation_mw', 'class_ow_overall', 'relation_ow_overall',
                'class_ow_yearly', 'relation_ow_yearly', 'class_mw_overall', 'relation_mw_overall', 'class_mw_yearly',
                'relation_mw_yearly']

        # Save each of the DataFrames to CSV files
        for key in keys:
            # Create the correct folder structure for class and relation
            if 'class' in key:
                output_dir_final = os.path.join(output_dir, self.name, "class_raw")
            elif 'relation' in key:
                output_dir_final = os.path.join(output_dir, self.name, "relation_raw")
            else:
                raise ValueError("Unexpected key in years_stereotypes_data.")

            # Create the directory if it does not exist
            os.makedirs(output_dir_final, exist_ok=True)

            # Save the DataFrame to a CSV file
            csv_path = os.path.join(output_dir_final, f'years_stereotypes_{key}.csv')
            self.years_stereotypes_data[key].to_csv(csv_path)
            logger.success(f"{key} stereotypes data saved to {csv_path}.")

    def calculate_and_save_models_by_year(self, output_dir: str):
        # Initialize dictionaries to count the number of models per year
        model_count = {}

        # Loop through self.models to count models for each year based on whether they have class or relation data
        for model in self.models:
            model_year = model.year

            if model_year not in model_count:
                model_count[model_year] = 0
            model_count[model_year] += 1  # Assuming all models have at least one stereotype of any type

        # Convert the dictionaries into DataFrames
        df_model_count = pd.DataFrame(list(model_count.items()), columns=['year', 'num_models'])

        # Calculate the total number of models
        total_models = df_model_count['num_models'].sum()

        # Calculate the ratio
        df_model_count['ratio'] = df_model_count['num_models'] / total_models

        # Sort by year to ensure chronological order
        df_model_count = df_model_count.sort_values(by='year').reset_index(drop=True)

        # Store the results in self.years_models_number
        self.years_models_number = df_model_count

        # Create folder if it does not exist
        output_dir_final = os.path.join(output_dir, self.name)
        os.makedirs(output_dir_final, exist_ok=True)

        # Save models per year data
        models_csv_path = os.path.join(output_dir_final, 'years_models_number.csv')
        self.years_models_number.to_csv(models_csv_path, index=False)
        logger.success(f"Models per year data saved to {models_csv_path}.")

    def _normalize_stereotypes_overall(self, case) -> None:
        df = self.years_stereotypes_data[case]

        # Sum of all values in the DataFrame (excluding the 'year' index)
        total_sum = df.to_numpy().sum()

        # Normalize the DataFrame so that the sum of all values is 1
        df_normalized = df / total_sum

        # Store the result in years_stereotypes_data for overall normalization
        self.years_stereotypes_data[f'{case}_overall'] = df_normalized

    def _normalize_stereotypes_yearly(self, case) -> None:
        # Normalize for both 'class' and 'relation'
        df = self.years_stereotypes_data[case]

        # Normalize each row so that the sum of values in each row is 1
        df_normalized = df.div(df.sum(axis=1), axis=0)

        # Store the result in years_stereotypes_data for yearly normalization
        self.years_stereotypes_data[f'{case}_yearly'] = df_normalized

    def save_stereotypes_count_by_year(self, output_dir: str) -> None:
        """
        Save a CSV file that reports the number of class and relation stereotypes for each year, including ratio, cumulative,
        ontouml, none, and other class/relation columns, and their respective ratios and cumulative values.
        """
        # Initialize a dictionary to hold the stereotype counts per year
        year_data = {}

        # Loop through each model and accumulate class and relation stereotypes by year
        for model in self.models:
            model_year = model.year

            # Initialize the year entry if not present
            if model_year not in year_data:
                year_data[model_year] = {'num_class': 0, 'ontouml_class': 0, 'none_class': 0, 'other_class': 0,
                                         'num_relation': 0, 'ontouml_relation': 0, 'none_relation': 0,
                                         'other_relation': 0}

            # Process class stereotypes
            for stereotype, count in model.class_stereotypes.items():
                year_data[model_year]['num_class'] += count
                if stereotype == 'none':
                    year_data[model_year]['none_class'] += count
                elif stereotype == 'other':
                    year_data[model_year]['other_class'] += count
                else:
                    year_data[model_year]['ontouml_class'] += count

            # Process relation stereotypes
            for stereotype, count in model.relation_stereotypes.items():
                year_data[model_year]['num_relation'] += count
                if stereotype == 'none':
                    year_data[model_year]['none_relation'] += count
                elif stereotype == 'other':
                    year_data[model_year]['other_relation'] += count
                else:
                    year_data[model_year]['ontouml_relation'] += count

        # Convert the dictionary to a DataFrame for easier CSV saving
        df_year_data = pd.DataFrame.from_dict(year_data, orient='index').reset_index()
        df_year_data.columns = ['year', 'num_class', 'ontouml_class', 'none_class', 'other_class', 'num_relation',
                                'ontouml_relation', 'none_relation', 'other_relation']

        # Sort by year to ensure chronological order
        df_year_data = df_year_data.sort_values(by='year').reset_index(drop=True)

        # Calculate the totals for classes and relations
        total_classes = df_year_data['num_class'].sum()
        total_relations = df_year_data['num_relation'].sum()

        total_ontouml_classes = df_year_data['ontouml_class'].sum()
        total_none_classes = df_year_data['none_class'].sum()
        total_other_classes = df_year_data['other_class'].sum()

        total_ontouml_relations = df_year_data['ontouml_relation'].sum()
        total_none_relations = df_year_data['none_relation'].sum()
        total_other_relations = df_year_data['other_relation'].sum()

        # Ratio and cumulative columns for classes
        df_year_data['ratio_class'] = df_year_data['num_class'] / total_classes
        df_year_data['cumulative_class'] = df_year_data['num_class'].cumsum()
        df_year_data['cumulative_ratio_class'] = df_year_data['cumulative_class'] / total_classes

        # Ratio and cumulative columns for ontouml, none, other classes
        df_year_data['ratio_ontouml_class'] = df_year_data['ontouml_class'] / total_ontouml_classes

        df_year_data['cumulative_ontouml_class'] = df_year_data['ontouml_class'].cumsum()
        df_year_data['cumulative_ratio_ontouml_class'] = df_year_data[
                                                             'cumulative_ontouml_class'] / total_ontouml_classes

        df_year_data['ratio_none_class'] = df_year_data['none_class'] / total_none_classes
        df_year_data['cumulative_none_class'] = df_year_data['none_class'].cumsum()
        df_year_data['cumulative_ratio_none_class'] = df_year_data['cumulative_none_class'] / total_none_classes

        df_year_data['ratio_other_class'] = df_year_data['other_class'] / total_other_classes
        df_year_data['cumulative_other_class'] = df_year_data['other_class'].cumsum()
        df_year_data['cumulative_ratio_other_class'] = df_year_data['cumulative_other_class'] / total_other_classes

        # Ratio and cumulative columns for relations
        df_year_data['ratio_relation'] = df_year_data['num_relation'] / total_relations
        df_year_data['cumulative_relation'] = df_year_data['num_relation'].cumsum()
        df_year_data['cumulative_ratio_relation'] = df_year_data['cumulative_relation'] / total_relations

        # Ratio and cumulative columns for ontouml, none, other relations
        df_year_data['ratio_ontouml_relation'] = df_year_data['ontouml_relation'] / total_ontouml_relations
        df_year_data['cumulative_ontouml_relation'] = df_year_data['ontouml_relation'].cumsum()
        df_year_data['cumulative_ratio_ontouml_relation'] = df_year_data[
                                                                'cumulative_ontouml_relation'] / total_ontouml_relations

        df_year_data['ratio_none_relation'] = df_year_data['none_relation'] / total_none_relations
        df_year_data['cumulative_none_relation'] = df_year_data['none_relation'].cumsum()
        df_year_data['cumulative_ratio_none_relation'] = df_year_data['cumulative_none_relation'] / total_none_relations

        df_year_data['ratio_other_relation'] = df_year_data['other_relation'] / total_other_relations
        df_year_data['cumulative_other_relation'] = df_year_data['other_relation'].cumsum()
        df_year_data['cumulative_ratio_other_relation'] = df_year_data[
                                                              'cumulative_other_relation'] / total_other_relations

        # Ratios related to general category

        df_year_data['ratio_ontouml_to_total_class'] = df_year_data['ontouml_class'] / total_classes
        df_year_data['ratio_none_to_total_class'] = df_year_data['none_class'] / total_classes
        df_year_data['ratio_other_to_total_class'] = df_year_data['other_class'] / total_classes

        df_year_data['ratio_ontouml_to_total_relation'] = df_year_data['ontouml_relation'] / total_relations
        df_year_data['ratio_none_to_total_relation'] = df_year_data['none_relation'] / total_relations
        df_year_data['ratio_other_to_total_relation'] = df_year_data['other_relation'] / total_relations

        df_year_data['cumulative_ratio_ontouml_to_total_class'] = df_year_data[
                                                                      'cumulative_ontouml_class'] / total_classes
        df_year_data['cumulative_ratio_none_to_total_class'] = df_year_data['cumulative_none_class'] / total_classes
        df_year_data['cumulative_ratio_other_to_total_class'] = df_year_data['cumulative_other_class'] / total_classes

        df_year_data['cumulative_ratio_ontouml_to_total_relation'] = df_year_data[
                                                                         'cumulative_ontouml_relation'] / total_relations
        df_year_data['cumulative_ratio_none_to_total_relation'] = df_year_data[
                                                                      'cumulative_none_relation'] / total_relations
        df_year_data['cumulative_ratio_other_to_total_relation'] = df_year_data[
                                                                       'cumulative_other_relation'] / total_relations

        # Define output directory and create it if necessary
        output_dir_final = os.path.join(output_dir, self.name)
        os.makedirs(output_dir_final, exist_ok=True)

        # Save the DataFrame to a CSV file
        csv_path = os.path.join(output_dir_final, 'stereotypes_count_by_year.csv')
        df_year_data.to_csv(csv_path, index=False)

        logger.success(f"Stereotypes count by year data saved to {csv_path}.")
