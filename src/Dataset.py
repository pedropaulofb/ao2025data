import copy
import csv
import math
import os
from copy import deepcopy

import numpy as np
import pandas as pd
from icecream import ic
from loguru import logger

from src import ModelData
from src.statistics_calculations_datasets import calculate_class_and_relation_metrics, calculate_stats, calculate_ratios
from src.statistics_calculations_stereotypes import calculate_stereotype_metrics
from src.statistics_calculations_stereotypes_extra import classify_and_save_spearman_correlations, \
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
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

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
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

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
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

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
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

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

    def identify_outliers(self) -> dict:
        """
        Identify outliers in the dataset based on IQR for 'total_classes', 'total_relations', and the class-to-relation ratio.

        :return: Dictionary of outlier models with the reason ('class', 'relation', 'ratio', or a combination of these).
        """
        # Retrieve the necessary statistics from the dataset's statistics dictionary
        Q1_classes = self.statistics['class_total_q1']
        Q3_classes = self.statistics['class_total_q3']
        IQR_classes = self.statistics['class_total_iqr']

        Q1_relations = self.statistics['relation_total_q1']
        Q3_relations = self.statistics['relation_total_q3']
        IQR_relations = self.statistics['relation_total_iqr']

        # Calculate the class-to-relation ratio for each model
        ratios = [model.total_class_number / model.total_relation_number if model.total_relation_number > 0 else np.nan
                  for model in self.models]
        df = pd.DataFrame({'model': [model.name for model in self.models],
                           'total_classes': [model.total_class_number for model in self.models],
                           'total_relations': [model.total_relation_number for model in self.models],
                           'class_relation_ratio': ratios})

        # Step 1: Define the bounds for non-outliers using the IQR method for total_classes and total_relations
        lower_bound_classes = Q1_classes - 1.5 * IQR_classes
        upper_bound_classes = Q3_classes + 1.5 * IQR_classes

        lower_bound_relations = Q1_relations - 1.5 * IQR_relations
        upper_bound_relations = Q3_relations + 1.5 * IQR_relations

        # # Step 2: Calculate IQR for the class-to-relation ratio - DISCARDED
        # Q1_ratio = df['class_relation_ratio'].quantile(0.25)
        # Q3_ratio = df['class_relation_ratio'].quantile(0.75)
        # IQR_ratio = Q3_ratio - Q1_ratio
        #
        # lower_bound_ratio = Q1_ratio - 1.5 * IQR_ratio
        # upper_bound_ratio = Q3_ratio + 1.5 * IQR_ratio

        # Step 3: Identify outliers
        outliers = {}

        for index, model in df.iterrows():
            class_outlier = (
                    model['total_classes'] < lower_bound_classes or model['total_classes'] > upper_bound_classes)
            relation_outlier = (model['total_relations'] < lower_bound_relations or model[
                'total_relations'] > upper_bound_relations)
            # ratio_outlier = (model['class_relation_ratio'] < lower_bound_ratio or
            #                  model['class_relation_ratio'] > upper_bound_ratio)

            # Determine if it's a class outlier, relation outlier, ratio outlier, or a combination
            # if class_outlier and relation_outlier and ratio_outlier:
            #     outliers[model['model']] = 'class, relation, ratio;'
            if class_outlier and relation_outlier:
                outliers[model['model']] = 'class, relation;'
            # elif class_outlier and ratio_outlier:
            #     outliers[model['model']] = 'class, ratio;'
            # elif relation_outlier and ratio_outlier:
            #     outliers[model['model']] = 'relation, ratio;'
            elif class_outlier:
                outliers[model['model']] = 'class;'
            elif relation_outlier:
                outliers[model['model']] = 'relation;'  # elif ratio_outlier:  #     outliers[model['model']] = 'ratio;'

        # Step 4: Log the results
        if not outliers:
            logger.success(f"No outliers found in dataset '{self.name}'. All models are within the normal range.")
        else:
            logger.warning(f"Outliers found in dataset '{self.name}': {outliers}")

        return list(outliers.keys())

    def fork_without_outliers(self, outliers: list[str]) -> 'Dataset':
        """
        Create a new Dataset instance by removing models that are in the outliers list.

        :param outliers: List of model names that are outliers.
        :return: A new Dataset instance without the outliers.
        """
        # Create a deep copy of the current dataset
        new_dataset = copy.deepcopy(self)

        new_dataset.name = self.name + "_filtered"

        # Remove models whose names are in the outliers list
        new_dataset.models = [model for model in new_dataset.models if model.name not in outliers]

        new_dataset.reset_statistics()

        return new_dataset

    def reset_statistics(self) -> None:
        """
        Reset the statistics for the current Dataset instance by clearing the statistics dictionary.
        """
        self.statistics = {}
        for model in self.models:
            model.statistics = {}  # Reset statistics for each model as well
        logger.info(f"Statistics reset for dataset '{self.name}' and all its models.")

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

            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

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
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

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
            total_corr_occurrence_df = pd.DataFrame({'stereotype': total_corr_occurrence.index,
                                                     'total_correlation': total_corr_occurrence.values,
                                                     'rank': total_corr_occurrence_rank.astype(
                                                         int).values})

            # Step 3: Calculate total correlations and ranks for model-wise
            total_corr_model = spearman_model.abs().sum(axis=1) - 1  # Subtract 1 to exclude self-correlation
            total_corr_model_rank = total_corr_model.rank(ascending=False, method='min')

            # Step 4: Create a DataFrame for model-wise results
            total_corr_model_df = pd.DataFrame(
                {'stereotype': total_corr_model.index, 'total_correlation': total_corr_model.values,
                 'rank': total_corr_model_rank.astype(int).values})

            # Step 5: Define the output directories and file paths
            output_subdir = os.path.join(output_dir, self.name, subdir)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            occurrence_filepath = os.path.join(output_subdir, 'spearman_correlation_total_occurrence_wise.csv')
            model_filepath = os.path.join(output_subdir, 'spearman_correlation_total_model_wise.csv')

            # Step 6: Save the occurrence-wise and model-wise DataFrames to separate CSV files
            total_corr_occurrence_df.to_csv(occurrence_filepath, index=False)
            logger.success(
                f"Dataset '{self.name}', case '{subdir}': total occurrence-wise correlation saved successfully in '{occurrence_filepath}'.")

            total_corr_model_df.to_csv(model_filepath, index=False)
            logger.success(
                f"Dataset '{self.name}', case '{subdir}': total model-wise correlation saved successfully in '{model_filepath}'.")

    def classify_and_save_geometric_mean_correlation(self, output_dir: str) -> None:
        """
        Classifies and saves the geometric mean of the total correlations (occurrence-wise and model-wise)
        for class/relation (raw/clean) stereotypes to a CSV file, with a rank column.

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

                # Calculate geometric mean for the current stereotype
                geometric_mean = math.sqrt(occurrence_sum * model_sum)
                geometric_mean_values.append(geometric_mean)

            # Step 4: Create a DataFrame for the geometric mean results
            geometric_mean_df = pd.DataFrame(
                {'stereotype': common_stereotypes, 'total_correlation': geometric_mean_values})

            # Step 5: Calculate rank based on geometric mean (descending order)
            geometric_mean_df['rank'] = geometric_mean_df['total_correlation'].rank(ascending=False,
                                                                                             method='min').astype(int)

            # Step 6: Define the output directory and file path
            output_subdir = os.path.join(output_dir, self.name, subdir)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

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
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

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
            if x_metric in statistics and y_metric in statistics:
                # Create a DataFrame from the two columns (x_metric and y_metric)
                df = pd.DataFrame({x_metric: statistics[x_metric], y_metric: statistics[y_metric]})

                # Define the subdirectory for this case
                case_output_dir = os.path.join(output_dir, self.name, case)
                if not os.path.exists(case_output_dir):
                    os.makedirs(case_output_dir)

                # Call the calculate_quadrants_and_save function for this case without index_col
                calculate_quadrants_and_save(df, x_metric, y_metric, case_output_dir)

                logger.success(f"Quadrants calculated and saved for '{case}' case in '{case_output_dir}'.")

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
            if not os.path.exists(case_output_dir):
                os.makedirs(case_output_dir)

            # Step 8: Save the average model data to a CSV file
            avg_model_filepath = os.path.join(case_output_dir, f'average_model.csv')
            avg_model_df.to_csv(avg_model_filepath, index=False)
            logger.success(
                f"Average model for dataset '{self.name}', case '{case}' successfully saved to {avg_model_filepath}.")

            # Step 9: Assert to check that the sum of the stereotypes' rounded average matches the expected value
            expected_avg_total = avg_total

            # Check if the sum of avg_stereotypes matches the expected average
            assert avg_stereotypes.sum() == expected_avg_total, f"Sum of rounded average stereotypes {avg_stereotypes.sum()} does not match expected average {expected_avg_total} for case {case}."

    def calculate_and_save_stereotypes_by_year(self,output_dir:str) -> pd.DataFrame:
        # Create dictionaries to hold the sum of stereotypes per year for class and relation
        yearly_data_class = {}
        yearly_data_relation = {}

        cases = {
            'class': (self.data_class, yearly_data_class),
            'relation': (self.data_relation, yearly_data_relation)
        }

        # Loop through each model and accumulate the stereotype counts by year
        for analysis, (content, yearly_data) in cases.items():
            for model in self.models:
                model_year = model.year

                if model_year not in yearly_data:
                    yearly_data[model_year] = np.zeros(len(content.columns) - 1,
                                                       dtype=int)  # Initialize with zeros for each stereotype, ensure dtype=int

                # Fetch the row for the current model and ensure it matches the expected length
                model_data = content.loc[content['model'] == model.name].iloc[0, 1:].astype(int).values

                if len(model_data) == len(yearly_data[model_year]):
                    # Sum the stereotype counts for each year (excluding the 'model' column)
                    yearly_data[model_year] += model_data
                else:
                    raise ValueError(f"Mismatch in number of columns for model {model.name} in {analysis} analysis.")

            # Convert the dictionary to a DataFrame
            stereotypes = content.columns[1:]  # Get the stereotype names
            df_yearly = pd.DataFrame.from_dict(yearly_data, orient='index', columns=stereotypes)

            # Add the 'year' as the index
            df_yearly.index.name = 'year'

            # Sort the DataFrame by the 'year' index in ascending order
            df_yearly = df_yearly.sort_index(ascending=True)

            # Store the result in years_stereotypes_data
            self.years_stereotypes_data[analysis] = df_yearly

            self._normalize_stereotypes_overall(analysis)
            self._normalize_stereotypes_yearly(analysis)



        keys = ['class','relation','class_overall','relation_overall','class_yearly','relation_yearly']

        for key in keys:

            # Create folder if it does not exist
            if 'class' in key:
                output_dir_final = os.path.join(output_dir, self.name, "class_raw")
            elif 'relation' in key:
                output_dir_final = os.path.join(output_dir, self.name, "relation_raw")
            else:
                raise ValueError

            if not os.path.exists(output_dir_final):
                os.makedirs(output_dir_final)

            csv_path = os.path.join(output_dir_final,f'years_stereotypes_{key}.csv')
            self.years_stereotypes_data[key].to_csv(csv_path)
            logger.success(f"Class stereotypes data saved to {csv_path}.")

    def calculate_and_save_models_by_year(self,output_dir:str):
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
        if not os.path.exists(output_dir_final):
            os.makedirs(output_dir_final)

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

    def _normalize_stereotypes_yearly(self,case) -> None:
        # Normalize for both 'class' and 'relation'
        df = self.years_stereotypes_data[case]

        # Normalize each row so that the sum of values in each row is 1
        df_normalized = df.div(df.sum(axis=1), axis=0)

        # Store the result in years_stereotypes_data for yearly normalization
        self.years_stereotypes_data[f'{case}_yearly'] = df_normalized
