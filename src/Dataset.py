import copy
import csv
import math
import os

import numpy as np
import pandas as pd
from icecream import ic
from loguru import logger

from src import ModelData
from src.statistics_calculations import calculate_class_and_relation_metrics, calculate_stats, calculate_ratios
from src.utils import append_unique_preserving_order


class Dataset():
    def __init__(self, name: str, models: list[ModelData]) -> None:

        self.name: str = name
        self.models: list[ModelData] = models
        self.statistics = {}

    def count_models(self) -> int:
        """Return the number of models in the dataset."""
        return len(self.models)

    def save_dataset_general_data_csv(self, output_dir: str) -> None:

        output_dir = os.path.join(output_dir, self.name)

        # Create folder if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir, f'{self.name}_data.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(["model", "year", "total_class_number", "total_relation_number"])
            # Write the data for each model
            for model in self.models:
                writer.writerow([model.name, model.year, model.total_class_number, model.total_relation_number])

    def save_dataset_class_data_csv(self, output_dir: str) -> None:
        output_dir = os.path.join(output_dir, self.name)

        # Create folder if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir, f'{self.name}_class_data.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            # Extract all the stereotypes from the first model (assuming all models have the same stereotypes)
            stereotypes = list(self.models[0].class_stereotypes.keys())
            # Write the header
            writer.writerow(["model"] + stereotypes)
            # Write the data for each model
            for model in self.models:
                row = [model.name] + [model.class_stereotypes[st] for st in stereotypes]
                writer.writerow(row)

    def save_dataset_relation_data_csv(self, output_dir: str) -> None:
        output_dir = os.path.join(output_dir, self.name)

        # Create folder if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir, f'{self.name}_relation_data.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            # Extract all the stereotypes from the first model's relation_stereotypes
            stereotypes = list(self.models[0].relation_stereotypes.keys())
            # Write the header
            writer.writerow(["model"] + stereotypes)
            # Write the data for each model
            for model in self.models:
                row = [model.name] + [model.relation_stereotypes[st] for st in stereotypes]
                writer.writerow(row)

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

        logger.success(f"Statistics calculated for dataset '{self.name}' and stored in 'self.statistics'.")

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

        # Step 2: Calculate IQR for the class-to-relation ratio
        Q1_ratio = df['class_relation_ratio'].quantile(0.25)
        Q3_ratio = df['class_relation_ratio'].quantile(0.75)
        IQR_ratio = Q3_ratio - Q1_ratio

        lower_bound_ratio = Q1_ratio - 1.5 * IQR_ratio
        upper_bound_ratio = Q3_ratio + 1.5 * IQR_ratio

        # Step 3: Identify outliers
        outliers = {}

        for index, model in df.iterrows():
            class_outlier = model['total_classes'] < lower_bound_classes or model['total_classes'] > upper_bound_classes
            relation_outlier = model['total_relations'] < lower_bound_relations or model[
                'total_relations'] > upper_bound_relations
            ratio_outlier = model['class_relation_ratio'] < lower_bound_ratio or model[
                'class_relation_ratio'] > upper_bound_ratio

            # Determine if it's a class outlier, relation outlier, ratio outlier, or a combination
            if class_outlier and relation_outlier and ratio_outlier:
                outliers[model['model']] = 'class, relation, ratio;'
            elif class_outlier and relation_outlier:
                outliers[model['model']] = 'class, relation;'
            elif class_outlier and ratio_outlier:
                outliers[model['model']] = 'class, ratio;'
            elif relation_outlier and ratio_outlier:
                outliers[model['model']] = 'relation, ratio;'
            elif class_outlier:
                outliers[model['model']] = 'class;'
            elif relation_outlier:
                outliers[model['model']] = 'relation;'
            elif ratio_outlier:
                outliers[model['model']] = 'ratio;'

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
