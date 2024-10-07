import csv
import os

import pandas as pd
from loguru import logger

from src import ModelData
from src.dataset_calculations import calculate_class_and_relation_metrics, calculate_stats, calculate_ratios


class Dataset():
    def __init__(self, name: str, models:list[ModelData]) -> None:

        self.name: str = name
        self.models: list[ModelData] = models
        self.statistics = {}

    def count_models(self) -> int:
        """Return the number of models in the dataset."""
        return len(self.models)

    def generate_dataset_general_data_csv(self,output_dir:str) -> None:
        with open(os.path.join(output_dir,f'{self.name}.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(["model", "year", "total_class_number", "total_relation_number"])
            # Write the data for each model
            for model in self.models:
                writer.writerow([model.name, model.year, model.total_class_number, model.total_relation_number])

    def generate_dataset_class_data_csv(self,output_dir:str) -> None:
        with open(os.path.join(output_dir,f'{self.name}_class.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            # Extract all the stereotypes from the first model (assuming all models have the same stereotypes)
            stereotypes = list(self.models[0].class_stereotypes.keys())
            # Write the header
            writer.writerow(["model"] + stereotypes)
            # Write the data for each model
            for model in self.models:
                row = [model.name] + [model.class_stereotypes[st] for st in stereotypes]
                writer.writerow(row)

    def generate_dataset_relation_data_csv(self,output_dir:str) -> None:
        with open(os.path.join(output_dir,f'{self.name}_relation.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            # Extract all the stereotypes from the first model's relation_stereotypes
            stereotypes = list(self.models[0].relation_stereotypes.keys())
            # Write the header
            writer.writerow(["model"] + stereotypes)
            # Write the data for each model
            for model in self.models:
                row = [model.name] + [model.relation_stereotypes[st] for st in stereotypes]
                writer.writerow(row)

    def calculate_statistics(self) -> None:
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

        # Step 4: Calculate ratios
        ratios = calculate_ratios(
            class_metrics['total_classes'], relation_metrics['total_relations'],
            class_metrics['stereotyped_classes'], relation_metrics['stereotyped_relations'],
            class_metrics['non_stereotyped_classes'], relation_metrics['non_stereotyped_relations'],
            class_metrics['ontouml_classes'], relation_metrics['ontouml_relations'],
            class_metrics['non_ontouml_classes'], relation_metrics['non_ontouml_relations']
        )

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