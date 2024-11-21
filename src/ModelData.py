import csv

from src.calculations.statistics_calculations_datasets import calculate_ratios


class ModelData:
    def __init__(self, name: str, year: int, is_classroom: bool, total_class: int, total_relation: int) -> None:

        self.name: str = name
        self.year: int = year
        self.is_classroom: bool = is_classroom
        self.total_class_number: int = total_class
        self.total_relation_number: int = total_relation
        self.statistics: dict = {}

        # List of attributes stored in the 'stats' dictionary
        self.class_stereotypes: list[str] = ["abstract", "category", "collective", "datatype", "enumeration", "event",
                                             "historicalRole", "historicalRoleMixin", "kind", "mixin", "mode", "phase",
                                             "phaseMixin", "quality", "quantity", "relator", "role", "roleMixin",
                                             "situation", "subkind", "type", "none", "other"]

        # List of attributes stored in the 'stats' dictionary
        self.relation_stereotypes: list[str] = ["bringsAbout", "characterization", "comparative", "componentOf",
                                                "creation", "derivation", "externalDependence", "formal",
                                                "historicalDependence", "instantiation", "manifestation", "material",
                                                "mediation", "memberOf", "participation", "participational",
                                                "subCollectionOf", "subQuantityOf", "termination", "triggers", "none",
                                                "other"]

        # Initialize 'stats' dictionary with default value 0 for all attributes
        self.class_stereotypes: dict[str, int] = {attr: 0 for attr in self.class_stereotypes}
        self.relation_stereotypes: dict[str, int] = {attr: 0 for attr in self.relation_stereotypes}

        # Create a mapping from normalized keys to original keys
        self.normalized_class_stereotypes = {
            attr.strip().lower().replace("-", "").replace("_", "").replace(" ", ""): attr
            for attr in [
                "abstract", "category", "collective", "datatype", "enumeration", "event",
                "historicalRole", "historicalRoleMixin", "kind", "mixin", "mode", "phase",
                "phaseMixin", "quality", "quantity", "relator", "role", "roleMixin",
                "situation", "subkind", "type", "none", "other"
            ]
        }

        self.normalized_relation_stereotypes = {
            attr.strip().lower().replace("-", "").replace("_", "").replace(" ", ""): attr
            for attr in [
                "bringsAbout", "characterization", "comparative", "componentOf", "creation",
                "derivation", "externalDependence", "formal", "historicalDependence", "instantiation",
                "manifestation", "material", "mediation", "memberOf", "participation",
                "participational", "subCollectionOf", "subQuantityOf", "termination", "triggers", "none",
                "other"
            ]
        }

        # Separate dictionaries for invalid stereotypes and their counts
        self.invalid_class_stereotypes: dict[str, int] = {}
        self.invalid_relation_stereotypes: dict[str, int] = {}

    def count_stereotypes(self, stereotype_type: str, input_csv_path: str) -> None:
        """
        Count the stereotypes for the current model instance based on the provided CSV file.
        Tracks invalid stereotypes in separate dictionaries for class and relation stereotypes.
        """

        # Choose the appropriate stereotypes dictionary and normalization map
        if stereotype_type == 'class':
            stereotype_dict = self.class_stereotypes
            normalization_map = self.normalized_class_stereotypes
            invalid_dict = self.invalid_class_stereotypes
        elif stereotype_type == 'relation':
            stereotype_dict = self.relation_stereotypes
            normalization_map = self.normalized_relation_stereotypes
            invalid_dict = self.invalid_relation_stereotypes
        else:
            raise ValueError("Invalid stereotype_type. Must be 'class' or 'relation'.")

        # Read the CSV file containing stereotypes and counts
        with open(input_csv_path, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                model_id = row["model_id"]
                stereotype = row["stereotype"]
                count = int(row["count"])

                # Process the row only if the model_id matches the current instance's name
                if model_id == self.name:
                    # Normalize the stereotype from the CSV
                    normalized_stereotype = (
                        stereotype.strip()
                        .lower()
                        .replace("-", "")
                        .replace("_", "")
                        .replace(" ", "")
                    )

                    # Check if the normalized stereotype is valid
                    if normalized_stereotype in normalization_map:
                        # Use the normalization map to find the corresponding original stereotype
                        original_stereotype = normalization_map[normalized_stereotype]
                        stereotype_dict[original_stereotype] += count
                    else:
                        # Add or update the count for invalid stereotypes in the appropriate dictionary
                        if normalized_stereotype in invalid_dict:
                            invalid_dict[normalized_stereotype] += count
                        else:
                            # Add to list of invalid and also to 'other' count.
                            invalid_dict[normalized_stereotype] = count
                            stereotype_dict["other"] += count

    def calculate_none(self) -> None:
        """
        Calculate 'none' for both class and relation stereotypes.
        """
        # Calculate 'none' for class stereotypes
        total_class_counted = sum(value for key, value in self.class_stereotypes.items() if key != "none")
        self.class_stereotypes["none"] = self.total_class_number - total_class_counted

        # Calculate 'none' for relation stereotypes
        total_relation_counted = sum(value for key, value in self.relation_stereotypes.items() if key != "none")
        self.relation_stereotypes["none"] = self.total_relation_number - total_relation_counted

    def calculate_statistics(self) -> None:
        """
        Calculate statistics based on the class and relation stereotypes and store them in self.statistics.
        """
        # Calculate total and stereotyped/non-stereotyped classes and relations
        total_classes = self.total_class_number
        total_relations = self.total_relation_number

        stereotyped_classes = total_classes - self.class_stereotypes["none"]
        non_stereotyped_classes = self.class_stereotypes["none"]

        stereotyped_relations = total_relations - self.relation_stereotypes["none"]
        non_stereotyped_relations = self.relation_stereotypes["none"]

        # OntoUML stereotypes exclude 'none' and 'other'
        ontouml_classes = total_classes - self.class_stereotypes["none"] - self.class_stereotypes["other"]
        ontouml_relations = total_relations - self.relation_stereotypes["none"] - self.relation_stereotypes["other"]

        non_ontouml_classes = self.class_stereotypes["none"] + self.class_stereotypes["other"]
        non_ontouml_relations = self.relation_stereotypes["none"] + self.relation_stereotypes["other"]

        # Step 1: Calculate ratios
        ratios = calculate_ratios(total_classes, total_relations, stereotyped_classes, stereotyped_relations,
            non_stereotyped_classes, non_stereotyped_relations, ontouml_classes, ontouml_relations, non_ontouml_classes,
            non_ontouml_relations)

        # Step 2: Count unique class and relation stereotypes (excluding "none" and "other")
        unique_class_stereotypes = sum(
            1 for key, value in self.class_stereotypes.items() if value > 0 and key not in ["none", "other"])
        unique_relation_stereotypes = sum(
            1 for key, value in self.relation_stereotypes.items() if value > 0 and key not in ["none", "other"])

        # Step 3: Store statistics in self.statistics
        self.statistics = {"total_classes": total_classes, "stereotyped_classes": stereotyped_classes,
            "non_stereotyped_classes": non_stereotyped_classes, "ontouml_classes": ontouml_classes,
            "non_ontouml_classes": non_ontouml_classes, "total_relations": total_relations,
            "stereotyped_relations": stereotyped_relations, "non_stereotyped_relations": non_stereotyped_relations,
            "ontouml_relations": ontouml_relations, "non_ontouml_relations": non_ontouml_relations,
            "unique_class_stereotypes": unique_class_stereotypes,
            "unique_relation_stereotypes": unique_relation_stereotypes}

        # Add the ratios to the statistics
        self.statistics.update(ratios)
