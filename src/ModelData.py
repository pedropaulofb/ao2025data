import csv


class ModelData:
    def __init__(self, name: str, year: int, is_classroom: bool, total_class:int, total_relation:int) -> None:

        self.name: str = name
        self.year: int = year
        self.is_classroom: bool = is_classroom
        self.total_class_number: int = total_class
        self.total_relation_number: int = total_relation


        # List of attributes stored in the 'stats' dictionary
        self.class_stereotypes: list[str] = ["abstract", "category", "collective", "datatype", "enumeration", "event",
                                       "historicalRole", "historicalRoleMixin", "kind", "mixin", "mode", "phase",
                                       "phaseMixin", "quality", "quantity", "relator", "role", "roleMixin", "situation",
                                       "subkind", "type", "none", "other"]

        # List of attributes stored in the 'stats' dictionary
        self.relation_stereotypes: list[str] = ["bringsAbout", "characterization", "comparative", "componentOf", "creation",
                                          "derivation", "externalDependence", "historicalDependence", "instantiation",
                                          "manifestation", "material", "mediation", "memberOf", "participation",
                                          "participational", "subCollectionOf", "subQuantityOf", "termination",
                                          "triggers", "none", "other"]

        # Initialize 'stats' dictionary with default value 0 for all attributes
        self.class_stereotypes: dict[str, int] = {attr: 0 for attr in self.class_stereotypes}
        self.relation_stereotypes: dict[str, int] = {attr: 0 for attr in self.relation_stereotypes}

    def count_stereotypes(self, stereotype_type: str, input_csv_path: str) -> None:
        """
        Count the stereotypes for the current model instance based on the provided CSV file.
        """

        # Validate the stereotype_type argument
        if stereotype_type not in ['class', 'relation']:
            raise ValueError("Invalid stereotype_type. Must be 'class' or 'relation'.")

        # Choose the appropriate stereotypes dictionary based on stereotype_type
        if stereotype_type == 'class':
            stereotype_dict = self.class_stereotypes
        elif stereotype_type == 'relation':
            stereotype_dict = self.relation_stereotypes

        # Read the CSV file containing stereotypes and counts
        with open(input_csv_path, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                model_id = row["model_id"]
                stereotype = row["stereotype"]
                count = int(row["count"])

                # Process the row only if the model_id matches the current instance's name
                if model_id == self.name:
                    # Check if the stereotype is one of the predefined ones
                    if stereotype in stereotype_dict:
                        # Update the corresponding stereotype count
                        stereotype_dict[stereotype] += count
                    else:
                        # If not in predefined stereotypes, add to 'other'
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