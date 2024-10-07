import csv

from icecream import ic

from src.ModelData import ModelData


def instantiate_models_from_csv(input_models_data_csv_path: str, input_number_stereotypes_csv_path: str) -> list[
    ModelData]:
    models_list = []

    # Step 1: Read the from input_number_class_relation_csv to map model_id to count_class and count_relation
    class_relation_data = {}
    with open(input_number_stereotypes_csv_path, mode='r', newline='') as file2:
        reader2 = csv.DictReader(file2)
        for row in reader2:
            model_id = row["model_id"]
            count_class = int(row["count_class"])
            count_relation = int(row["count_relation"])
            class_relation_data[model_id] = {"count_class": count_class, "count_relation": count_relation}

    # Step 2: Read from input_models_data_csv and instantiate ModelData objects, merging the data from both files
    with open(input_models_data_csv_path, mode='r', newline='') as file1:
        reader1 = csv.DictReader(file1)
        for row in reader1:
            name = row["model"]
            year = int(row["year"])
            is_classroom = row["is_classroom"] == "True"  # Convert string to boolean

            # Get total_class and total_relation from the second file using the model_id
            total_class = class_relation_data[name]["count_class"]
            total_relation = class_relation_data[name]["count_relation"]

            # Instantiate ModelData with merged data
            model_data = ModelData(name, year, is_classroom, total_class, total_relation)
            models_list.append(model_data)

    return models_list


