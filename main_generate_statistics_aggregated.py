import pandas as pd
import os

from icecream import ic
from loguru import logger

# Define the groups of constructs
AGGREGATION_GROUPS = {
    "sortal": ["kind", "phase", "collective", "mode", "quantity", "relator", "subkind", "role", "historicalRole",
               "type", "quality"],
    "non_sortal": ["category", "mixin", "phaseMixin", "roleMixin", "historicalRoleMixin"],
    "rigid": ["category", "kind", "collective", "mode", "quantity", "relator", "subkind", "quality"],
    "non_rigid": ["mixin", "phaseMixin", "roleMixin", "phase", "role", "historicalRole", "historicalRoleMixin"],
    "abstracts": ["abstract", "datatype", "enumeration"],
    "temporal": ["event", "situation", "historicalRoleMixin", "historicalRole"],
    "other": ["other"],
    "none": ["none"],
    "undef": ["other", "none"]
}

AGGREGATION_GROUPS["rigid_sortal"] = list(set(AGGREGATION_GROUPS["rigid"]) & set(AGGREGATION_GROUPS["sortal"]))
AGGREGATION_GROUPS["rigid_non_sortal"] = list(set(AGGREGATION_GROUPS["rigid"]) & set(AGGREGATION_GROUPS["non_sortal"]))
AGGREGATION_GROUPS["non_rigid_sortal"] = list(set(AGGREGATION_GROUPS["non_rigid"]) & set(AGGREGATION_GROUPS["sortal"]))
AGGREGATION_GROUPS["non_rigid_non_sortal"] = list(set(AGGREGATION_GROUPS["non_rigid"]) & set(AGGREGATION_GROUPS["non_sortal"]))

# Function to calculate occurrences for each group
def calculate_group_occurrences(df, groups):
    # Create a new DataFrame to store the results
    result_df = pd.DataFrame(df['model'])

    # Iterate through each group in 'groups' dictionary
    for group_name, elements in groups.items():
        # For each group, sum up the occurrences of elements present in that group
        result_df[group_name] = df[elements].sum(axis=1)

    return result_df

if __name__ == "__main__":
    # Load the input CSV file
    input_dir = './outputs/consolidated_data'

    # Check if input directory exists
    if not os.path.exists(input_dir):
        logger.error(f"Input directory '{input_dir}' does not exist.")
    else:
        # Get all consolidated data file names
        input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        for input_file in input_files:
            analysis = input_file.replace(".csv", "")
            if input_file.startswith("cs_"):
                input_file_path = os.path.normpath(os.path.join(input_dir, input_file))

                logger.info(f"Processing file: {input_file_path}")
                try:
                    df = pd.read_csv(input_file_path)
                except Exception as e:
                    logger.error(f"Error reading file {input_file_path}: {e}")
                    continue

                # Calculate the group occurrences
                result_df = calculate_group_occurrences(df, AGGREGATION_GROUPS)

                # Save the result to a CSV file
                output_file_path = os.path.join(input_dir, "aggregated", input_file)
                result_df.to_csv(output_file_path, index=False)
                logger.success(f"Aggregated data successfully saved in {output_file_path}.")
            else:
                logger.info(f"Skipping {analysis}.")
