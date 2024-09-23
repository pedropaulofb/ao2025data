import pandas as pd
import os

from loguru import logger

# Define the groups of constructs
groups = {
    "sortal": ["kind", "phase", "collective", "mode", "quantity", "relator", "subkind", "role", "historicalRole",
               "type", "quality"],
    "non_sortal": ["category", "mixin", "phaseMixin", "roleMixin", "historicalRoleMixin"],
    "rigid": ["category", "kind", "collective", "mode", "quantity", "relator", "subkind", "quality"],
    "non_rigid": ["mixin", "phaseMixin", "roleMixin", "phase", "role", "historicalRole", "historicalRoleMixin"],
    "abstracts": ["abstract", "datatype", "enumeration"],
    "temporal": ["event", "situation", "historicalRoleMixin", "historicalRole"],
    "other" : ["other"],
    "none" : ["none"],
    "undef": ["other", "none"]
}


# Utility function to save a DataFrame to a CSV file with error handling.
def save_to_csv(dataframe, filepath, message):
    try:
        dataframe.to_csv(filepath, index=False)
        logger.success(message)
    except Exception as e:
        logger.error(f"Failed to save {filepath}: {e}")


# Load data files
def load_data(file_path):
    return pd.read_csv(file_path)


# Function to aggregate statistics for each group
def calculate_group_statistics(df, group):
    group_data = df[df['Construct'].isin(group)]

    # Sum total frequencies
    total_frequency = group_data['Total Frequency'].sum()

    # Group frequency sum
    group_frequency = group_data['Group Frequency'].sum()

    # Ubiquity index (grouped average)
    ubiquity_index = group_data['Ubiquity Index (Group Frequency per Group)'].mean()

    # Global relative frequency (sum of relative frequencies)
    global_relative_frequency = group_data['Global Relative Frequency (Occurrence-wise)'].sum()

    # Group-wise relative frequency
    global_relative_frequency_groupwise = group_data['Global Relative Frequency (Group-wise)'].sum()

    # Return aggregated statistics
    return {
        'Total Frequency': total_frequency,
        'Group Frequency': group_frequency,
        'Ubiquity Index (Group Frequency per Group)': ubiquity_index,
        'Global Relative Frequency (Occurrence-wise)': global_relative_frequency,
        'Global Relative Frequency (Group-wise)': global_relative_frequency_groupwise
    }


# Main function to process all groups in each folder
def process_groups(input_folder, output_folder):
    # Load frequency analysis file
    frequency_analysis_file = os.path.join(input_folder, 'frequency_analysis.csv')

    if not os.path.exists(frequency_analysis_file):
        logger.warning(f"File {frequency_analysis_file} does not exist. Skipping folder.")
        return

    frequency_analysis_df = load_data(frequency_analysis_file)

    # Prepare a list to collect results
    results = []

    # Iterate over the groups and calculate statistics
    for group_name, group_constructs in groups.items():
        logger.info(f"Calculating statistics for group: {group_name}")
        group_stats = calculate_group_statistics(frequency_analysis_df, group_constructs)
        group_stats['Construct'] = group_name
        results.append(group_stats)

    # Convert results to DataFrame
    group_statistics_df = pd.DataFrame(results)

    # Reorder columns to have 'Construct' as the first column
    columns_order = ['Construct'] + [col for col in group_statistics_df.columns if col != 'Construct']
    group_statistics_df = group_statistics_df[columns_order]

    # Save to CSV
    output_file = os.path.join(output_folder, f'{os.path.basename(input_folder)}_aggr.csv')
    save_to_csv(group_statistics_df, output_file, f"Aggregated group statistics saved in {output_file}.")


if __name__ == "__main__":
    base_dir = './outputs/statistics'
    aggregated_dir = os.path.join(base_dir, 'aggregated')

    # Create the 'aggregated' folder if it does not exist
    if not os.path.exists(aggregated_dir):
        os.makedirs(aggregated_dir)

    # Iterate through all subfolders in the 'statistics' folder
    subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and f != 'aggregated']

    for subfolder in subfolders:
        # Calculating only for class stereotype (as they are the only grouped stereotypes)
        if subfolder.startswith("cs_"):
            input_folder = os.path.join(base_dir, subfolder)

            logger.info(f"Processing folder: {subfolder}")
            process_groups(input_folder, aggregated_dir)
