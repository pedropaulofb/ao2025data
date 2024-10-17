import pandas as pd
from icecream import ic
from loguru import logger


def compare_and_generate_quadrant_csv(output_file_path, q_before, q_after, q_general=None):
    # Read the input CSV files

    df_before = pd.read_csv(q_before)
    df_after = pd.read_csv(q_after)
    df_general = pd.read_csv(q_general) if q_general is not None else None

    # Check if all files have the 'Stereotype' and 'quadrant' columns
    required_columns = ['Stereotype', 'quadrant']
    for df, name in zip([df_before, df_after, df_general], ['q_before', 'q_after', 'q_general']):
        if df is not None:
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns in {name}. Expected columns: {required_columns}")
                return

    # Set index as 'Stereotype' for easier comparison
    df_before.set_index('Stereotype', inplace=True)
    df_after.set_index('Stereotype', inplace=True)
    if df_general is not None:
        df_general.set_index('Stereotype', inplace=True)

    # Get the common stereotypes across the datasets
    if df_general is not None:
        common_stereotypes = df_before.index.intersection(df_after.index).intersection(df_general.index)
        all_stereotypes = df_before.index.union(df_after.index).union(df_general.index)
    else:
        common_stereotypes = df_before.index.intersection(df_after.index)
        all_stereotypes = df_before.index.union(df_after.index)

    # Log warnings for Stereotypes that are missing in any of the datasets
    missing_stereotypes = all_stereotypes.difference(common_stereotypes)
    for missing_st in missing_stereotypes:
        logger.warning(f"Stereotype '{missing_st}' is not present in all CSV input files.")

    # Prepare the output DataFrame
    output_df = pd.DataFrame(index=common_stereotypes)
    output_df['quadrant_before'] = df_before.loc[common_stereotypes, 'quadrant']
    output_df['quadrant_after'] = df_after.loc[common_stereotypes, 'quadrant']

    if df_general is not None:
        output_df['quadrant_general'] = df_general.loc[common_stereotypes, 'quadrant']
    else:
        output_df['quadrant_general'] = None  # Leave the column empty if no general data is provided

    # Reset index to move Stereotype back to a column
    output_df.reset_index(inplace=True)

    # Save the output DataFrame to a CSV file
    output_df.to_csv(output_file_path, index=False)
    logger.success(f"Combined quadrant CSV file '{output_file_path}' successfully generated.")
