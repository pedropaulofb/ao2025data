import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger


def generate_non_ontouml_visualization(in_dir_path, out_dir_path, file_path):
    # Load your data
    df = pd.read_csv(os.path.join(in_dir_path, file_path))

    # Filter the data for the columns you want ('year', 'none', 'other') and make a deep copy to avoid SettingWithCopyWarning
    if 'year' not in df.columns or 'none' not in df.columns or 'other' not in df.columns:
        logger.error("Missing 'year', 'none', or 'other' columns in the input data.")
        return

    df_filtered = df[['year', 'none', 'other']].copy()  # Explicitly make a copy

    # Convert 'none' and 'other' to percentages
    df_filtered['none'] = df_filtered['none'] * 100
    df_filtered['other'] = df_filtered['other'] * 100

    # Convert 'year' to string using astype
    df_filtered['year'] = df_filtered['year'].astype(str)

    # Sort by year just in case it's not sorted
    df_filtered = df_filtered.sort_values('year')

    # Calculate the 5-year moving average for 'none' and 'other'
    df_filtered['none_moving_avg'] = df_filtered['none'].rolling(window=5, min_periods=1).mean()
    df_filtered['other_moving_avg'] = df_filtered['other'].rolling(window=5, min_periods=1).mean()

    # Reshape the DataFrame for Seaborn plotting
    df_melted = df_filtered.melt(id_vars='year', value_vars=['none', 'other'], var_name='Element',
                                 value_name='Frequency')

    # Create the plot
    plt.figure(figsize=(16, 9))  # Adjusted figure size for better layout
    sns.set(style="whitegrid")

    # Define specific colors for 'none' and 'other'
    colors = {'none': 'blue', 'other': 'red'}

    # Barplot with bars side by side (dodge=True ensures side-by-side bars)
    ax = sns.barplot(x='year', y='Frequency', hue='Element', data=df_melted, dodge=True, alpha=0.7, palette=colors)

    # Plot 'none' moving average with the same blue color and label for the legend
    sns.lineplot(x='year', y='none_moving_avg', data=df_filtered, color='blue', linewidth=2.5,
                 label='none (5-year avg)')

    # Plot 'other' moving average with the same red color and label for the legend
    sns.lineplot(x='year', y='other_moving_avg', data=df_filtered, color='red', linewidth=2.5,
                 label='other (5-year avg)')

    # Improve readability by rotating x-axis labels and reducing the number of ticks
    plt.xticks(rotation=45, ha='right')

    # Adjust number of x-ticks for large datasets
    if len(df_filtered['year'].unique()) > 20:  # Arbitrary limit, adjust as needed
        ax.set_xticks(ax.get_xticks()[::2])  # Show every second year

    # Determine if the file is model-wise or occurrence-wise and yearly or overall normalized
    if 'modelwise' in file_path:
        frequency_type = 'Model-wise'
    else:
        frequency_type = 'Occurrence-wise'

    if 'yearly' in file_path:
        normalization_type = 'Yearly-normalized'
    else:
        normalization_type = 'Overall-normalized'

    # Build the title
    title = f"{frequency_type} {normalization_type} Data of 'none' and 'other'"

    # Generate a figure name from the file_path (remove extension and append .png)
    fig_name = os.path.splitext(file_path)[0] + "_visualization.png"

    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Frequency (%)')  # If the values represent percentages; otherwise, remove (%)
    plt.title(title, fontweight='bold')

    # Ensure output directory exists
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    # Save the figure
    plt.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")

    # Close the plot to free memory
    plt.close()
