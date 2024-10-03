import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger


def generate_non_ontouml_visualization(in_dir_path, out_dir_path, file_path, window_size=5):
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

    # Calculate the moving average with the specified window size
    df_filtered['none_moving_avg'] = df_filtered['none'].rolling(window=window_size, min_periods=1).mean()
    df_filtered['other_moving_avg'] = df_filtered['other'].rolling(window=window_size, min_periods=1).mean()

    # Reshape the DataFrame for Seaborn plotting
    df_filtered.melt(id_vars='year', value_vars=['none', 'other'], var_name='Element', value_name='Frequency')

    # Define specific hex colors for 'none' and 'other' bars
    bar_colors = {'none': '#d62728', 'other': '#1f77b4'}  # Red for 'none' and blue for 'other' bars

    # Define specific hex colors for 'none' and 'other' lines (darker shades for higher contrast)
    line_colors = {'none': '#8c1e1e', 'other': '#0d3f6b'}  # Darker red for 'none' and darker blue for 'other' lines

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(16, 9))  # Adjusted figure size for better layout
    sns.set(style="whitegrid")

    # Plot 'none' and 'other' bars with manual color application
    years = df_filtered['year']

    # Manually plot the 'none' bars
    ax1.bar(years, df_filtered['none'], color=bar_colors['none'], alpha=0.7, label='none (occurrence-wise)', width=0.4,
            align='center', edgecolor='none')

    # Manually plot the 'other' bars
    ax1.bar(years, df_filtered['other'], color=bar_colors['other'], alpha=0.7, label='other (occurrence-wise)',
            width=0.4, align='edge', edgecolor='none')

    # Plot 'none' moving average with different color for the line
    sns.lineplot(x='year', y='none_moving_avg', data=df_filtered, ax=ax1, color=line_colors['none'], linewidth=2.5,
                 label=f'none ({window_size}-year avg)')

    # Plot 'other' moving average with different color for the line
    sns.lineplot(x='year', y='other_moving_avg', data=df_filtered, ax=ax1, color=line_colors['other'], linewidth=2.5,
                 label=f'other ({window_size}-year avg)')

    # Improve readability by rotating x-axis labels and reducing the number of ticks
    plt.xticks(rotation=45, ha='right')

    # Determine if the file is model-wise or occurrence-wise and yearly or overall normalized
    if 'modelwise' in file_path:
        frequency_type = 'Model-wise'
    else:
        frequency_type = 'Occurrence-wise'

    if 'yearly' in file_path:
        normalization_type = 'Yearly-normalized'
    else:
        normalization_type = 'Overall-normalized'

    # Build the title with window size for moving average
    title = f"{frequency_type} {normalization_type} Data of 'none' and 'other' ({window_size}-year avg)"

    # Generate a figure name from the file_path (remove extension and append .png)
    fig_name = os.path.splitext(file_path)[0] + f"_visualization_{window_size}_year_avg.png"

    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Frequency (%)')  # If the values represent percentages; otherwise, remove (%)
    plt.title(title, fontweight='bold')

    # Ensure both vertical and horizontal gridlines are displayed
    ax1.grid(True, which='both', axis='both')  # Ensure both x and y gridlines are shown
    # Move gridlines behind the bars
    ax1.set_axisbelow(True)

    # Ensure output directory exists
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    # Save the figure
    plt.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")

    # Close the plot to free memory
    plt.close()

