import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from icecream import ic
from loguru import logger


def generate_non_ontouml_visualization(df, out_dir_path, file_name, year_start=None, year_end=None, window_size=5):
    # Reset index and convert 'year' to int
    df = df.reset_index()
    df['year'] = df['year'].astype(int)

    # If year_start and year_end are provided, filter the DataFrame to include only those years
    if year_start is not None:
        df = df[df['year'] >= year_start]
    if year_end is not None:
        df = df[df['year'] <= year_end]

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
    if 'mw' in file_name:
        frequency_type = 'Model-wise'
    else:
        frequency_type = 'Occurrence-wise'

    if 'yearly' in file_name:
        normalization_type = 'Yearly-normalized'
    else:
        normalization_type = 'Overall-normalized'

    # Build the title with window size for moving average
    title = f"{frequency_type} {normalization_type} Data of 'none' and 'other' ({window_size}-year avg)"

    # Generate a figure name from the file_name (remove extension and append .png)
    fig_name = f"non_ontouml_analysis_{file_name}_{window_size}_year_avg.png"

    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Frequency (%)')  # If the values represent percentages; otherwise, remove (%)
    plt.title(title, fontweight='bold')

    # Ensure both vertical and horizontal gridlines are displayed
    ax1.grid(True, which='both', axis='both')  # Ensure both x and y gridlines are shown
    # Move gridlines behind the bars
    ax1.set_axisbelow(True)

    # Ensure output directory exists
    os.makedirs(out_dir_path, exist_ok=True)


    # Save the figure
    plt.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")

    # Close the plot to free memory
    plt.close()

def generate_non_ontouml_combined_visualization(df_occurrence, df_modelwise, out_dir_path, file_name, year_start=None, year_end=None):
    df_occurrence = df_occurrence.reset_index()
    df_modelwise = df_modelwise.reset_index()

    # Convert 'year' to int for filtering
    df_occurrence['year'] = df_occurrence['year'].astype(int)
    df_modelwise['year'] = df_modelwise['year'].astype(int)

    # Filter based on year_start and year_end
    if year_start is not None:
        df_occurrence = df_occurrence[df_occurrence['year'] >= year_start]
        df_modelwise = df_modelwise[df_modelwise['year'] >= year_start]
    if year_end is not None:
        df_occurrence = df_occurrence[df_occurrence['year'] <= year_end]
        df_modelwise = df_modelwise[df_modelwise['year'] <= year_end]

    # Ensure both DataFrames have the necessary columns ('year', 'none', 'other')
    if 'year' not in df_occurrence.columns or 'none' not in df_occurrence.columns or 'other' not in df_occurrence.columns:
        logger.error("Missing 'year', 'none', or 'other' columns in the occurrence-wise data.")
        return

    if 'year' not in df_modelwise.columns or 'none' not in df_modelwise.columns or 'other' not in df_modelwise.columns:
        logger.error("Missing 'year', 'none', or 'other' columns in the model-wise data.")
        return

    # Convert 'none' and 'other' to percentages for both occurrence-wise and model-wise data
    df_occurrence['none'] = df_occurrence['none'] * 100
    df_occurrence['other'] = df_occurrence['other'] * 100

    df_modelwise['none'] = df_modelwise['none'] * 100
    df_modelwise['other'] = df_modelwise['other'] * 100

    # Sort both DataFrames by year (just in case)
    df_occurrence = df_occurrence.sort_values('year')
    df_modelwise = df_modelwise.sort_values('year')

    # Convert 'year' to string in both DataFrames
    df_occurrence['year'] = df_occurrence['year'].astype(str)
    df_modelwise['year'] = df_modelwise['year'].astype(str)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(16, 9))  # Adjusted figure size for better layout

    # Define specific hex colors for 'none' and 'other' bars
    bar_colors = {'none': '#d62728', 'other': '#1f77b4'}  # Red for 'none' and blue for 'other' bars

    # Define specific hex colors for 'none' and 'other' lines (darker shades for higher contrast)
    line_colors = {'none': '#8c1e1e', 'other': '#0d3f6b'}  # Darker red for 'none' and darker blue for 'other' lines

    # Barplot for occurrence-wise frequencies ('none' and 'other') with bars side by side
    ax1.bar(df_occurrence['year'], df_occurrence['none'], color=bar_colors['none'], alpha=0.7,
            label='none (occurrence-wise)', width=0.4, align='center')
    ax1.bar(df_occurrence['year'], df_occurrence['other'], color=bar_colors['other'], alpha=0.7,
            label='other (occurrence-wise)', width=0.4, align='edge')

    # Set labels for the first y-axis (occurrence-wise)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Occurrence-wise Frequency (%)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Updated code to prevent y-lim warnings for occurrence-wise
    max_occurrence = max(df_occurrence[['none', 'other']].max())
    if max_occurrence > 0:
        ax1.set_ylim(0, max_occurrence * 1.1)
    else:
        ax1.set_ylim(0, 1)  # Set a default range if max_occurrence is zero or very small

    # Move gridlines behind the bars
    ax1.set_axisbelow(True)

    # Create a second y-axis for model-wise frequencies
    ax2 = ax1.twinx()

    # Plot model-wise frequencies ('none' and 'other') as lines without adding extra legends
    sns.lineplot(x='year', y='none', data=df_modelwise, ax=ax2, color=line_colors['none'], linewidth=2.5, legend=False)
    sns.lineplot(x='year', y='other', data=df_modelwise, ax=ax2, color=line_colors['other'], linewidth=2.5,
                 legend=False)

    # Updated code to prevent y-lim warnings for model-wise
    max_modelwise = max(df_modelwise[['none', 'other']].max())
    if max_modelwise > 0:
        ax2.set_ylim(0, max_modelwise * 1.1)
    else:
        ax2.set_ylim(0, 1)  # Set a default range if max_modelwise is zero or very small

    # Set labels for the second y-axis (model-wise)
    ax2.set_ylabel('Model-wise Frequency (%)', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Align both y-axes to have their zero points match
    ax1.set_ylim(0, ax1.get_ylim()[1])  # Ensure both start at zero
    ax2.set_ylim(0, ax2.get_ylim()[1])

    # Improve readability by rotating x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Explicitly control the gridlines
    ax1.grid(axis='x', color='lightgray')  # Only vertical gridlines on the primary axis (x-axis)
    ax2.grid(False)  # Disable gridlines on the secondary axis (y-axis)

    # Build the title
    title = f"Combined Occurrence-wise and Model-wise Data for 'none' and 'other'"

    # Generate a figure name from the file_name (remove extension and append .png)
    fig_name = f"non_ontouml_combined_visualization_{file_name}.png"

    # Add title
    plt.title(title, fontweight='bold')

    # Plot model-wise frequencies ('none' and 'other') as lines without adding extra legends
    line_none, = ax2.plot(df_modelwise['year'], df_modelwise['none'], color=line_colors['none'], linewidth=2.5,
                          label='none (model-wise)')
    line_other, = ax2.plot(df_modelwise['year'], df_modelwise['other'], color=line_colors['other'], linewidth=2.5,
                           label='other (model-wise)')

    # Combine legends for both occurrence-wise and model-wise data
    handles1, labels1 = ax1.get_legend_handles_labels()  # Handles for the bars (occurrence-wise)
    handles2 = [line_none, line_other]  # Manually get handles for the lines (model-wise)

    # Display the combined legend in the top left corner
    ax1.legend(handles1 + handles2, labels1 + ['none (model-wise)', 'other (model-wise)'], loc='upper right')

    # Save the figure
    plt.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")

    # Close the plot to free memory
    plt.close()



# TODO: Not tested. Test before using in future papers.
def generate_temporal_visualization(df, out_dir_path, file_name, stereotypes, year_start=None, year_end=None, window_size=5):
    # Reset index and convert 'year' to int
    df = df.reset_index()
    df['year'] = df['year'].astype(int)

    # If year_start and year_end are provided, filter the DataFrame to include only those years
    if year_start is not None:
        df = df[df['year'] >= year_start]
    if year_end is not None:
        df = df[df['year'] <= year_end]

    # Check if 'year' and the stereotypes exist in the DataFrame
    missing_columns = [col for col in ['year'] + stereotypes if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing columns in the input data: {missing_columns}")
        return

    # Filter the data for 'year' and stereotypes, and make a deep copy
    columns_to_keep = ['year'] + stereotypes
    df_filtered = df[columns_to_keep].copy()

    # Convert the stereotypes to percentages
    for term in stereotypes:
        df_filtered[term] = df_filtered[term] * 100

    # Convert 'year' to string
    df_filtered['year'] = df_filtered['year'].astype(str)

    # Sort by year just in case it's not sorted
    df_filtered = df_filtered.sort_values('year')

    # Calculate the moving average for each term
    for term in stereotypes:
        df_filtered[f'{term}_moving_avg'] = df_filtered[term].rolling(window=window_size, min_periods=1).mean()

    # Reshape the DataFrame for Seaborn plotting
    df_melted = df_filtered.melt(id_vars='year', value_vars=stereotypes, var_name='Element', value_name='Frequency')

    # Define specific hex colors for the bars and lines dynamically
    bar_colors = sns.color_palette("Set1", len(stereotypes))
    line_colors = sns.color_palette("dark", len(stereotypes))

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(16, 9))
    sns.set(style="whitegrid")

    years = df_filtered['year']

    # Manually plot the bars and lines for each term
    for idx, term in enumerate(stereotypes):
        ax1.bar(years, df_filtered[term], color=bar_colors[idx], alpha=0.7, label=f'{term} (occurrence-wise)',
                width=0.4, align='center', edgecolor='none')

        sns.lineplot(x='year', y=f'{term}_moving_avg', data=df_filtered, ax=ax1, color=line_colors[idx], linewidth=2.5,
                     label=f'{term} ({window_size}-year avg)')

    # Improve readability by rotating x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Determine if the file is model-wise or occurrence-wise and yearly or overall normalized
    if 'mw' in file_name:
        frequency_type = 'Model-wise'
    else:
        frequency_type = 'Occurrence-wise'

    if 'yearly' in file_name:
        normalization_type = 'Yearly-normalized'
    else:
        normalization_type = 'Overall-normalized'

    # Build the title with window size for moving average
    title = f"{frequency_type} {normalization_type} Data of {', '.join(stereotypes)} ({window_size}-year avg)"

    # Generate a figure name from the file_name
    fig_name = f"non_ontouml_analysis_{file_name}_{window_size}_year_avg.png"

    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Frequency (%)')
    plt.title(title, fontweight='bold')

    # Ensure gridlines are displayed
    ax1.grid(True, which='both', axis='both')
    ax1.set_axisbelow(True)

    # Ensure output directory exists
    os.makedirs(out_dir_path, exist_ok=True)


    # Save the figure
    plt.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")

    # Close the plot to free memory
    plt.close()

# TODO: Not tested. Test before using in future papers.
def generate_temporal_combined_visualization(df_occurrence, df_modelwise, out_dir_path, file_name, stereotypes,
                                                year_start=None, year_end=None):
    df_occurrence = df_occurrence.reset_index()
    df_modelwise = df_modelwise.reset_index()

    # Convert 'year' to int for filtering
    df_occurrence['year'] = df_occurrence['year'].astype(int)
    df_modelwise['year'] = df_modelwise['year'].astype(int)

    # Filter based on year_start and year_end
    if year_start is not None:
        df_occurrence = df_occurrence[df_occurrence['year'] >= year_start]
        df_modelwise = df_modelwise[df_modelwise['year'] >= year_start]
    if year_end is not None:
        df_occurrence = df_occurrence[df_occurrence['year'] <= year_end]
        df_modelwise = df_modelwise[df_modelwise['year'] <= year_end]

    # Ensure both DataFrames have the necessary columns ('year' and the specified stereotypes)
    missing_columns_occurrence = [col for col in ['year'] + stereotypes if col not in df_occurrence.columns]
    missing_columns_modelwise = [col for col in ['year'] + stereotypes if col not in df_modelwise.columns]

    if missing_columns_occurrence:
        logger.error(f"Missing columns in the occurrence-wise data: {missing_columns_occurrence}")
        return
    if missing_columns_modelwise:
        logger.error(f"Missing columns in the model-wise data: {missing_columns_modelwise}")
        return

    # Convert the stereotypes to percentages for both occurrence-wise and model-wise data
    for term in stereotypes:
        df_occurrence[term] = df_occurrence[term] * 100
        df_modelwise[term] = df_modelwise[term] * 100

    # Sort both DataFrames by year
    df_occurrence = df_occurrence.sort_values('year')
    df_modelwise = df_modelwise.sort_values('year')

    # Convert 'year' to string in both DataFrames
    df_occurrence['year'] = df_occurrence['year'].astype(str)
    df_modelwise['year'] = df_modelwise['year'].astype(str)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(16, 9))

    # Define dynamic color palettes for bars and lines based on the number of stereotypes
    bar_colors = sns.color_palette("Set1", len(stereotypes))
    line_colors = sns.color_palette("dark", len(stereotypes))

    # Plot bars for occurrence-wise frequencies for each term
    for idx, term in enumerate(stereotypes):
        ax1.bar(df_occurrence['year'], df_occurrence[term], color=bar_colors[idx], alpha=0.7,
                label=f'{term} (occurrence-wise)', width=0.4, align='center')

    # Set labels for the first y-axis (occurrence-wise)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Occurrence-wise Frequency (%)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Adjust the y-axis limit for occurrence-wise
    max_occurrence = max(df_occurrence[stereotypes].max())
    if max_occurrence > 0:
        ax1.set_ylim(0, max_occurrence * 1.1)
    else:
        ax1.set_ylim(0, 1)

    # Move gridlines behind the bars
    ax1.set_axisbelow(True)

    # Create a second y-axis for model-wise frequencies
    ax2 = ax1.twinx()

    # Plot lines for model-wise frequencies for each term
    for idx, term in enumerate(stereotypes):
        sns.lineplot(x='year', y=term, data=df_modelwise, ax=ax2, color=line_colors[idx], linewidth=2.5, legend=False)

    # Adjust the y-axis limit for model-wise
    max_modelwise = max(df_modelwise[stereotypes].max())
    if max_modelwise > 0:
        ax2.set_ylim(0, max_modelwise * 1.1)
    else:
        ax2.set_ylim(0, 1)

    # Set labels for the second y-axis (model-wise)
    ax2.set_ylabel('Model-wise Frequency (%)', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Align both y-axes to have their zero points match
    ax1.set_ylim(0, ax1.get_ylim()[1])  # Ensure both start at zero
    ax2.set_ylim(0, ax2.get_ylim()[1])

    # Improve readability by rotating x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Explicitly control the gridlines
    ax1.grid(axis='x', color='lightgray')  # Only vertical gridlines on the primary axis (x-axis)
    ax2.grid(False)  # Disable gridlines on the secondary axis (y-axis)

    # Build the title dynamically based on the stereotypes
    title = f"Combined Occurrence-wise and Model-wise Data for {', '.join(stereotypes)}"

    # Generate a figure name from the file_name
    fig_name = f"non_ontouml_combined_visualization_{file_name}.png"

    # Add title
    plt.title(title, fontweight='bold')

    # Manually get handles for the lines (model-wise)
    handles2 = [ax2.plot(df_modelwise['year'], df_modelwise[term], color=line_colors[idx], linewidth=2.5)[0] for
                idx, term in enumerate(stereotypes)]

    # Combine legends for both occurrence-wise and model-wise data
    handles1, labels1 = ax1.get_legend_handles_labels()  # Handles for the bars (occurrence-wise)
    labels2 = [f'{term} (model-wise)' for term in stereotypes]

    # Display the combined legend in the top left corner
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

    # Save the figure
    plt.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")

    # Close the plot to free memory
    plt.close()
