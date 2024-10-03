import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger

def generate_non_ontouml_combined_visualization(in_dir_path, out_dir_path, file_path_yearly, file_path_overall):
    # Load occurrence-wise and model-wise data
    df_occurrence = pd.read_csv(os.path.join(in_dir_path, file_path_yearly))
    df_modelwise = pd.read_csv(os.path.join(in_dir_path, file_path_overall))

    # Filter the data for the columns you want ('year', 'none', 'other') and make a deep copy
    df_occurrence = df_occurrence[['year', 'none', 'other']].copy()
    df_modelwise = df_modelwise[['year', 'none', 'other']].copy()

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
    ax1.set_ylim(0, max(df_occurrence[['none', 'other']].max()) * 1.1)  # Set y-limit for occurrence-wise

    # Move gridlines behind the bars
    ax1.set_axisbelow(True)

    # Create a second y-axis for model-wise frequencies
    ax2 = ax1.twinx()

    # Plot model-wise frequencies ('none' and 'other') as lines without adding extra legends
    sns.lineplot(x='year', y='none', data=df_modelwise, ax=ax2, color=line_colors['none'], linewidth=2.5, legend=False)
    sns.lineplot(x='year', y='other', data=df_modelwise, ax=ax2, color=line_colors['other'], linewidth=2.5,
                 legend=False)

    # Set labels for the second y-axis (model-wise)
    ax2.set_ylabel('Model-wise Frequency (%)', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(0, max(df_modelwise[['none', 'other']].max()) * 1.1)  # Set y-limit for model-wise

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

    # Generate a figure name from the file_path (remove extension and append .png)
    fig_name = os.path.splitext(file_path_yearly)[0] + "_combined_visualization.png"

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
    ax1.legend(handles1 + handles2, labels1 + ['none (model-wise)', 'other (model-wise)'], loc='upper left')

    # Ensure output directory exists
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)

    # Save the figure
    plt.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")

    # Close the plot to free memory
    plt.close()
