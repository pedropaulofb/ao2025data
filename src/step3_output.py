import os

import numpy as np
import pandas as pd
import seaborn as sns
from icecream import ic
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from src.utils import load_object, create_visualizations_out_dirs, append_chars_to_labels, bold_left_labels, color_text


def generate_visualizations(datasets, output_dir):
    datasets = load_object(datasets, "datasets")

    coverages = [0.9]
    for dataset in datasets:

        if ("until" in dataset.name) or ("after" in dataset.name):
            continue

        for coverage in coverages:
            plot_pareto_combined(dataset, output_dir, coverage)
        execute_non_ontouml_analysis(dataset, output_dir)


def plot_pareto_combined(dataset, output_dir: str, coverage_limit: float = None) -> None:
    """
    Plot combined Pareto chart for occurrence-wise and group-wise stereotype frequencies from class and relation data
    (both raw and clean).

    :param dataset: Dataset object containing models and statistics.
    :param output_dir: Directory to save the generated Pareto charts.
    :param coverage_limit: Optional coverage limit value (between 0 and 1) to draw a vertical line on the plot.
    """
    # Create output directories
    class_clean_out, relation_clean_out = create_visualizations_out_dirs(output_dir,
                                                                         dataset.name)

    # Define a helper function to create Pareto charts for a given dataset
    def create_pareto_chart(data_occurrence, data_groupwise, title, output_path, coverage_limit=None):
        # Calculate the percentage frequency for occurrence-wise and group-wise data
        data_occurrence['Percentage Frequency'] = (data_occurrence['Frequency'] / data_occurrence[
            'Frequency'].sum()) * 100
        data_groupwise['Percentage Group-wise Frequency'] = (data_groupwise['Group-wise Frequency'] / data_groupwise[
            'Group-wise Frequency'].sum()) * 100

        # Ensure that group-wise data is ordered based on occurrence-wise data
        data_groupwise = data_groupwise.set_index('Stereotype').reindex(data_occurrence['Stereotype']).reset_index()

        # Calculate cumulative percentages for occurrence-wise and group-wise data
        data_occurrence['Cumulative Percentage'] = data_occurrence['Percentage Frequency'].cumsum()
        data_groupwise['Group-wise Cumulative Percentage'] = data_groupwise['Percentage Group-wise Frequency'].cumsum()

        # Reset index to ensure integer indexing
        data_occurrence = data_occurrence.reset_index(drop=True)
        data_groupwise = data_groupwise.reset_index(drop=True)

        # Create the plot
        fig, ax1 = plt.subplots(figsize=(16, 9), tight_layout=True)

        # Set solid white background for the figure
        fig.patch.set_alpha(1)  # Ensure complete figure background opacity
        fig.patch.set_facecolor('white')  # Set background color to white

        # Bar plot for occurrence-wise frequency
        bar_width = 0.4
        ax1.bar(data_occurrence['Stereotype'], data_occurrence['Percentage Frequency'], width=bar_width,
                label='Aggregate Occurrence', align='center', color='skyblue', zorder=1)

        # Bar plot for group-wise frequency (shifted to the right)
        ax1.bar(data_groupwise['Stereotype'], data_groupwise['Percentage Group-wise Frequency'], width=bar_width,
                label='Model Coverage', align='edge', color='lightgreen', zorder=1)

        # Calculate medians
        median_occurrence = np.median(data_occurrence['Percentage Frequency'])
        median_groupwise = np.median(data_groupwise['Percentage Group-wise Frequency'])

        # Plot horizontal dashed lines for the medians (keep simple labels for the legend)
        ax1.axhline(median_occurrence, color='blue', linestyle='--', linewidth=2.0, label='Aggregate Occurrence Median')
        ax1.axhline(median_groupwise, color='green', linestyle='--', linewidth=2.0, label='Model Coverage Median')

        # Annotate the medians at the rightmost end of the lines
        ax1.annotate(f'{median_occurrence:.2f}%', xy=(len(data_occurrence) - 1, median_occurrence), xytext=(5, 5),
                     textcoords='offset points', color='blue', fontsize=10, fontweight='bold')

        ax1.annotate(f'{median_groupwise:.2f}%', xy=(len(data_groupwise) - 1, median_groupwise), xytext=(5, 5),
                     textcoords='offset points', color='green', fontsize=10, fontweight='bold')

        # Set labels and title
        ax1.set_ylabel('Aggregate Occurrence (%)', fontsize=12)
        # Title ommited for the paper
        # ax1.set_title(title, fontweight='bold', fontsize=14)

        # Rotate the labels by 90 degrees for better readability
        ax1.tick_params(axis='x', rotation=90)

        # Line plot for occurrence-wise cumulative percentage
        ax2 = ax1.twinx()  # Create a second y-axis

        ax2.plot(range(len(data_occurrence)), data_occurrence['Cumulative Percentage'], color='blue', marker='o',
                 label='Cumulative Aggregate Occurrence', linestyle='-', linewidth=2, zorder=2)

        # Line plot for group-wise cumulative percentage
        ax2.plot(range(len(data_groupwise)), data_groupwise['Group-wise Cumulative Percentage'], color='green',
                 marker='o', label='Cumulative Model Coverage', linestyle='-', linewidth=2, zorder=2)

        if coverage_limit is not None:
            coverage_limit_percent = coverage_limit * 100

            # Find the index where both cumulative percentages exceed the coverage limit
            for i in range(len(data_occurrence)):
                occurrence_cumulative = data_occurrence.loc[i, 'Cumulative Percentage']
                groupwise_cumulative = data_groupwise.loc[i, 'Group-wise Cumulative Percentage']
                if min(occurrence_cumulative, groupwise_cumulative) >= coverage_limit_percent:
                    # Plot the vertical dashed line at this index
                    ax1.axvline(x=i, color='red', linestyle='--', linewidth=2.0, label=f'Projected Coverage ≥ {coverage_limit_percent:.1f}%')

                    # Annotate the occurrence-wise cumulative percentage, automatically adjusted
                    ax2.annotate(f'{occurrence_cumulative:.1f}%', xy=(i, occurrence_cumulative),
                                 textcoords='offset points', xytext=(5, -5), ha='left', va='top', fontsize=10,
                                 color='blue')

                    # Annotate the group-wise cumulative percentage, automatically adjusted
                    ax2.annotate(f'{groupwise_cumulative:.1f}%', xy=(i, groupwise_cumulative),
                                 textcoords='offset points', xytext=(5, -5), ha='left', va='top', fontsize=10,
                                 color='green')

                    red_line_index = i
                    break

        # Set the y-axis label and ticks for the cumulative percentage
        ax2.set_ylabel('Model Coverage (%)', fontsize=12)
        ax2.set_yticks(range(0, 101, 10))

        ax1.tick_params(axis='x', rotation=45)  # Rotate by 45 degrees
        texts = ax1.get_xticklabels()  # Color specific x-axis labels
        # Modify the labels with symbols
        texts = append_chars_to_labels(texts, data_occurrence, data_groupwise, median_occurrence, median_groupwise)

        bold_left_labels(texts, red_line_index)

        for label in texts:
            # Align labels to the right, so the last character is near the corresponding tick
            label.set_ha('right')
            # Adjust the label positioning slightly if needed (use pad for further fine-tuning)
            label.set_position(
                (label.get_position()[0] - 0.02, label.get_position()[1]))  # Adjust the x-position slightly

        # Set the new x-tick labels
        ax1.set_xticks(range(len(texts)))  # Make sure the number of ticks matches the number of labels
        ax1.set_xticklabels(texts)  # Apply the new labels

        color_text(ax1.get_xticklabels())

        # Get the current x-tick positions
        x_positions = range(len(texts))

        # Explicitly set the x-tick positions to match the number of labels
        ax1.set_xticks(x_positions)

        # Apply the updated labels back to the axis
        ax1.set_xticklabels(texts)

        ax2.set_axisbelow(True)  # Ensure gridlines are below other plot elements
        ax1.grid(False)
        ax2.grid(True, axis='y', which='major')

        # Extract handles and labels from both axes
        handles1, labels1 = ax1.get_legend_handles_labels()  # Bar plot legend info (ax1)
        handles2, labels2 = ax2.get_legend_handles_labels()  # Line plot legend info (ax2)

        # Combine the handles and labels from both legends
        combined_handles = handles1 + handles2
        combined_labels = labels1 + labels2

        # Reordering legend elements
        combined_handles = [combined_handles[3], combined_handles[4], combined_handles[5],
                            combined_handles[6], combined_handles[0], combined_handles[1],combined_handles[2]]
        combined_labels = [combined_labels[3], combined_labels[4], combined_labels[5],
                            combined_labels[6], combined_labels[0], combined_labels[1],combined_labels[2]]


        combined_legend = ax2.legend(combined_handles, combined_labels, loc='center right', bbox_to_anchor=(1, 0.5),
                                     frameon=True, facecolor='white', edgecolor='black', framealpha=1, shadow=True,
                                     handletextpad=1.5, borderaxespad=2, borderpad=1.5)


        # Set the zorder of the combined legend higher than other elements
        combined_legend.set_zorder(10)
        plt.draw()  # Force re-rendering of the plot, ensuring legend is drawn last

        # Customize the title: left-align and make it bold
        combined_legend.get_title().set_fontweight('bold')  # Make the title bold

        # Create proxy artists for the symbols
        legend_elements = [Line2D([0], [0], marker='+', color='black', linestyle='None', markersize=7,
                                  label='Stereotype ≥ median (Aggregate Occurrence)'),
                           Line2D([0], [0], marker='$*$', color='black', linestyle='None', markersize=7,
                                  label='Stereotype ≥ median (Model Coverage)')]

        # Add these proxies to the existing legend
        ax2.legend(handles=combined_handles + legend_elements,  # Combine previous handles with new ones
                   loc='center right', bbox_to_anchor=(1, 0.5), frameon=True, facecolor='white', edgecolor='black',
                   framealpha=1, shadow=True, handletextpad=1.5, borderaxespad=2, borderpad=1.5)

        # Save the plot
        fig_name = f"pareto_combined_cov_{coverage_limit}.png"
        # Ensure tight layout for the figure before saving
        plt.tight_layout()
        fig.savefig(os.path.join(output_path, fig_name), dpi=300, bbox_inches='tight')

        logger.success(f"Figure {fig_name} successfully saved in {output_path}.")
        plt.close(fig)

    # Now generate Pareto charts for class clean and relation clean datasets
    create_pareto_chart(pd.DataFrame(dataset.class_statistics_clean['rank_frequency_distribution']),
                        pd.DataFrame(dataset.class_statistics_clean['rank_groupwise_frequency_distribution']),
                        'Class Stereotypes: Aggregate Occurrence and Model Coverage Analysis', class_clean_out, coverage_limit)

    create_pareto_chart(pd.DataFrame(dataset.relation_statistics_clean['rank_frequency_distribution']),
                        pd.DataFrame(dataset.relation_statistics_clean['rank_groupwise_frequency_distribution']),
                        'Relation Stereotypes: Aggregate Occurrence and Model Coverage Analysis', relation_clean_out, coverage_limit)


def execute_non_ontouml_analysis(dataset, out_dir_path):
    st_types = ['class', 'relation']
    st_norms = ['yearly']

    years_start = [2015]

    for st_type in st_types:
        final_out_dir = os.path.join(out_dir_path, dataset.name, f"{st_type}_raw")

        # Create folder if it does not exist
        os.makedirs(final_out_dir, exist_ok=True)

        for st_norm in st_norms:
            df_occurrence = dataset.years_stereotypes_data[f'{st_type}_ow_{st_norm}']
            df_modelwise = dataset.years_stereotypes_data[f'{st_type}_mw_{st_norm}']

            for year_start in years_start:
                generate_non_ontouml_combined_visualization(df_occurrence, df_modelwise, final_out_dir,
                                                            f'{st_type}_{st_norm}_{year_start}', year_start=year_start)


def generate_non_ontouml_combined_visualization(df_occurrence, df_modelwise, out_dir_path, file_name, year_start=None,
                                                year_end=None):
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
            label='none (Aggregate Occurence)', width=0.4, align='center')
    ax1.bar(df_occurrence['year'], df_occurrence['other'], color=bar_colors['other'], alpha=0.7,
            label='other (Aggregate Occurence)', width=0.4, align='edge')

    # Set labels for the first y-axis (occurrence-wise)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Aggregate Occurrence (%)', color='black')
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
    ax2.set_ylabel('Model Coverage (%)', color='black')
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

    # Title ommited for the paper
    # plt.title(title, fontweight='bold')

    # Plot model-wise frequencies ('none' and 'other') as lines without adding extra legends
    line_none, = ax2.plot(df_modelwise['year'], df_modelwise['none'], color=line_colors['none'], linewidth=2.5,
                          label='none (Model Coverage)')
    line_other, = ax2.plot(df_modelwise['year'], df_modelwise['other'], color=line_colors['other'], linewidth=2.5,
                           label='other (Model Coverage)')

    # Combine legends for both occurrence-wise and model-wise data
    handles1, labels1 = ax1.get_legend_handles_labels()  # Handles for the bars (occurrence-wise)
    handles2 = [line_none, line_other]  # Manually get handles for the lines (model-wise)

    # Display the combined legend in the top left corner
    ax1.legend(handles1 + handles2, labels1 + ['none (Model Coverage)', 'other (Model Coverage)'], loc='upper right')

    # plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(out_dir_path, fig_name), dpi=300, bbox_inches='tight')
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")

    # Close the plot to free memory
    plt.close()
