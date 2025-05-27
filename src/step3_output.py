import os

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from src.utils import load_object, append_chars_to_labels, bold_left_labels, color_text


def generate_visualizations(datasets, output_dir):
    datasets = load_object(datasets, "datasets")

    coverages = [0.9]
    for dataset in datasets:

        if ("until" in dataset.name) or ("after" in dataset.name):
            continue

        for coverage in coverages:
            plot_pareto_combined(dataset, output_dir, coverage)


def plot_pareto_combined(dataset, output_dir: str, coverage_limit: float = None) -> None:
    """
    Plot combined Pareto chart for occurrence-wise and group-wise stereotype frequencies from class and relation data
    (both raw and clean).

    :param dataset: Dataset object containing models and statistics.
    :param output_dir: Directory to save the generated Pareto charts.
    :param coverage_limit: Optional coverage limit value (between 0 and 1) to draw a vertical line on the plot.
    """
    # Create output directories
    common_output_path = os.path.join(output_dir, dataset.name)
    os.makedirs(common_output_path, exist_ok=True)

    # Define a helper function to create Pareto charts for a given dataset
    def create_pareto_chart(data_occurrence, data_groupwise, output_path, coverage_limit=None, suffix=""):

        # Centralized Color Config
        COLORS = {
            # RTC = Green (bright)
            "rtc_bar": "#00E676",  # Bright neon green
            "rtc_line": "#0D6E34",  # Dark green
            "rtc_median": "#0D6E34",
            "rtc_annotation": "#0D6E34",

            # RMC = Purple (bright)
            "rmc_bar": "#EAA9F5",  # Light orchid/lavender
            "rmc_line": "#C61BE3",  # Bright vivid purple
            "rmc_median": "#C61BE3",
            "rmc_annotation": "#C61BE3",

            # Other
            "coverage_line": "lightgrey",
            "coverage_marker": "red"
        }

        # Calculate RTC and RMC as percentages
        data_occurrence['RTC (%)'] = (data_occurrence['Frequency'] / data_occurrence['Frequency'].sum()) * 100
        data_groupwise['RMC (%)'] = (data_groupwise['Group-wise Frequency'] / data_groupwise[
            'Group-wise Frequency'].sum()) * 100

        # Ensure that group-wise data is ordered based on occurrence-wise data
        data_groupwise = data_groupwise.set_index('Stereotype').reindex(data_occurrence['Stereotype']).reset_index()

        # Calculate cumulative percentages for occurrence-wise and group-wise data
        data_occurrence['Cumulative RTC (%)'] = data_occurrence['RTC (%)'].cumsum()
        data_groupwise['Cumulative RMC (%)'] = data_groupwise['RMC (%)'].cumsum()

        # Reset index to ensure integer indexing
        data_occurrence = data_occurrence.reset_index(drop=True)
        data_groupwise = data_groupwise.reset_index(drop=True)

        # Create the plot
        # fig, ax1 = plt.subplots(figsize=(16, 9), tight_layout=True)
        fig, ax1 = plt.subplots(figsize=(8, 5), tight_layout=True)

        # Set solid white background for the figure
        fig.patch.set_alpha(1)  # Ensure complete figure background opacity
        fig.patch.set_facecolor('white')  # Set background color to white

        # Bar plot for occurrence-wise frequency
        bar_width = 0.4
        ax1.bar(data_occurrence['Stereotype'], data_occurrence['RTC (%)'], width=bar_width,
                label='Relative Total Count (RTC)', align='center', color=COLORS["rtc_bar"], zorder=2)

        ax1.bar(data_groupwise['Stereotype'], data_groupwise['RMC (%)'], width=bar_width,
                label='Relative Model Coverage (RMC)', align='edge', color=COLORS["rmc_bar"], zorder=2)

        # Calculate medians
        median_occurrence = np.median(data_occurrence['RTC (%)'])
        median_groupwise = np.median(data_groupwise['RMC (%)'])

        # Plot horizontal dashed lines for the medians (keep simple labels for the legend)
        ax1.axhline(median_occurrence, color=COLORS["rtc_median"], linestyle='--', linewidth=2.0, label='RTC Median', zorder=3)
        ax1.axhline(median_groupwise, color=COLORS["rmc_median"], linestyle='--', linewidth=2.0, label='RMC Median', zorder=3)

        # Annotate the medians at the rightmost end of the lines
        # Used (-7, 10) and (-50, 2) for relation

        ax1.annotate(f'{median_occurrence:.2f}%', xy=(len(data_occurrence) - 1, median_occurrence), xytext=(-7, 2),
                     textcoords='offset points', color=COLORS["rtc_annotation"], fontsize=10, fontweight='bold', zorder=5)

        ax1.annotate(f'{median_groupwise:.2f}%', xy=(len(data_groupwise) - 1, median_groupwise), xytext=(-7, 2),
                     textcoords='offset points', color=	COLORS["rmc_annotation"], fontsize=10, fontweight='bold', zorder=5)

        # Set labels and title
        ax1.set_ylabel('Stereotype RTC & RMC (%)', fontsize=12)
        # Title omitted for the paper
        # ax1.set_title(title, fontweight='bold', fontsize=14)

        # Rotate the labels by 90 degrees for better readability
        ax1.tick_params(axis='x', rotation=90)

        # Line plot for occurrence-wise cumulative percentage
        ax2 = ax1.twinx()  # Create a second y-axis

        ax2.plot(range(len(data_occurrence)), data_occurrence['Cumulative RTC (%)'], color=COLORS["rtc_line"], marker='o',
                 label='Cumulative RTC', linestyle='-', linewidth=2, zorder=2)

        ax2.plot(range(len(data_groupwise)), data_groupwise['Cumulative RMC (%)'], color=COLORS["rmc_line"],
                 marker='o', label='Cumulative RMC', linestyle='-', linewidth=2, zorder=2)

        if coverage_limit is not None:
            coverage_limit_percent = coverage_limit * 100
            ax2.axhline(
                y=coverage_limit_percent,
                color=COLORS["coverage_line"],
                linestyle=(0, (5, 5)),  # More spaced-out dashes
                linewidth=1.0,
                zorder=-5,  # Ensure it remains in the background
                label='90% Coverage'
            )

            # Find the index where both cumulative percentages exceed the coverage limit
            for i in range(len(data_occurrence)):
                occurrence_cumulative = data_occurrence.loc[i, 'Cumulative RTC (%)']
                groupwise_cumulative = data_groupwise.loc[i, 'Cumulative RMC (%)']
                if min(occurrence_cumulative, groupwise_cumulative) >= coverage_limit_percent:
                    # Plot the vertical dashed line at this index
                    ax1.axvline(x=i, color=COLORS["coverage_marker"], linestyle='--', linewidth=2.0,
                                label=f'Projected Cumulative ≥ {coverage_limit_percent:.1f}%')

                    # Annotate the occurrence-wise cumulative percentage, automatically adjusted
                    ax2.annotate(f'{occurrence_cumulative:.1f}%', xy=(i, occurrence_cumulative),
                                 textcoords='offset points', xytext=(3, 14), ha='left', va='top', fontsize=10,
                                 color=COLORS["rtc_annotation"])

                    # Annotate the group-wise cumulative percentage, automatically adjusted
                    ax2.annotate(f'{groupwise_cumulative:.1f}%', xy=(i, groupwise_cumulative),
                                 textcoords='offset points', xytext=(3, -5), ha='left', va='top', fontsize=10,
                                 color=COLORS["rmc_annotation"])

                    red_line_index = i
                    break

        # Set the y-axis label and ticks for the cumulative percentage
        ax2.set_ylabel('Cumulative RTC & RMC (%)', fontsize=12)
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

        ax1.grid(False)
        ax2.grid(False)

        # Extract handles and labels from both axes
        handles1, labels1 = ax1.get_legend_handles_labels()  # Bar plot legend info (ax1)
        handles2, labels2 = ax2.get_legend_handles_labels()  # Line plot legend info (ax2)

        # Combine the handles and labels from both legends
        combined_handles = handles1 + handles2
        combined_labels = labels1 + labels2

        # Reordering legend elements
        combined_handles = [combined_handles[3], combined_handles[4], combined_handles[5],
                            combined_handles[6], combined_handles[0], combined_handles[1], combined_handles[2]]
        combined_labels = [combined_labels[3], combined_labels[4], combined_labels[5],
                           combined_labels[6], combined_labels[0], combined_labels[1], combined_labels[2]]

        combined_legend = ax2.legend(combined_handles, combined_labels, loc='center right', bbox_to_anchor=(1, 0.5),
                                     frameon=True, facecolor='white', edgecolor='black', framealpha=1, shadow=True)

        # Set the zorder of the combined legend higher than other elements
        combined_legend.set_zorder(10)
        plt.draw()  # Force re-rendering of the plot, ensuring legend is drawn last

        # Customize the title: left-align and make it bold
        combined_legend.get_title().set_fontweight('bold')  # Make the title bold

        # Create proxy artists for the symbols
        legend_elements = [Line2D([0], [0], marker='+', color='black', linestyle='None', markersize=7,
                                  label='Stereotype ≥ median (RTC)'),
                           Line2D([0], [0], marker='$*$', color='black', linestyle='None', markersize=7,
                                  label='Stereotype ≥ median (RMC)')]

        # Add these proxies to the existing legend
        ax2.legend(handles=combined_handles + legend_elements,  # Combine previous handles with new ones
                   loc='center right', bbox_to_anchor=(1, 0.5), frameon=True, facecolor='white', edgecolor='black',
                   framealpha=1, shadow=True, handletextpad=1, borderaxespad=1, borderpad=0.5, fontsize=8)

        # Save the plot
        fig_name = f"pareto_combined_cov_{coverage_limit}_{suffix}.png"
        # Ensure tight layout for the figure before saving
        plt.tight_layout()
        fig.savefig(os.path.join(output_path, fig_name), dpi=300, bbox_inches='tight')

        logger.success(f"Figure {fig_name} successfully saved in {output_path}.")
        plt.close(fig)

    # Now generate Pareto charts for class clean and relation clean datasets
    create_pareto_chart(pd.DataFrame(dataset.class_statistics_clean['rank_frequency_distribution']),
                        pd.DataFrame(dataset.class_statistics_clean['rank_groupwise_frequency_distribution']),
                        common_output_path, coverage_limit, suffix="class")

    create_pareto_chart(pd.DataFrame(dataset.relation_statistics_clean['rank_frequency_distribution']),
                        pd.DataFrame(dataset.relation_statistics_clean['rank_groupwise_frequency_distribution']),
                        common_output_path, coverage_limit, suffix="relation")
