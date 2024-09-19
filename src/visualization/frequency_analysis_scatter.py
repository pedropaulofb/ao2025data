import os
import re
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from loguru import logger

from src.color_legend import color_text


# Function to format the metric name
def format_metric_name(metric):
    # Convert to lowercase
    formatted_metric = metric.lower()
    # Replace spaces with underscores
    formatted_metric = formatted_metric.replace(' ', '_')
    # Remove text inside parentheses (including parentheses)
    formatted_metric = re.sub(r'\s*\(.*?\)', '', formatted_metric)
    return formatted_metric


# Function to create all possible scatter plots for the metric combinations
def execute_visualization_frequency_analysis_scatter(in_dir_path, out_dir_path, file_path):
    df = pd.read_csv(os.path.join(in_dir_path, file_path))

    # Convert the 'Construct' column to a categorical type to ensure proper ordering in the plots
    df['Construct'] = pd.Categorical(df['Construct'], categories=df['Construct'].unique(), ordered=True)

    # Get all numeric columns (excluding 'Construct')
    numeric_columns = df.select_dtypes(include='number').columns

    # Generate all unique combinations of two metrics
    for x_metric, y_metric in combinations(numeric_columns, 2):
        fig = plt.figure(figsize=(16, 9), tight_layout=True)

        # Define the base colors for the plot (12 distinct colors)
        base_palette = sns.color_palette('tab10', n_colors=12)
        extended_palette = base_palette + base_palette[:11]  # 12 colors + 11 more to make 23 total

        texts = []
        # Plotting the scatter plot with circles and adding labels to each point
        for i, construct in enumerate(df['Construct'].unique()):
            subset = df[df['Construct'] == construct]
            # Plot the point (circle marker)
            plt.scatter(subset[x_metric], subset[y_metric],
                        color=extended_palette[i], marker='o', s=100, edgecolor='w', label=construct)

            # Add label next to each point
            for j in range(len(subset)):
                text = plt.text(subset[x_metric].values[j],
                                subset[y_metric].values[j], construct, fontsize=8,
                                color='black', ha='left', va='top')
                texts.append(text)

        # Apply the color_text function to the list of text objects
        color_text(texts)

        # Adjust text to avoid overlap
        adjust_text(texts)

        # Adding labels and title
        plt.xlabel(x_metric.replace('_', ' ').title())
        plt.ylabel(y_metric.replace('_', ' ').title())
        plt.title(f'{x_metric.replace("_", " ").title()} vs. {y_metric.replace("_", " ").title()}', fontweight='bold')

        # Calculate median values
        median_y = df[y_metric].median()
        median_x = df[x_metric].median()

        # Adding a cross to separate the plot into four quadrants using the median
        plt.axhline(y=median_y, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=median_x, color='black', linestyle='--', linewidth=1)

        # Adding text for the median values at the extreme (highest values) parts of the lines
        plt.text(df[x_metric].max(), median_y, f'median: {median_y:.2f}', color='gray', fontsize=10,
                 ha='right', va='bottom')
        plt.text(median_x, df[y_metric].max(), f'median: {median_x:.2f}', color='gray',
                 fontsize=10, ha='right', va='top', rotation=90)

        # Remove the legend (as labels are now added directly to the points)
        plt.legend().remove()

        formatted_x_metric = format_metric_name(x_metric)
        formatted_y_metric = format_metric_name(y_metric)

        plt.tight_layout()
        fig_name = f'frequency_analysis_{formatted_x_metric}_vs_{formatted_y_metric}.png'
        fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
        logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
        plt.close(fig)
