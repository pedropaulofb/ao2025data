import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import combinations
from adjustText import adjust_text


def execute_visualization_movement(path_file_A, path_file_B, out_dir_path):
    # Load data from moment A and moment B using the provided file paths
    df_A = pd.read_csv(path_file_A)
    df_B = pd.read_csv(path_file_B)

    # Check if both dataframes have the same constructs and columns
    assert all(df_A['Construct'] == df_B['Construct']), "The constructs in both moments must match."

    # Convert 'Construct' column to categorical type for proper ordering
    df_A['Construct'] = pd.Categorical(df_A['Construct'], categories=df_A['Construct'].unique(), ordered=True)
    df_B['Construct'] = pd.Categorical(df_B['Construct'], categories=df_B['Construct'].unique(), ordered=True)

    # Get all numeric columns (excluding 'Construct')
    numeric_columns = df_A.select_dtypes(include='number').columns

    # Generate all unique combinations of two metrics for scatter plots
    for x_metric, y_metric in combinations(numeric_columns, 2):
        fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)

        # Define the base color palette (12 distinct colors) for constructs
        base_palette = sns.color_palette('tab10', n_colors=12)
        extended_palette = base_palette + base_palette[:11]  # Extending to 23 colors

        texts = []
        # Plot both moments A and B and draw arrows between the points
        for i, construct in enumerate(df_A['Construct'].unique()):
            subset_A = df_A[df_A['Construct'] == construct]
            subset_B = df_B[df_B['Construct'] == construct]

            # Plot moment A (start point)
            ax.scatter(subset_A[x_metric], subset_A[y_metric],
                       color=extended_palette[i], marker='o', s=100, edgecolor='w', label=construct)

            # Plot moment B (end point)
            ax.scatter(subset_B[x_metric], subset_B[y_metric],
                       color=extended_palette[i], marker='o', s=100, edgecolor='w')

            # Draw arrow from moment A to moment B
            ax.arrow(subset_A[x_metric].values[0], subset_A[y_metric].values[0],
                     subset_B[x_metric].values[0] - subset_A[x_metric].values[0],
                     subset_B[y_metric].values[0] - subset_A[y_metric].values[0],
                     head_width=0.02, head_length=0.05, fc=extended_palette[i], ec=extended_palette[i])

            # Add label next to the start point
            text = ax.text(subset_A[x_metric].values[0], subset_A[y_metric].values[0], construct,
                           fontsize=8, color='black', ha='left', va='top')
            texts.append(text)

        # Adjust text to avoid overlap
        adjust_text(texts)

        # Adding labels and title
        ax.set_xlabel(x_metric.replace('_', ' ').title())
        ax.set_ylabel(y_metric.replace('_', ' ').title())
        ax.set_title(f'Movement of {x_metric.replace("_", " ").title()} vs. {y_metric.replace("_", " ").title()}',
                     fontweight='bold')

        # Calculate and plot median lines for moment B
        median_y = df_B[y_metric].median()
        median_x = df_B[x_metric].median()
        ax.axhline(y=median_y, color='black', linestyle='--', linewidth=1)
        ax.axvline(x=median_x, color='black', linestyle='--', linewidth=1)

        # Adding text for the median values at the extreme parts of the lines
        ax.text(df_B[x_metric].max(), median_y, f'median: {median_y:.2f}', color='gray', fontsize=10, ha='right',
                va='bottom')
        ax.text(median_x, df_B[y_metric].max(), f'median: {median_x:.2f}', color='gray', fontsize=10, ha='right',
                va='top', rotation=90)

        # Save figure
        formatted_x_metric = x_metric.replace('_', ' ').title()
        formatted_y_metric = y_metric.replace('_', ' ').title()
        fig_name = f'movement_analysis_{formatted_x_metric}_vs_{formatted_y_metric}.png'
        fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
        print(f"Figure {fig_name} successfully saved in {out_dir_path}.")
        plt.close(fig)


# Usage example
path_file_A = '../../outputs/statistics/cs_ontouml_no_classroom_until_2017_f/frequency_analysis.csv'
path_file_B = '../../outputs/statistics/cs_ontouml_no_classroom_after_2018_f/frequency_analysis.csv'
out_dir_path = '.'
execute_visualization_movement(path_file_A, path_file_B, out_dir_path)
