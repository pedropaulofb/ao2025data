import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import combinations
from adjustText import adjust_text
from loguru import logger

import pandas as pd

def calculate_quadrants(df_A, df_B, x_metric, y_metric, out_dir_path):
    # Calculate median values for moment A
    median_y_A = df_A[y_metric].median()
    median_x_A = df_A[x_metric].median()

    # Calculate median values for moment B
    median_y_B = df_B[y_metric].median()
    median_x_B = df_B[x_metric].median()

    # Function to determine the quadrant for a given point
    def get_quadrant(x, y, median_x, median_y):
        if x >= median_x and y >= median_y:
            return 'Q1'  # Top right
        elif x < median_x and y >= median_y:
            return 'Q2'  # Top left
        elif x < median_x and y < median_y:
            return 'Q3'  # Bottom left
        else:
            return 'Q4'  # Bottom right

    # Create a list to hold the results
    results = []

    # Iterate over each construct and determine its quadrant in A and B
    for construct in df_A['Construct'].unique():
        subset_A = df_A[df_A['Construct'] == construct]
        subset_B = df_B[df_B['Construct'] == construct]

        # Get the x and y values for this construct in moment A and B
        x_A, y_A = subset_A[x_metric].values[0], subset_A[y_metric].values[0]
        x_B, y_B = subset_B[x_metric].values[0], subset_B[y_metric].values[0]

        # Determine quadrants for both moment A and B
        quadrant_A = get_quadrant(x_A, y_A, median_x_A, median_y_A)
        quadrant_B = get_quadrant(x_B, y_B, median_x_B, median_y_B)

        # Append the results as a dictionary
        results.append({
            'construct': construct,
            'quadrant_start': quadrant_A,
            'quadrant_end': quadrant_B
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file
    output_file = os.path.join(out_dir_path, f'quadrant_analysis_{x_metric}_vs_{y_metric}.csv')
    results_df.to_csv(output_file, index=False)

    logger.success(f"Quadrant analysis saved to {output_file}.")



def execute_visualization_movement(path_file_A, path_file_B, out_dir_path):

    # Create the directory to save if it does not exist
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)
        logger.success(f"Created directory: {out_dir_path}")
    else:
        logger.info(f"Directory already exists: {out_dir_path}")

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
        fig, ax = plt.subplots(figsize=(32, 18), tight_layout=True)

        # Define a color palette with 23 distinct colors
        extended_palette = sns.color_palette("tab20", n_colors=23)

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

            # Draw dashed line (without arrowhead) from moment A to moment B
            ax.plot([subset_A[x_metric].values[0], subset_B[x_metric].values[0]],
                    [subset_A[y_metric].values[0], subset_B[y_metric].values[0]],
                    color=extended_palette[i], linestyle='--', linewidth=0.75, dashes=(5, 10))


            # Add arrowhead (without line) pointing from A to B
            ax.annotate("",
                        xy=(subset_B[x_metric].values[0], subset_B[y_metric].values[0]),  # arrow ends at B
                        xytext=(subset_A[x_metric].values[0], subset_A[y_metric].values[0]),  # starts at A
                        arrowprops=dict(arrowstyle="-|>", color=extended_palette[i], lw=1.0))

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

        # Calculate median values for moment A
        median_y_A = df_A[y_metric].median()
        median_x_A = df_A[x_metric].median()

        # Calculate median values for moment B
        median_y_B = df_B[y_metric].median()
        median_x_B = df_B[x_metric].median()

        # Plot median lines for moment A (light red)
        ax.axhline(y=median_y_A, color='lightcoral', linestyle='--', linewidth=1, label=f'Median {y_metric} (A)')
        ax.axvline(x=median_x_A, color='lightcoral', linestyle='--', linewidth=1, label=f'Median {x_metric} (A)')

        # Plot median lines for moment B (light blue)
        ax.axhline(y=median_y_B, color='lightblue', linestyle='--', linewidth=1, label=f'Median {y_metric} (B)')
        ax.axvline(x=median_x_B, color='lightblue', linestyle='--', linewidth=1, label=f'Median {x_metric} (B)')

        # Optionally add text for the median values
        ax.text(df_B[x_metric].max(), median_y_B, f'median B: {median_y_B:.2f}', color='lightblue', fontsize=10,
                ha='right', va='bottom')
        ax.text(df_A[x_metric].max(), median_y_A, f'median A: {median_y_A:.2f}', color='lightcoral', fontsize=10,
                ha='right', va='bottom')

        ax.text(median_x_B, df_B[y_metric].max(), f'median B: {median_x_B:.2f}', color='lightblue', fontsize=10,
                ha='right', va='top', rotation=90)
        ax.text(median_x_A, df_A[y_metric].max(), f'median A: {median_x_A:.2f}', color='lightcoral', fontsize=10,
                ha='right', va='top', rotation=90)

        # Save figure
        formatted_x_metric = x_metric.replace('_', ' ').title()
        formatted_y_metric = y_metric.replace('_', ' ').title()
        fig_name = f'movement_analysis_{formatted_x_metric}_vs_{formatted_y_metric}.png'
        fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
        logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
        plt.close(fig)

        calculate_quadrants(df_A, df_B, 'Total Frequency', 'Group Frequency', out_dir_path)