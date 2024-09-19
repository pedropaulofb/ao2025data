import os
from itertools import combinations

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from loguru import logger


def plot_custom_quadrant_flow_chart(results_df, out_dir_path):
    # Data setup from your DataFrame
    constructs = results_df['construct']  # Construct labels (first column)
    quadrant_start = results_df['quadrant_start']  # Start quadrants (A)
    quadrant_end = results_df['quadrant_end']  # End quadrants (B)

    # Quadrant mapping to numerical values (lanes), reversed to have Q1 at the top
    quadrant_map = {'Q1': 4, 'Q2': 3, 'Q3': 2, 'Q4': 1}

    # Lists to hold positions for start and end to avoid overlap
    start_positions = []
    end_positions = []

    # Function to space constructs based on their number
    def evenly_space_constructs(constructs_in_quadrant, q_num):
        n = len(constructs_in_quadrant)
        if n == 1:
            return [q_num]  # If only one construct, place it in the middle of the quadrant
        else:
            lower_limit = q_num - 0.5
            upper_limit = q_num + 0.5
            # Calculate the margin (offset) between each construct and the quadrant limits
            margin = (upper_limit - lower_limit) / (n + 1)
            return [lower_limit + margin * (i + 1) for i in range(n)]  # Space constructs evenly with margins

    # Group constructs by their start and end quadrants for spacing purposes
    constructs_by_start = results_df.groupby('quadrant_start')['construct'].apply(list).to_dict()
    constructs_by_end = results_df.groupby('quadrant_end')['construct'].apply(list).to_dict()

    # Apply even spacing within each quadrant for start and end positions
    start_spacing = {}
    end_spacing = {}

    for q in quadrant_map:
        if q in constructs_by_start:
            start_spacing[q] = evenly_space_constructs(constructs_by_start[q], quadrant_map[q])
        if q in constructs_by_end:
            end_spacing[q] = evenly_space_constructs(constructs_by_end[q], quadrant_map[q])

    # Create counters to track positioning for each construct within its quadrant
    start_counters = {q: 0 for q in quadrant_map}
    end_counters = {q: 0 for q in quadrant_map}

    # Darker, stronger colors for the lines and labels
    cmap = cm.get_cmap('tab10', len(constructs))  # Use 'tab10' colormap for stronger, darker colors

    # Color scheme for quadrants with stronger colors (ensuring matching in the legend)
    quadrant_colors = ['#abd194', '#d194a6', '#8894ba', '#c2ab93']  # Q1-Q4 colors (in reversed order)

    # Calculate positions for each construct
    for i, construct in enumerate(constructs):
        start_q = quadrant_start.iloc[i]
        end_q = quadrant_end.iloc[i]

        # Get the position for start and end, adjusting for the current count in each quadrant
        start_position = start_spacing[start_q][start_counters[start_q]]
        start_counters[start_q] += 1

        if start_q == end_q:
            # Keep the same position for a straight line within the same quadrant
            end_position = start_position
        else:
            end_position = end_spacing[end_q][end_counters[end_q]]
            end_counters[end_q] += 1

        # Append the positions to the respective lists
        start_positions.append(start_position)
        end_positions.append(end_position)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)  # Adjusted figure size to accommodate the legend

    # X-axis positions for moments A and B
    x = [0, 1]  # Two moments: A and B

    # Plot background lanes for each quadrant with stronger colors (in reversed order)
    for i, (q_label, q_num) in enumerate(quadrant_map.items()):
        ax.fill_betweenx([q_num - 0.5, q_num + 0.5], 0, 1, color=quadrant_colors[i], alpha=0.3)

    # Plot lines between lanes (quadrants) for each construct
    for i, construct in enumerate(constructs):
        line_color = cmap(i)  # Get unique, stronger color for the current construct
        ax.plot(x, [start_positions[i], end_positions[i]], marker='o', color=line_color)

        # Add the labels for the constructs on the left side (Moment A) in the same color as the line
        ax.text(-0.05, start_positions[i], construct, verticalalignment='center', horizontalalignment='right',
                color=line_color)

    # Remove quadrant labels from the y-axis (not needed anymore)
    ax.set_yticks([])

    # Set ticks and labels for the x-axis and move them closer to the plot
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Until 2018', 'After 2018'], fontsize=12)
    ax.spines['bottom'].set_visible(False)
    # Modify this line to adjust the padding for x-axis labels
    ax.tick_params(axis='x', pad=-15)  # Reduce padding to bring x labels closer
    ax.tick_params(axis='x', which='both', length=0)  # This will remove the small tick lines on the x-axis

    # Add a title to the plot
    ax.set_title("Construct Movement Between Quadrants Over Time", fontsize=16, pad=0, fontweight='bold')

    # Remove unnecessary spines and gridlines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(False)

    # Adjust x-axis limits so the labels fit
    ax.set_xlim([-0.2, 1.2])

    # Add a legend on the right side for the quadrant colors and move it closer to the top
    legend_patches = [mpatches.Patch(color=quadrant_colors[i], label=f'Q{i + 1}') for i in range(4)]
    ax.legend(handles=legend_patches, title="Quadrants", loc='upper left', bbox_to_anchor=(0.90, 0.95), fontsize=12)

    fig_name = 'quadrant_movement.png'
    fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
    plt.close(fig)


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
        results.append({'construct': construct, 'quadrant_start': quadrant_A, 'quadrant_end': quadrant_B})

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file
    output_file = os.path.join(out_dir_path, f'quadrant_analysis_{x_metric}_vs_{y_metric}.csv')
    results_df.to_csv(output_file, index=False)

    logger.success(f"Quadrant analysis saved to {output_file}.")

    # Call plot with the results DataFrame, not the file path
    plot_custom_quadrant_flow_chart(results_df,out_dir_path)


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
            ax.scatter(subset_A[x_metric], subset_A[y_metric], color=extended_palette[i], marker='o', s=100,
                       edgecolor='w', label=construct)

            # Plot moment B (end point)
            ax.scatter(subset_B[x_metric], subset_B[y_metric], color=extended_palette[i], marker='o', s=100,
                       edgecolor='w')

            # Draw dashed line (without arrowhead) from moment A to moment B
            ax.plot([subset_A[x_metric].values[0], subset_B[x_metric].values[0]],
                    [subset_A[y_metric].values[0], subset_B[y_metric].values[0]], color=extended_palette[i],
                    linestyle='--', linewidth=0.75, dashes=(5, 10))

            # Add arrowhead (without line) pointing from A to B
            ax.annotate("", xy=(subset_B[x_metric].values[0], subset_B[y_metric].values[0]),  # arrow ends at B
                        xytext=(subset_A[x_metric].values[0], subset_A[y_metric].values[0]),  # starts at A
                        arrowprops=dict(arrowstyle="-|>", color=extended_palette[i], lw=1.0))

            # Add label next to the start point
            text = ax.text(subset_A[x_metric].values[0], subset_A[y_metric].values[0], construct, fontsize=8,
                           color='black', ha='left', va='top')
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
