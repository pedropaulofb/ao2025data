import os
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from icecream import ic
from loguru import logger


# Function to load data and plot the lines for each construct with optional smoothing and filtering
def plot_constructs_over_time(in_dir_path, out_dir_path, file_name, selected_constructs='all', window_size=1):
    # Load the CSV file
    csv_file = os.path.join(in_dir_path, file_name)
    df = pd.read_csv(csv_file, index_col='year')

    # Calculate the top and bottom 50% based on the maximum value of each construct
    construct_max = df.max().sort_values(ascending=False)
    num_constructs = len(construct_max)

    # Filter constructs based on 'selected_constructs' argument
    if selected_constructs == 'top':
        # Select top 50% constructs
        top_constructs = construct_max.index[:num_constructs // 2]
        df = df[top_constructs]
    elif selected_constructs == 'bottom':
        # Select bottom 50% constructs
        bottom_constructs = construct_max.index[num_constructs // 2:]
        df = df[bottom_constructs]
    # If 'all' is passed or no valid option, all constructs are used (default behavior)

    # Apply rolling mean if window_size is greater than 1
    if window_size > 1:
        df = df.rolling(window=window_size, min_periods=1).mean()

    # Reset index to have 'year' as a column for Seaborn plotting
    df_reset = df.reset_index()

    # Reshape the DataFrame from wide to long format
    df_melted = df_reset.melt(id_vars='year', var_name='Construct', value_name='Value')

    # Set plot style
    sns.set(style="whitegrid")

    # Line plot for the selected constructs
    fig, ax = plt.subplots(figsize=(16, 9))
    sns.lineplot(data=df_melted, x='year', y='Value', hue='Construct', palette="tab10", linewidth=2.5, ax=ax)

    # Determine if the file is "yearly" or "overall" based on the file name
    if "yearly" in file_name:
        normalization_type = "Yearly Normalized"
    elif "overall" in file_name:
        normalization_type = "Overall Normalized"
    else:
        normalization_type = "Unknown Normalization"

    # Customize plot title based on filtering, smoothing, and normalization type
    if window_size > 1:
        if selected_constructs == 'top':
            title = f'Top 50% Construct Proportions Over Time ({normalization_type}, Smoothed, Window: {window_size})'
        elif selected_constructs == 'bottom':
            title = f'Bottom 50% Construct Proportions Over Time ({normalization_type}, Smoothed, Window: {window_size})'
        else:
            title = f'Construct Proportions Over Time ({normalization_type}, Smoothed, Window: {window_size})'
    else:
        if selected_constructs == 'top':
            title = f'Top 50% Construct Proportions Over Time ({normalization_type})'
        elif selected_constructs == 'bottom':
            title = f'Bottom 50% Construct Proportions Over Time ({normalization_type})'
        else:
            title = f'Construct Proportions Over Time ({normalization_type})'

    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.set_ylabel('Proportion (%)')

    # Add legend and layout adjustment
    plt.legend(title='Constructs', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the figure instead of showing it
    fig_name = f"{file_name.replace('.csv', '')}_constructs_{selected_constructs}_window{window_size}.png"
    fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
    plt.close(fig)



# Function to load data and plot constructs divided into quartiles (25% each)
def plot_constructs_in_quartiles(in_dir_path, out_dir_path, file_name, window_size=1):
    # Load the CSV file
    csv_file = os.path.join(in_dir_path, file_name)
    df = pd.read_csv(csv_file, index_col='year')

    # Calculate the max value of each construct to determine the quartiles
    construct_max = df.max().sort_values(ascending=False)
    num_constructs = len(construct_max)

    # Divide the constructs into four groups (quartiles)
    top_25 = construct_max.index[:num_constructs // 4]
    second_25 = construct_max.index[num_constructs // 4:num_constructs // 2]
    third_25 = construct_max.index[num_constructs // 2:num_constructs * 3 // 4]
    bottom_25 = construct_max.index[num_constructs * 3 // 4:]

    quartile_groups = [
        ('Top 25%', top_25),
        ('Second Quartile (26-50%)', second_25),
        ('Third Quartile (51-75%)', third_25),
        ('Bottom 25%', bottom_25)
    ]

    # Set plot style
    sns.set(style="whitegrid")

    # Plot each quartile group
    for title, constructs in quartile_groups:
        # Filter the data to the selected constructs
        df_quartile = df[constructs]

        # Apply rolling mean if window_size is greater than 1
        if window_size > 1:
            df_quartile = df_quartile.rolling(window=window_size, min_periods=1).mean()

        # Reset index for plotting
        df_reset = df_quartile.reset_index()

        # Reshape the DataFrame for Seaborn plotting
        df_melted = df_reset.melt(id_vars='year', var_name='Construct', value_name='Value')

        # Plot each quartile
        fig, ax = plt.subplots(figsize=(16, 9))
        sns.lineplot(data=df_melted, x='year', y='Value', hue='Construct', palette="tab10", linewidth=2.5, ax=ax)

        # Determine if the file is "yearly" or "overall" based on the file name
        if "yearly" in file_name:
            normalization_type = "Yearly Normalized"
        elif "overall" in file_name:
            normalization_type = "Overall Normalized"
        else:
            normalization_type = "Unknown Normalization"

        # Customize plot title based on quartile, smoothing, and normalization type
        if window_size > 1:
            plot_title = f'{title} Construct Proportions Over Time ({normalization_type}, Smoothed, Window: {window_size})'
        else:
            plot_title = f'{title} Construct Proportions Over Time ({normalization_type})'

        ax.set_title(plot_title)
        ax.set_xlabel('Year')
        ax.set_ylabel('Proportion (%)')

        # Add legend and layout adjustment
        plt.legend(title='Constructs', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Save the figure instead of showing it
        fig_name = f"{file_name.replace('.csv', '')}_quartile_{title.replace(' ', '_')}_window{window_size}.png"
        fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
        logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
        plt.close(fig)

# Function to plot stacked bar chart
def plot_stacked_bar(in_dir_path, out_dir_path, file_name):
    # Load the CSV file
    csv_file = os.path.join(in_dir_path, file_name)
    df = pd.read_csv(csv_file, index_col='year')

    # Set up color palettes: 12 solid colors and 12 colors with texture ('.')
    solid_colors = sns.color_palette("tab20", 12)  # Use a seaborn palette for 12 solid colors
    textured_colors = sns.color_palette("tab20", 12)  # Another set of colors for textured

    # Create a list of patches for the legend
    legend_patches = []

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(16, 9))

    # Initialize a list to keep track of the bottom for the stacked bar plot
    bottom = [0] * len(df)

    # Loop through each construct and plot the bar chart
    for idx, construct in enumerate(df.columns):
        if idx < 12:
            # Plot the first 12 constructs with solid colors
            ax.bar(df.index, df[construct], bottom=bottom, color=solid_colors[idx], label=construct)
            legend_patches.append(mpatches.Patch(color=solid_colors[idx], label=construct))
        else:
            # Plot the next constructs with dots texture ('.') and the second color palette
            texture = '.'
            ax.bar(df.index, df[construct], bottom=bottom, color=textured_colors[idx - 12], hatch=texture, label=construct)
            legend_patches.append(mpatches.Patch(color=textured_colors[idx - 12], hatch=texture, label=construct))

        # Update the bottom for the next construct
        bottom = [i + j for i, j in zip(bottom, df[construct])]

    # Determine if the file is "yearly" or "overall" based on the file name
    if "yearly" in file_name:
        normalization_type = "Yearly Normalized"
    elif "overall" in file_name:
        normalization_type = "Overall Normalized"
    else:
        normalization_type = "Unknown Normalization"

    # Set labels and title
    ax.set_xlabel('Year')
    ax.set_ylabel('Proportion (%)')
    ax.set_title(f'Construct Proportions Over Time (Stacked Bar, {normalization_type})')

    # Ensure only integer years (no fractions) are shown on the x-axis
    ax.set_xticks(df.index)  # Set the x-axis ticks to the index (years)
    ax.set_xticklabels(df.index.astype(int), rotation=45, ha="right")  # Rotate the year labels for better readability

    # Add legend to the plot
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust the layout
    plt.tight_layout()

    # Save the figure
    fig_name = f"{file_name.replace('.csv', '')}_stacked_bar.png"
    fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
    plt.close(fig)


def plot_heatmap(in_dir_path, out_dir_path, file_name):
    # Load CSV file
    csv_file = os.path.join(in_dir_path, file_name)
    df = pd.read_csv(csv_file, index_col='year')

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(16, 9))
    sns.heatmap(df.T, cmap='coolwarm', ax=ax, annot=True, fmt=".1f", cbar_kws={'label': 'Proportion (%)'})

    ax.set_xlabel('Year')
    ax.set_ylabel('Construct')
    # Determine if the file is "yearly" or "overall" based on the file name
    if "yearly" in file_name:
        normalization_type = "Yearly Normalized"
    elif "overall" in file_name:
        normalization_type = "Overall Normalized"
    else:
        normalization_type = "Unknown Normalization"

    # Set plot title with normalization type
    ax.set_title(f'Construct Proportions Over Time (Heatmap, {normalization_type})')

    plt.tight_layout()

    # Save figure
    fig_name = f"{file_name.replace('.csv', '')}_heatmap.png"
    fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
    plt.close(fig)

# Function to load data and plot the bump chart for each construct
def plot_constructs_over_time_bump(in_dir_path, out_dir_path, file_name, selected_constructs='all', window_size=1):
    # Load the CSV file
    csv_file = os.path.join(in_dir_path, file_name)
    df = pd.read_csv(csv_file, index_col='year')

    # Calculate the top and bottom 50% based on the maximum value of each construct
    construct_max = df.max().sort_values(ascending=False)
    num_constructs = len(construct_max)

    # Filter constructs based on 'selected_constructs' argument
    if selected_constructs == 'top':
        # Select top 50% constructs
        top_constructs = construct_max.index[:num_constructs // 2]
        df = df[top_constructs]
    elif selected_constructs == 'bottom':
        # Select bottom 50% constructs
        bottom_constructs = construct_max.index[num_constructs // 2:]
        df = df[bottom_constructs]
    # If 'all' is passed or no valid option, all constructs are used (default behavior)

    # Apply rolling mean if window_size > 1
    if window_size > 1:
        df = df.rolling(window=window_size, min_periods=1).mean()

    # Rank the constructs for each year (1 = highest value, N = lowest value)
    df_ranks = df.rank(axis=1, method='dense', ascending=False)

    # Set plot style
    sns.set(style="whitegrid")

    # Plot bump chart
    fig, ax = plt.subplots(figsize=(16, 9))

    # Plot each construct's rank over time
    for construct in df_ranks.columns:
        ax.plot(df_ranks.index, df_ranks[construct], label=construct, linewidth=2)

    # Invert y-axis so that Rank 1 is at the top
    ax.invert_yaxis()

    # Determine if the file is "yearly" or "overall" based on the file name
    if "yearly" in file_name:
        normalization_type = "Yearly Normalized"
    elif "overall" in file_name:
        normalization_type = "Overall Normalized"
    else:
        normalization_type = "Unknown Normalization"

    # Customize plot title based on filtering, smoothing, and normalization type
    if window_size > 1:
        if selected_constructs == 'top':
            title = f'Top 50% Construct Rankings Over Time ({normalization_type}, Smoothed, Window: {window_size})'
        elif selected_constructs == 'bottom':
            title = f'Bottom 50% Construct Rankings Over Time ({normalization_type}, Smoothed, Window: {window_size})'
        else:
            title = f'Construct Rankings Over Time ({normalization_type}, Smoothed, Window: {window_size})'
    else:
        if selected_constructs == 'top':
            title = f'Top 50% Construct Rankings Over Time ({normalization_type})'
        elif selected_constructs == 'bottom':
            title = f'Bottom 50% Construct Rankings Over Time ({normalization_type})'
        else:
            title = f'Construct Rankings Over Time ({normalization_type})'

    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.set_ylabel('Rank')
    ax.legend(title='Constructs', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout
    plt.tight_layout()

    # Save the figure instead of showing it
    fig_name = f"{file_name.replace('.csv', '')}_bump_chart_{selected_constructs}_window{window_size}.png"
    fig.savefig(os.path.join(out_dir_path, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {out_dir_path}.")
    plt.close(fig)
