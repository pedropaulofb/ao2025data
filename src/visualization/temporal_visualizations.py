import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger


# Function to load data and plot the lines for each construct with optional smoothing and filtering
def plot_constructs_over_time(in_dir_path, out_dir_path, file_name, selected_constructs='all', window_size=1):
    # Load the CSV file
    csv_file = os.path.join(in_dir_path, file_name)
    df = pd.read_csv(csv_file, index_col='year')

    # Normalize the data to percentages (convert raw values to percentages per year)
    df = df.div(df.sum(axis=1), axis=0) * 100

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

    # Customize plot title based on filtering and smoothing
    if window_size > 1:
        if selected_constructs == 'top':
            title = f'Top 50% Construct Proportions Over Time (Smoothed, Window: {window_size})'
        elif selected_constructs == 'bottom':
            title = f'Bottom 50% Construct Proportions Over Time (Smoothed, Window: {window_size})'
        else:
            title = f'Construct Proportions Over Time (Smoothed, Window: {window_size})'
    else:
        if selected_constructs == 'top':
            title = 'Top 50% Construct Proportions Over Time'
        elif selected_constructs == 'bottom':
            title = 'Bottom 50% Construct Proportions Over Time'
        else:
            title = 'Construct Proportions Over Time'

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

    # Normalize the data to percentages (convert raw values to percentages per year)
    df = df.div(df.sum(axis=1), axis=0) * 100

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
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=df_melted, x='year', y='Value', hue='Construct', palette="tab10", linewidth=2.5, ax=ax)

        # Customize plot title based on quartile and smoothing
        if window_size > 1:
            plot_title = f'{title} Construct Proportions Over Time (Smoothed, Window: {window_size})'
        else:
            plot_title = f'{title} Construct Proportions Over Time'

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