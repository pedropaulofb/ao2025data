import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.font_manager import FontProperties

from src.color_legend import color_text
from src.create_figure_subdir import create_figures_subdir


def execute_visualization_frequency_analysis(file_path):
    df = pd.read_csv(file_path)
    save_dir = create_figures_subdir(file_path)

    # Convert the 'Construct' column to a categorical type to ensure proper ordering in the plots
    df['Construct'] = pd.Categorical(df['Construct'], categories=df['Construct'].unique(), ordered=True)

    ### 1) Side-by-Side Bar Chart with Double Axis for Total Frequency and Ubiquity Index

    fig, ax1 = plt.subplots(figsize=(16, 9), tight_layout=True)

    # Define the x positions
    x = range(len(df))

    # Plot Total Frequency on the primary y-axis
    ax1.bar([i - 0.2 for i in x], df['Total Frequency'], color='#254c94', width=0.4, label='Total Frequency')
    ax1.set_ylabel('Total Construct Frequency', color='#254c94')
    ax1.tick_params(axis='y', labelcolor='#254c94')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Construct'], rotation=90)

    # Color specific x-axis labels
    color_text(ax1.get_xticklabels())

    # Create a secondary y-axis for Ubiquity Index
    ax2 = ax1.twinx()
    ax2.bar([i + 0.2 for i in x], df['Ubiquity Index (Group Frequency per Group)'], color='#d9403d', width=0.4,
            label='Ubiquity Index')
    ax2.set_ylabel('Ubiquity Index (Frequency Across Groups)', color='#d9403d')
    ax2.tick_params(axis='y', labelcolor='#d9403d')

    plt.title('Total Frequency and Ubiquity Index by Construct', fontweight='bold')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    fig.tight_layout()

    fig_name = 'total_frequency_vs_ubiquity_index_by_construct.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")
    plt.close(fig)

    ### 2) Side-by-Side Bar Chart with Double Axis for Total Frequency vs. Group Frequency

    fig, ax1 = plt.subplots(figsize=(16, 9), tight_layout=True)

    # Plot Total Frequency on the primary y-axis
    x = range(len(df))
    ax1.bar(x, df['Total Frequency'], color='#254c94', width=0.4, label='Total Frequency')
    ax1.set_ylabel('Total Frequency', color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.set_xticks([i + 0.2 for i in x])
    ax1.set_xticklabels(df['Construct'], rotation=90)

    # Color specific x-axis labels
    color_text(ax1.get_xticklabels())

    # Create a secondary y-axis for Group Frequency
    ax2 = ax1.twinx()
    ax2.bar([i + 0.4 for i in x], df['Group Frequency'], color='orange', width=0.4, label='Group Frequency')
    ax2.set_ylabel('Frequency of Constructs in Groups', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.title('Total Frequency vs. Group Frequency Across Constructs', fontweight='bold')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    fig.tight_layout()

    fig_name = 'total_frequency_vs_group_frequency_across_constructs.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")
    plt.close(fig)

    ### 3) Scatter plot for Ubiquity Index vs. Total Frequency per Group

    # Constructs in the top-right quadrant (high on both axes) are those that are not only frequently used in the models where they appear but are also used in many different models. These are likely the most "core" or "essential" constructs, which are both common and widely adopted by modelers.
    # Constructs in the bottom-right quadrant (high "Total Frequency per Group" but low "Ubiquity Index") are used heavily when they do appear but only in a small number of models. This might suggest constructs with specialized or niche use cases.
    # Constructs in the top-left quadrant (low "Total Frequency per Group" but high "Ubiquity Index") are used in many different models but not very frequently within each model. This could indicate constructs that are necessary for a broad range of models but are used sparingly.
    # Constructs in the bottom-left quadrant (low on both axes) are used infrequently and in few models, potentially indicating constructs with limited relevance or applicability.

    fig = plt.figure(figsize=(16, 9), tight_layout=True)

    # Define the base colors for the plot (12 distinct colors)
    base_palette = sns.color_palette('tab10', n_colors=12)

    # Extend the palette to handle all 23 categories by repeating colors
    extended_palette = base_palette + base_palette[:11]  # 12 colors + 11 more to make 23 total

    # Define marker types (12 circles 'o' and 11 squares 's')
    markers = ['o'] * 12 + ['s'] * 11

    # Plotting the scatter plot with different colors and markers
    for i, construct in enumerate(df['Construct'].unique()):
        subset = df[df['Construct'] == construct]
        plt.scatter(subset['Total Frequency per Group'], subset['Ubiquity Index (Group Frequency per Group)'],
                    color=extended_palette[i], marker=markers[i], s=100, edgecolor='w', label=construct)

    # Adding labels and title
    plt.xlabel('Total Construct Frequency per Group')
    plt.ylabel('Ubiquity Index (Diversity Across Models)')
    plt.title('Construct Usage: Ubiquity Index vs. Total Frequency per Group', fontweight='bold')

    # Adding a cross to separate the plot into four quadrants
    plt.axhline(y=df['Ubiquity Index (Group Frequency per Group)'].mean(), color='black', linestyle='--', linewidth=1)
    plt.axvline(x=df['Total Frequency per Group'].mean(), color='black', linestyle='--', linewidth=1)

    # Customize the legend to color 'none' and 'other'
    bold_font = FontProperties(weight='bold')
    legend = plt.legend(title='Constructs', title_fontproperties=bold_font, bbox_to_anchor=(1.02, 1), loc='upper left')

    color_text(legend.get_texts())

    plt.tight_layout()
    fig_name = 'construct_usage_ubiquity_index_vs_total_frequency.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")
    plt.close(fig)

    ### 4) Side-by-Side Donut Charts for Occurrence-wise and Group-wise Relative Frequencies with Colors and Dots Texture

    # Calculate the percentages
    percent_occurrence = df['Global Relative Frequency (Occurrence-wise)'] * 100
    percent_group = df['Global Relative Frequency (Group-wise)'] * 100

    # Prepare labels with percentages
    labels_occurrence = [f'{label}: {percentage:.1f}%' for label, percentage in
                         zip(df['Construct'], percent_occurrence)]
    labels_group = [f'{label}: {percentage:.1f}%' for label, percentage in zip(df['Construct'], percent_group)]

    # Create a palette with 12 distinct colors
    base_palette = sns.color_palette('tab10',
                                     n_colors=12)  # 'tab10' provides 10 colors; use 'tab12' or any suitable palette
    extended_palette = base_palette + base_palette  # Repeat the 12 colors to handle 24 categories

    # Define the hatch pattern (dots) for every other category
    hatches = ['' if i < 12 else '...' for i in range(24)]  # Apply dots for the second set of 12 colors

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), tight_layout=True)

    # Donut chart for Occurrence-wise relative frequency
    wedges, texts, autotexts = ax1.pie(df['Global Relative Frequency (Occurrence-wise)'],
                                       labels=['' for _ in df['Construct']],  # Empty labels to avoid clutter
                                       autopct='', startangle=140, colors=extended_palette, wedgeprops=dict(width=0.4))

    # Apply dot texture to every other wedge
    for i, wedge in enumerate(wedges):
        wedge.set_hatch(hatches[i % 24])  # Use the hatch pattern for every other wedge

    # Add custom legend with colored text
    bold_font = FontProperties(weight='bold')
    legend_occurrence = ax1.legend(wedges, labels_occurrence, title="Constructs", loc="center left",
                                   title_fontproperties=bold_font, bbox_to_anchor=(1, 0, 0.5, 1))

    color_text(legend_occurrence.get_texts())

    ax1.set_title('Relative Frequency of Constructs by Occurrence', fontweight='bold')

    # Donut chart for Group-wise relative frequency
    wedges, texts, autotexts = ax2.pie(df['Global Relative Frequency (Group-wise)'],
                                       labels=['' for _ in df['Construct']],  # Empty labels to avoid clutter
                                       autopct='', startangle=140, colors=extended_palette, wedgeprops=dict(width=0.4))

    # Apply dot texture to every other wedge
    for i, wedge in enumerate(wedges):
        wedge.set_hatch(hatches[i % 24])  # Use the hatch pattern for every other wedge

    # Add custom legend with colored text
    legend_group = ax2.legend(wedges, labels_group, title="Constructs", loc="center left",
                              title_fontproperties=bold_font, bbox_to_anchor=(1, 0, 0.5, 1))
    color_text(legend_group.get_texts())

    ax2.set_title('Relative Frequency of Constructs by Group Usage', fontweight='bold')

    plt.tight_layout()
    fig_name = 'relative_frequency_of_constructs_by_occurrence_and_group_usage.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")
    plt.close(fig)
