import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from src.create_figure_subdir import create_figures_subdir


def execute_visualization_group4(file_path):
    # Load the data from the CSV file

    df = pd.read_csv(file_path)
    save_dir = create_figures_subdir(file_path)

    # Rename the columns for shorter labels
    df.rename(columns={'Standard Deviation': 'SD', '25th Percentile (Q1)': 'Q1', '75th Percentile (Q3)': 'Q3',
                       'Interquartile Range (IQR)': 'IQR'}, inplace=True)

    # 1. Box Plot for Mean, Standard Deviation, and Variance
    # Convert the columns to numeric, in case they are read as strings
    numeric_columns = ['Mean', 'Median', 'Mode', 'SD', 'Variance', 'Skewness', 'Kurtosis', 'Q1', 'Q3', 'IQR']

    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Box Plot with Logarithmic Scale
    fig = fig = plt.figure(figsize=(12, 8), tight_layout=True)
    sns.boxplot(data=df[['Mean', 'SD', 'Variance']])
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.title('Box Plot for Mean, Standard Deviation, and Variance (Log Scale)')
    plt.ylabel('Log Scale Values')

    fig_name = 'group4_fig1.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")

    # 2. Box Plot for Q1, Q3, and IQR
    fig = fig = plt.figure(figsize=(12, 8), tight_layout=True)
    sns.boxplot(data=df[['Q1', 'Q3', 'IQR']], orient='v')
    plt.title('Box Plot: Q1, Q3, and IQR')
    plt.ylabel('Values')

    fig_name = 'group4_fig2.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")

    # 3. Heatmap of Correlation Between Measures
    # Convert the columns to numeric, in case they are read as strings

    # List of numeric columns with updated names
    numeric_columns = ['Mean', 'Median', 'SD', 'Variance', 'Skewness', 'Kurtosis', 'Q1', 'Q3', 'IQR']

    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Heatmap of Correlation Between Measures with Tilted Labels and Shortened Names
    fig = fig = plt.figure(figsize=(12, 8), tight_layout=True)
    correlation_matrix = df[numeric_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

    plt.title('Correlation Heatmap of Measures')
    plt.xticks(rotation=45)  # Tilt the x-axis labels by 45 degrees
    plt.yticks(rotation=45)  # Tilt the y-axis labels by 45 degrees

    fig_name = 'group4_fig3.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")

    # Apply log transformation to measures with a wide range
    df['Log_Variance'] = np.log1p(df['Variance'])

    # 4. Scatter Plot for Mean vs. Log_Variance with Regression Line
    # Apply log transformation to measures with a wide range
    df['Log_Variance'] = np.log1p(df['Variance'])

    # Plot: Scatter Plot for Mean vs. Log_Variance with Polynomial Regression Curve and Confidence Interval
    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    sns.regplot(data=df, x='Mean', y='Log_Variance', scatter_kws={'alpha': 0.7, 'label': 'Data Points'},
                # Add label for scatter points
                order=2, ci=95, line_kws={'color': 'red', 'label': 'Polynomial Fit (Degree 2)'}
                # Add label for regression line
                )
    plt.title('Scatter Plot with Polynomial Regression Curve and Confidence Interval: Mean vs. Log(Variance)')
    plt.xlabel('Mean')
    plt.ylabel('Log(Variance)')
    plt.grid(True)

    # Fit a polynomial of degree 2 to the data
    degree = 2
    coeffs = np.polyfit(df['Mean'], df['Log_Variance'], degree)
    p = np.poly1d(coeffs)

    # Construct the polynomial equation as a string
    equation = "y = " + " + ".join([f"{coeff:.3f}*x^{i}" for i, coeff in enumerate(coeffs[::-1])])

    # Display the equation on the plot
    plt.text(0.05, 0.9, equation, transform=plt.gca().transAxes, fontsize=10, color='red')

    plt.legend()  # Now, it should find the labeled elements
    fig_name = 'group4_fig4.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")

    # 5. Scatter Plot with Regression Line for Skewness vs. Kurtosis
    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    sns.regplot(x='Skewness', y='Kurtosis', data=df, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
    plt.title('Scatter Plot with Regression Line: Skewness vs. Kurtosis')
    plt.xlabel('Skewness')
    plt.ylabel('Kurtosis')
    fig_name = 'group4_fig5.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")

    # 6. Bar Chart for Mean Values Across Constructs

    # Calculate the Standard Error of the Mean (SEM)
    df['SEM'] = df['SD'] / np.sqrt(1)  # Since count = 1 for each 'Construct', SEM = SD / sqrt(1)

    # Calculate asymmetric error bars, ensuring they do not go below zero
    lower_error = np.where(df['Mean'] - df['SEM'] < 0, df['Mean'],
                           df['SEM'])  # Adjust lower error to prevent negative values
    upper_error = df['SEM']

    # Plot: Bar Chart for Mean Values Across Constructs with Error Bars
    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    sns.barplot(x='Construct', y='Mean', data=df, hue='Construct',  # Use 'hue' to match 'x' variable
                palette='viridis', errorbar=None,  # Replace deprecated 'ci' with 'errorbar=None'
                legend=False  # Avoid adding a legend
                )

    # Add custom error bars using asymmetric error bars
    plt.errorbar(x=df['Construct'], y=df['Mean'], yerr=[lower_error, upper_error],
                 # Asymmetric error bars: lower and upper limits
                 fmt='none',  # No markers, just error bars
                 ecolor='gray',  # Error bar color
                 elinewidth=2,  # Error bar line width
                 capsize=5  # Size of the error bar caps
                 )

    # Customize x-axis labels
    ax = plt.gca()  # Get the current axis
    x_labels = ax.get_xticklabels()  # Get x-axis labels

    # Loop through each x-axis label and change color for 'none' and 'other'
    for label in x_labels:
        if label.get_text() == 'none':
            label.set_color('blue')  # Set color to blue for 'none'
        elif label.get_text() == 'other':
            label.set_color('red')  # Set color to red for 'other'

    plt.title('Mean Values Across Constructs with SD Error Bars', fontsize=14, fontweight='bold')
    plt.xlabel('Construct Type', fontsize=12)
    plt.ylabel('Mean Value with SD Error Bars', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig_name = 'group4_fig6.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")

    # 5. Plot for Distribution of Skewness and Kurtosis
    # Create the figure and axis
    fig, ax1 = plt.subplots(figsize=(12, 8), tight_layout=True)

    # Plot the bar chart for Skewness
    sns.barplot(x='Construct', y='Skewness', data=df, ax=ax1, hue='Construct',  # Use 'hue' to match 'x' variable
                palette='viridis', errorbar=None,  # Replace deprecated 'ci' with 'errorbar=None'
                legend=False  # Avoid adding a legend
                )

    # Customize the first y-axis for Skewness
    ax1.set_ylabel('Skewness', fontsize=12)
    ax1.set_xlabel('Construct', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title('Skewness and Kurtosis Across Constructs', fontsize=14, fontweight='bold')

    # Create a second y-axis for Kurtosis
    ax2 = ax1.twinx()

    # Plot the line chart for Kurtosis
    sns.lineplot(x='Construct', y='Kurtosis', data=df, ax=ax2, color='red', marker='o', linewidth=2, label='Kurtosis')

    # Customize the second y-axis for Kurtosis
    ax2.set_ylabel('Kurtosis', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Combine legends for both plots
    ax1.legend(['Skewness'], loc='upper left')
    ax2.legend(loc='upper right')

    # Set the x-axis tick positions to ensure a fixed number of ticks
    ax1.set_xticks(ax1.get_xticks())  # Fix the ticks before setting their labels

    # Now set the rotation and alignment for x-axis tick labels on the first axis (ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, ha='right')

    # Loop through each x-axis label and change color for 'none' and 'other'
    x_labels = ax1.get_xticklabels()
    for label in x_labels:
        if label.get_text() == 'none':
            label.set_color('blue')  # Set color to blue for 'none'
        elif label.get_text() == 'other':
            label.set_color('red')  # Set color to red for 'other'

    plt.tight_layout()
    fig_name = 'group4_fig7.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")

    # 8. Density Plot for Skewness and Kurtosis
    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    sns.kdeplot(df['Skewness'], fill=True, color='blue', label='Skewness')  # Use 'fill=True' instead of 'shade=True'
    sns.kdeplot(df['Kurtosis'], fill=True, color='red', label='Kurtosis')  # Use 'fill=True' instead of 'shade=True'
    plt.title('Density Plot for Skewness and Kurtosis')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    fig_name = 'group4_fig8.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")

    # 9. Radar Charts
    # Select top 5 constructs by mean value
    top_constructs = df.nlargest(5, 'Mean')

    # 9.1. Radar chart for Group 1: Central Tendency Analysis
    central_tendency_columns = ['Mean', 'Median', 'IQR']
    df_radar_central_tendency = top_constructs[central_tendency_columns]
    df_radar_central_tendency_normalized = (df_radar_central_tendency - df_radar_central_tendency.min()) / (
            df_radar_central_tendency.max() - df_radar_central_tendency.min())

    # Number of variables for this group
    num_vars_central_tendency = len(central_tendency_columns)
    angles_central_tendency = [n / float(num_vars_central_tendency) * 2 * np.pi for n in
                               range(num_vars_central_tendency)]
    angles_central_tendency += angles_central_tendency[:1]

    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax1 = plt.subplot(111, polar=True)
    for index, row in df_radar_central_tendency_normalized.iterrows():
        values = row.tolist()
        values += values[:1]
        construct_label = top_constructs.loc[index, 'Construct']
        ax1.plot(angles_central_tendency, values, label=construct_label)
        ax1.fill(angles_central_tendency, values, alpha=0.1)

    plt.xticks(angles_central_tendency[:-1], central_tendency_columns)
    plt.title('Radar Chart: Central Tendency Analysis')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    fig_name = 'group4_fig9.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")

    # 9.2. Radar chart for Group 2: Variability and Spread
    spread_columns = ['SD', 'Variance', 'IQR', 'Q1', 'Q3']
    df_radar_spread = top_constructs[spread_columns]
    df_radar_spread_normalized = (df_radar_spread - df_radar_spread.min()) / (
            df_radar_spread.max() - df_radar_spread.min())

    num_vars_spread = len(spread_columns)
    angles_spread = [n / float(num_vars_spread) * 2 * np.pi for n in range(num_vars_spread)]
    angles_spread += angles_spread[:1]

    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax2 = plt.subplot(111, polar=True)
    for index, row in df_radar_spread_normalized.iterrows():
        values = row.tolist()
        values += values[:1]
        construct_label = top_constructs.loc[index, 'Construct']
        ax2.plot(angles_spread, values, label=construct_label)
        ax2.fill(angles_spread, values, alpha=0.1)

    plt.xticks(angles_spread[:-1], spread_columns)
    plt.title('Radar Chart: Variability and Spread Analysis')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    fig_name = 'group4_fig10.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")

    # 9.3. Radar chart for Group 3: Comprehensive Spread and Shape Analysis
    comprehensive_columns = ['Mean', 'SD', 'Skewness', 'Kurtosis', 'IQR']
    df_radar_comprehensive = top_constructs[comprehensive_columns]
    df_radar_comprehensive_normalized = (df_radar_comprehensive - df_radar_comprehensive.min()) / (
            df_radar_comprehensive.max() - df_radar_comprehensive.min())

    num_vars_comprehensive = len(comprehensive_columns)
    angles_comprehensive = [n / float(num_vars_comprehensive) * 2 * np.pi for n in range(num_vars_comprehensive)]
    angles_comprehensive += angles_comprehensive[:1]

    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax4 = plt.subplot(111, polar=True)
    for index, row in df_radar_comprehensive_normalized.iterrows():
        values = row.tolist()
        values += values[:1]
        construct_label = top_constructs.loc[index, 'Construct']
        ax4.plot(angles_comprehensive, values, label=construct_label)
        ax4.fill(angles_comprehensive, values, alpha=0.1)

    plt.xticks(angles_comprehensive[:-1], comprehensive_columns)
    plt.title('Radar Chart: Comprehensive Spread and Shape Analysis')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    fig_name = 'group4_fig11.png'
    fig.savefig(os.path.join(save_dir, fig_name), dpi=300)
    logger.success(f"Figure {fig_name} successfully saved in {save_dir}.")


execute_visualization_group4('../outputs/analyses/cs_analyses/central_tendency_dispersion.csv')
