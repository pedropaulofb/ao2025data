# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from backup.src.utils import color_text


# Function to filter out stereotypes that appear too infrequently
def filter_infrequent_stereotypes(df, frequency_df, group_frequency_df, threshold=0.05):
    """
    Filter stereotypes based on a fixed threshold for total occurrences and group-wise occurrences.
    """

    # Calculate the 5% threshold for occurrences
    total_occurrences = frequency_df['Frequency'].sum()
    occurrence_threshold = threshold * total_occurrences  # % of total occurrences

    # Calculate the 5% threshold for group-wise occurrences
    total_groups = group_frequency_df['Group-wise Frequency'].sum()
    group_occurrence_threshold = threshold * total_groups  # % of group-wise occurrences

    # Filter stereotypes based on occurrence-wise frequency
    frequent_stereotypes_occurrence = frequency_df[frequency_df['Frequency'] >= occurrence_threshold]['Stereotype']

    # Filter stereotypes based on group-wise frequency
    frequent_stereotypes_group = \
        group_frequency_df[group_frequency_df['Group-wise Frequency'] >= group_occurrence_threshold]['Stereotype']

    # Keep only stereotypes that meet both criteria
    frequent_stereotypes = set(frequent_stereotypes_occurrence).intersection(set(frequent_stereotypes_group))

    # Filter the main DataFrame to keep only frequent stereotypes
    df_filtered = df[df['Stereotype'].isin(frequent_stereotypes)].dropna(subset=['Stereotype'])

    return df_filtered


def calculate_and_rank_distances(pca_df, num_components=2):
    """
    Function to calculate distances from the origin and rank stereotypes.
    """
    if num_components == 2:
        # Calculate 2D distances (PC1, PC2)
        pca_df['Distance from Origin (2D)'] = np.sqrt(pca_df['PC1'] ** 2 + pca_df['PC2'] ** 2)
    elif num_components == 3:
        # Calculate 3D distances (PC1, PC2, PC3)
        pca_df['Distance from Origin (3D)'] = np.sqrt(pca_df['PC1'] ** 2 + pca_df['PC2'] ** 2 + pca_df['PC3'] ** 2)

    return pca_df


def plot_rank_plot(pca_df, distance_column, num_components=2):
    """
    Function to create a rank plot for stereotypes based on distances.
    """
    # Sort the stereotypes by the calculated distance
    ranked_pca_df = pca_df.sort_values(by=distance_column, ascending=False)

    # Set up the figure
    plt.figure(figsize=(10, 6))

    # Create the rank plot without the palette (color applied uniformly)
    sns.barplot(x=distance_column, y='Stereotype', data=ranked_pca_df, color='lightblue')

    # Set labels and title
    plt.title(f'Rank Plot: Stereotypes by Distance from Origin ({num_components}D)', fontsize=14)
    plt.xlabel('Distance from Origin')
    plt.ylabel('Stereotype')

    # Apply color to specific y-tick labels (stereotypes)
    color_text(plt.gca().get_yticklabels())  # Apply color_text function to the y-axis labels

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_pca_2d(file_path_central_tendency, file_path_diversity, file_path_mi, file_path_corr, file_path_frequency,
                file_path_group_frequency):
    # Load the CSV files into pandas DataFrames
    df_central_tendency = pd.read_csv(file_path_central_tendency)
    df_diversity = pd.read_csv(file_path_diversity)
    df_mi = pd.read_csv(file_path_mi)
    df_corr = pd.read_csv(file_path_corr)

    # Merge the central tendency and diversity measures by 'Stereotype'
    df = pd.merge(df_central_tendency, df_diversity, on='Stereotype', how='inner')

    # Add the average mutual information and average Spearman correlation for each stereotype
    df_mi['Avg Mutual Information'] = df_mi.drop(columns=['Stereotype']).mean(axis=1)
    df_corr['Avg Spearman Correlation'] = df_corr.drop(columns=['Stereotype']).mean(axis=1)

    # Merge with both 'Stereotype' and 'Avg Mutual Information'
    df = pd.merge(df, df_mi[['Stereotype', 'Avg Mutual Information']], on='Stereotype', how='inner')

    # Merge with both 'Stereotype' and 'Avg Spearman Correlation'
    df = pd.merge(df, df_corr[['Stereotype', 'Avg Spearman Correlation']], on='Stereotype', how='inner')

    # Load frequency data
    frequency_df = pd.read_csv(file_path_frequency)
    group_frequency_df = pd.read_csv(file_path_group_frequency)

    # Apply filtering based on % occurrence thresholds
    df = filter_infrequent_stereotypes(df, frequency_df, group_frequency_df, 0.01)

    # Define the final features to use in PCA (central tendency, diversity, and dependency measures)
    features = ['Standard Deviation', 'Skewness', 'Kurtosis', 'Interquartile Range (IQR)', 'Shannon Entropy',
                'Gini Coefficient', 'Simpson Index', 'Avg Mutual Information', 'Avg Spearman Correlation']

    # Separate out the 'Stereotype' column for labeling later
    stereotypes = df['Stereotype'].dropna()  # Remove any NaN values in 'Stereotype'

    # Standardize the features before applying PCA
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])

    # Perform PCA, reduce dimensions to 2 principal components for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)

    # Create a new DataFrame with the PCA results and stereotype labels
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df['Stereotype'] = stereotypes.reset_index(drop=True)  # Reset index to avoid mismatches

    # Set up the figure
    plt.figure(figsize=(16, 9))

    # Plot the PCA results using seaborn with a scatter plot
    sns.scatterplot(x='PC1', y='PC2', hue='Stereotype', data=pca_df, palette='Set1', s=100, legend=False)

    # Add dashed gray lines for the origin (0,0) cross
    plt.axhline(0, color='lightgray', linestyle='--', linewidth=1)
    plt.axvline(0, color='lightgray', linestyle='--', linewidth=1)

    # Add labels for each stereotype in the scatter plot without a square box
    texts = []
    for i in range(pca_df.shape[0]):
        text_obj = plt.text(x=pca_df['PC1'][i] + 0.05, y=pca_df['PC2'][i], s=pca_df['Stereotype'][i],
                            fontdict=dict(color='black', size=10), alpha=0.7)
        texts.append(text_obj)  # Collect the text objects

    # Adjust text positions to avoid overlap
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))
    color_text(texts)

    # Add labels and title
    plt.title('PCA Biplot: Core Elements of the Language (Filtered)', fontsize=14)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0] * 100:.2f}% Variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1] * 100:.2f}% Variance)')

    # Adjust the layout so everything fits well
    plt.tight_layout()

    # Show the plot
    plt.show()

    pca_df_2d = calculate_and_rank_distances(pca_df[['PC1', 'PC2', 'Stereotype']], num_components=2)
    plot_rank_plot(pca_df_2d, 'Distance from Origin (2D)', num_components=2)


def plot_pca_3d(file_path_central_tendency, file_path_diversity, file_path_mi, file_path_corr, file_path_frequency,
                file_path_group_frequency):
    # Load the CSV files into pandas DataFrames
    df_central_tendency = pd.read_csv(file_path_central_tendency)
    df_diversity = pd.read_csv(file_path_diversity)
    df_mi = pd.read_csv(file_path_mi)
    df_corr = pd.read_csv(file_path_corr)

    # Merge the central tendency and diversity measures by 'Stereotype'
    df = pd.merge(df_central_tendency, df_diversity, on='Stereotype', how='inner')

    # Add the average mutual information and average Spearman correlation for each stereotype
    df_mi['Avg Mutual Information'] = df_mi.drop(columns=['Stereotype']).mean(axis=1)
    df_corr['Avg Spearman Correlation'] = df_corr.drop(columns=['Stereotype']).mean(axis=1)

    # Merge with the main dataframe
    df = pd.merge(df, df_mi[['Stereotype', 'Avg Mutual Information']], on='Stereotype', how='inner')
    df = pd.merge(df, df_corr[['Stereotype', 'Avg Spearman Correlation']], on='Stereotype', how='inner')

    # Load frequency data
    frequency_df = pd.read_csv(file_path_frequency)
    group_frequency_df = pd.read_csv(file_path_group_frequency)

    # Apply filtering based on % occurrence thresholds
    df = filter_infrequent_stereotypes(df, frequency_df, group_frequency_df, 0.01)

    # Define the final features to use in PCA (central tendency, diversity, and dependency measures)
    features = ['Standard Deviation', 'Skewness', 'Kurtosis', 'Interquartile Range (IQR)', 'Shannon Entropy',
                'Gini Coefficient', 'Simpson Index', 'Avg Mutual Information', 'Avg Spearman Correlation']

    # Standardize the features before applying PCA
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])

    # Perform PCA, reduce dimensions to 3 principal components for visualization
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(scaled_features)

    # Create a new DataFrame with the PCA results and stereotype labels
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])
    # Ensure to drop any NaN values after filtering and reset index
    pca_df['Stereotype'] = df['Stereotype'].dropna().reset_index(drop=True)  # Reset index to avoid mismatch
    pca_df = pca_df.dropna().reset_index(
        drop=True)  # Drop rows with any NaN values in 'PC1', 'PC2', 'PC3', or 'Stereotype' and reset index

    # Set up the 3D figure (for the 3D plot)
    fig_3d = plt.figure(figsize=(16, 9))
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    # Plot the PCA results using a 3D scatter plot
    ax_3d.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=np.random.rand(len(pca_df)), s=100, cmap='Set1')

    # Add labels for each stereotype in the 3D scatter plot
    for i in range(pca_df.shape[0]):
        ax_3d.text(pca_df['PC1'][i], pca_df['PC2'][i], pca_df['PC3'][i], pca_df['Stereotype'][i], size=10, zorder=1)

    # Set labels for axes
    ax_3d.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0] * 100:.2f}% Variance)')
    ax_3d.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1] * 100:.2f}% Variance)')
    ax_3d.set_zlabel(f'Principal Component 3 ({pca.explained_variance_ratio_[2] * 100:.2f}% Variance)')

    # Set title
    ax_3d.set_title('3D PCA with Central Tendency, Diversity, and Dependency Measures', fontsize=14)

    # Show the 3D plot
    plt.show()

    # Create a new figure for the 3 2D subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Prepare lists to store the text objects
    texts_0 = []
    texts_1 = []
    texts_2 = []

    # Plot PC1 vs PC2
    sns.scatterplot(x='PC1', y='PC2', data=pca_df, ax=axes[0], hue='Stereotype', palette='Set1', s=100, legend=False)
    for i in range(pca_df.shape[0]):
        text_obj = axes[0].text(pca_df['PC1'][i], pca_df['PC2'][i], pca_df['Stereotype'][i], size=10, alpha=0.75)
        texts_0.append(text_obj)  # Append the text object to the list
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% Variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% Variance)')
    axes[0].set_title('PC1 vs PC2')

    # Plot PC1 vs PC3
    sns.scatterplot(x='PC1', y='PC3', data=pca_df, ax=axes[1], hue='Stereotype', palette='Set1', s=100, legend=False)
    for i in range(pca_df.shape[0]):
        text_obj = axes[1].text(pca_df['PC1'][i], pca_df['PC3'][i], pca_df['Stereotype'][i], size=10, alpha=0.75)
        texts_1.append(text_obj)  # Append the text object to the list
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% Variance)')
    axes[1].set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2] * 100:.2f}% Variance)')
    axes[1].set_title('PC1 vs PC3')

    # Plot PC2 vs PC3
    sns.scatterplot(x='PC2', y='PC3', data=pca_df, ax=axes[2], hue='Stereotype', palette='Set1', s=100, legend=False)
    for i in range(pca_df.shape[0]):
        text_obj = axes[2].text(pca_df['PC2'][i], pca_df['PC3'][i], pca_df['Stereotype'][i], size=10, alpha=0.75)
        texts_2.append(text_obj)  # Append the text object to the list
    axes[2].set_xlabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% Variance)')
    axes[2].set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2] * 100:.2f}% Variance)')
    axes[2].set_title('PC2 vs PC3')

    # For PC1 vs PC2
    axes[0].axhline(0, color='lightgray', linestyle='--', linewidth=1)
    axes[0].axvline(0, color='lightgray', linestyle='--', linewidth=1)

    # For PC1 vs PC3
    axes[1].axhline(0, color='lightgray', linestyle='--', linewidth=1)
    axes[1].axvline(0, color='lightgray', linestyle='--', linewidth=1)

    # For PC2 vs PC3
    axes[2].axhline(0, color='lightgray', linestyle='--', linewidth=1)
    axes[2].axvline(0, color='lightgray', linestyle='--', linewidth=1)

    # Adjust the text labels to avoid overlap
    adjust_text(texts_0, ax=axes[0])
    adjust_text(texts_1, ax=axes[1])
    adjust_text(texts_2, ax=axes[2])

    # Apply custom coloring to the text labels
    color_text(texts_0)
    color_text(texts_1)
    color_text(texts_2)

    # Adjust the layout for the 2D subplots
    plt.tight_layout()

    # Show the 2D subplots
    plt.show()

    # Example for 3D (PC1, PC2, PC3)
    pca_df_3d = calculate_and_rank_distances(pca_df[['PC1', 'PC2', 'PC3', 'Stereotype']], num_components=3)
    plot_rank_plot(pca_df_3d, 'Distance from Origin (3D)', num_components=3)


# Example usage
plot_pca_2d("../outputs/statistics/cs_ontouml_no_classroom_f/central_tendency_dispersion.csv",
            "../outputs/statistics/cs_ontouml_no_classroom_f/diversity_measures.csv",
            "../outputs/statistics/cs_ontouml_no_classroom_f/mutual_information.csv",
            "../outputs/statistics/cs_ontouml_no_classroom_f/spearman_correlation.csv",
            "../outputs/statistics/cs_ontouml_no_classroom_f/rank_frequency_distribution.csv",
            "../outputs/statistics/cs_ontouml_no_classroom_f/rank_groupwise_frequency_distribution.csv")

plot_pca_3d("../outputs/statistics/cs_ontouml_no_classroom_f/central_tendency_dispersion.csv",
            "../outputs/statistics/cs_ontouml_no_classroom_f/diversity_measures.csv",
            "../outputs/statistics/cs_ontouml_no_classroom_f/mutual_information.csv",
            "../outputs/statistics/cs_ontouml_no_classroom_f/spearman_correlation.csv",
            "../outputs/statistics/cs_ontouml_no_classroom_f/rank_frequency_distribution.csv",
            "../outputs/statistics/cs_ontouml_no_classroom_f/rank_groupwise_frequency_distribution.csv")
