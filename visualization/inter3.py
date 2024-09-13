import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from the CSV files
rank_df = pd.read_csv('../outputs/analyses/cs_analyses/rank_frequency_distribution.csv')
mutual_info_df = pd.read_csv('../outputs/analyses/cs_analyses/mutual_information.csv')

# Calculate the average mutual information for each construct
mutual_info_avg = mutual_info_df.drop(columns='Construct').mean(axis=1)
mutual_info_df['Avg Mutual Information'] = mutual_info_avg

# Merge rank data with mutual information data
merged_data = pd.merge(rank_df[['Construct', 'Rank']],
                       mutual_info_df[['Construct', 'Avg Mutual Information']],
                       on='Construct')

# Plot the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_data,
                x='Rank',
                y='Avg Mutual Information',
                hue='Construct',
                palette='viridis',
                size='Avg Mutual Information',
                sizes=(20, 200),
                legend=False)

# Customize the plot
plt.title('Scatter Plot of Rank vs. Mutual Information')
plt.xlabel('Rank')
plt.ylabel('Average Mutual Information')
plt.grid(True)

# Annotate each point with the construct name
for i in range(len(merged_data)):
    plt.text(x=merged_data['Rank'][i],
             y=merged_data['Avg Mutual Information'][i],
             s=merged_data['Construct'][i],
             fontsize=8,
             ha='center')

# Show the plot
plt.show()
