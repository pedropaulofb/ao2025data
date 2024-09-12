import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
data = pd.read_csv("../outputs/analyses/cs_analyses/rank_frequency_distribution.csv")

# Calculate the percentage frequency for the bar plot
data['Percentage Frequency'] = (data['Frequency'] / data['Frequency'].sum()) * 100

# 1. Pareto Chart for Rank-Frequency and Cumulative Percentage
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for Rank-Percentage Frequency with uniform color
sns.barplot(x='Construct', y='Percentage Frequency', data=data, ax=ax1, color='skyblue')
ax1.set_xlabel('Construct')
ax1.set_ylabel('Occurrence-wise Relative Frequency (%)')
ax1.set_title('Pareto Chart for Rank-Percentage Frequency and Cumulative Percentage')

# Rotate x-axis labels correctly and apply color to specific labels
for label in ax1.get_xticklabels():
    if label.get_text() == 'none':
        label.set_color('blue')
    elif label.get_text() == 'other':
        label.set_color('red')

# Rotate the labels by 45 degrees
ax1.tick_params(axis='x', rotation=45)

# Line plot for Cumulative Percentage
ax2 = ax1.twinx()
sns.lineplot(x='Construct', y='Cumulative Percentage', data=data, ax=ax2, color='red', marker='o')
ax2.set_ylabel('Cumulative Percentage (%)')
ax2.set_yticks(range(0, 101, 10))  # Set cumulative percentage grid at every 10%

plt.grid(True)
plt.show()
