import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Read data from the CSV file
file_path = '../outputs/analyses/cs_analyses/frequency_analysis.csv'
df = pd.read_csv(file_path)

# Step 2: Create a grouped bar chart with dual y-axes for Total Frequency and Group Frequency

# Define figure and axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Set the width of the bars
bar_width = 0.4

# Generate the x locations for the groups
x = np.arange(len(df['Construct']))

# Plot Total Frequency on the left y-axis
bars1 = ax1.bar(x - bar_width/2, df['Total Frequency'], bar_width, color='b', label='Total Frequency')
ax1.set_ylabel('Total Frequency', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create a secondary axis for Group Frequency on the right y-axis
ax2 = ax1.twinx()
bars2 = ax2.bar(x + bar_width/2, df['Group Frequency'], bar_width, color='g', label='Group Frequency')
ax2.set_ylabel('Group Frequency', color='g')
ax2.tick_params(axis='y', labelcolor='g')

# Set the x-axis with construct names
ax1.set_xticks(x)
ax1.set_xticklabels(df['Construct'], rotation=90)

# Set title and layout
plt.title('Total and Group Frequencies by Construct with Dual Y-Axis')

# Show the plot
fig.tight_layout()
plt.show()

# Step 3: Create a line plot to visualize the relative frequencies

plt.figure(figsize=(12, 6))
sns.lineplot(x='Construct', y='Global Relative Frequency (Occurrence-wise)', data=df, marker='o', label='Occurrence-wise')
sns.lineplot(x='Construct', y='Global Relative Frequency (Group-wise)', data=df, marker='o', label='Group-wise')
plt.xticks(rotation=90)
plt.title('Line Plot of Global Relative Frequencies by Construct')
plt.ylabel('Relative Frequency')
plt.legend()
plt.tight_layout()
plt.show()

# Step 4: Create a dot plot to visualize the relative frequencies

plt.figure(figsize=(12, 6))
plt.plot(df['Construct'], df['Global Relative Frequency (Occurrence-wise)'], 'bo', label='Occurrence-wise')
plt.plot(df['Construct'], df['Global Relative Frequency (Group-wise)'], 'go', label='Group-wise')
plt.xticks(rotation=90)
plt.title('Dot Plot of Global Relative Frequencies by Construct')
plt.ylabel('Relative Frequency')
plt.legend()
plt.tight_layout()
plt.show()