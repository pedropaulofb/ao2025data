import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data from the CSV file
file_path = '../outputs/analyses/cs_analyses/coverage_percentage.csv'
data = pd.read_csv(file_path)

# 1. Plot Line Chart for Coverage vs. Percentage
plt.figure(figsize=(10, 6))  # Set the figure size
sns.lineplot(data=data, x='Percentage', y='Coverage', marker='o')

# Improved title and labels
plt.title('Coverage Achieved by Top Percentages of Constructs', fontsize=14)
plt.xlabel('Top Percentage of Constructs Considered (%)', fontsize=12)
plt.ylabel('Coverage of Total Occurrences', fontsize=12)

# Add annotations for each point to display the corresponding "Top k Constructs" value
for i in range(len(data)):
    plt.text(
        data['Percentage'][i] + 1,  # Slightly adjust the x position (move to the right)
        data['Coverage'][i] + 0.015,  # Slightly adjust the y position (move upwards)
        f"k={data['Top k Constructs'][i]}",
        fontsize=10,
        ha='right'  # Horizontal alignment
    )

# Additional formatting
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True)  # Add a grid for better readability

# Display the plot
plt.show()