import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from the CSV file
file_path = '../outputs/analyses/cs_analyses/central_tendency_dispersion.csv'
df = pd.read_csv(file_path)

# 1. Box Plot for Mean, Standard Deviation, and Variance
# Convert the columns to numeric, in case they are read as strings
numeric_columns = ['Mean', 'Median', 'Mode', 'Standard Deviation', 'Variance', 'Skewness', 'Kurtosis',
                   '25th Percentile (Q1)', '75th Percentile (Q3)', 'Interquartile Range (IQR)']

df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Box Plot with Logarithmic Scale
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['Mean', 'Standard Deviation', 'Variance']])
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.title('Box Plot for Mean, Standard Deviation, and Variance (Log Scale)')
plt.ylabel('Log Scale Values')
plt.show()

# 2. Heatmap of Correlation Between Measures
# Convert the columns to numeric, in case they are read as strings

# Rename the columns for shorter labels
df.rename(columns={
    'Standard Deviation' : 'SD',
    '25th Percentile (Q1)': 'Q1',
    '75th Percentile (Q3)': 'Q3',
    'Interquartile Range (IQR)': 'IQR'
}, inplace=True)

# List of numeric columns with updated names
numeric_columns = ['Mean', 'Median', 'SD', 'Variance', 'Skewness', 'Kurtosis', 'Q1', 'Q3', 'IQR']

df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Heatmap of Correlation Between Measures with Tilted Labels and Shortened Names
plt.figure(figsize=(10, 8))
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

plt.title('Correlation Heatmap of Measures')
plt.xticks(rotation=45)  # Tilt the x-axis labels by 45 degrees
plt.yticks(rotation=45)  # Tilt the y-axis labels by 45 degrees
plt.show()

# # 3. Scatter Plot Matrix (Pair Plot)
# numeric_columns = ['Mean', 'Median', 'Mode', 'Standard Deviation', 'Variance', 'Skewness', 'Kurtosis',
#                    '25th Percentile (Q1)', '75th Percentile (Q3)', 'Interquartile Range (IQR)']
# sns.pairplot(df[numeric_columns])
# plt.suptitle('Scatter Plot Matrix of Measures', y=1.02)
# plt.show()
#
# # 4. Bar Chart for Mean Values Across Constructs
# plt.figure(figsize=(12, 6))
# sns.barplot(x='Construct', y='Mean', data=df, palette='viridis')
# plt.title('Bar Chart of Mean Values Across Constructs')
# plt.xticks(rotation=90)
# plt.ylabel('Mean')
# plt.show()
#
# # 5. Violin Plot for Distribution of Skewness and Kurtosis
# plt.figure(figsize=(10, 6))
# sns.violinplot(x='Construct', y='Skewness', data=df)
# plt.title('Violin Plot for Skewness Across Constructs')
# plt.xticks(rotation=90)
# plt.show()
#
# plt.figure(figsize=(10, 6))
# sns.violinplot(x='Construct', y='Kurtosis', data=df)
# plt.title('Violin Plot for Kurtosis Across Constructs')
# plt.xticks(rotation=90)
# plt.show()
