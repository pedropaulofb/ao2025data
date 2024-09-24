import pandas as pd
from icecream import ic

evaluations = ["cs","rs"]
cleans = ["t","f"]

base_file1 = 'outputs/visualizations/'
base_file2 = "outputs/visualizations/movement/"
end_file1 = '/quadrant_analysis_Global Relative Frequency (Occurrence-wise)_vs_Global Relative Frequency (Group-wise).csv'
end_file2 = '/quadrant_analysis_Total Frequency_vs_Group Frequency.csv'

for evaluation in evaluations:
    for clean in cleans:
        # Load the two CSV files
        file1 = base_file1 + evaluation + "_ontouml_no_classroom_" + clean + end_file1
        file2 = base_file2 + evaluation + "_ontouml_no_classroom_" + clean + end_file2

        # Read the CSV files into pandas dataframes
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # Merge the two dataframes on the 'construct' column
        merged_df = pd.merge(df1[['construct', 'quadrant']], df2[['construct', 'quadrant_start', 'quadrant_end']], on='construct')

        # Save the merged dataframe to a new CSV file
        output_file = base_file2 + evaluation + "_ontouml_no_classroom_" + clean + '/merged_quadrant_analysis.csv'
        merged_df.to_csv(output_file, index=False)

        print(f"CSV file has been created: {output_file}")
