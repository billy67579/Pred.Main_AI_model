import pandas as pd

# Read the first CSV file into a DataFrame
df1 = pd.read_csv('/home/besong/project/project_data/wind-check/t11.csv')

# Read the second CSV file into a DataFrame
df2 = pd.read_csv('/home/besong/project/project_data/wind-check/Sögel-7_1.csv')

# Read the third CSV file into a DataFrame
df3 = pd.read_csv('/home/besong/project/project_data/wind-check/Sögel-7_2.csv')

# Merge the DataFrames by columns (axis=1 means along the columns)
merged_df = pd.concat([df1, df2, df3], axis=1)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('/home/besong/project/project_data/wind-check/sögel_7.csv', index=False)

print("CSV files have been merged successfully!")
