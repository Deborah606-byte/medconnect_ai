import pandas as pd

# Load the CSV file
hospitals_df = pd.read_csv('hospitals.csv')

# Display the first few rows of the dataframe
print("Original DataFrame:")
print(hospitals_df.head())

# Drop duplicate rows
hospitals_df.drop_duplicates(inplace=True)

# Fill or drop missing values if necessary
# Example: Fill missing values with a placeholder or drop rows with missing values
# hospitals_df.fillna('Unknown', inplace=True)
hospitals_df.dropna(inplace=True)

# Convert text to consistent case if necessary
# Example: Convert all text columns to lowercase
text_columns = hospitals_df.select_dtypes(include='object').columns
for col in text_columns:
    hospitals_df[col] = hospitals_df[col].str.lower()

# Strip leading and trailing spaces from text columns
for col in text_columns:
    hospitals_df[col] = hospitals_df[col].str.strip()

# Display the cleaned dataframe
print("Cleaned DataFrame:")
print(hospitals_df.head())

# Save the cleaned dataframe back to a CSV file
hospitals_df.to_csv('cleaned_hospitals.csv', index=False)

print("Cleaned data saved to 'cleaned_hospitals.csv'")
