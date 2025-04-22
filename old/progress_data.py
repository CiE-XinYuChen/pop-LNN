import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the custom dataset (replace with your actual CSV file path)
data = pd.read_csv('train.csv')

# Select only the relevant columns
data = data[['ISO2_code', 'Year', 'Population']]  # Only keep 'ISO2_code', 'Year', and 'Population'

# Initialize a StandardScaler
scaler = StandardScaler()

# Create an empty list to store the processed data
scaled_data = []

# Process each country separately
for country_code, group in data.groupby('ISO2_code'):
    # For each country, standardize the 'Population' column
    group['Population'] = scaler.fit_transform(group[['Population']])
    
    # Append the processed data for each country
    scaled_data.append(group)

# Concatenate all the scaled data back into a single DataFrame
final_data = pd.concat(scaled_data)
# Sort by 'Year' in ascending order before saving
# final_data = final_data.sort_values(by='Year', ascending=True)

# Print the 'ISO2_code' and 'Population' columns after standardization for inspection
print(final_data[['ISO2_code', 'Year', 'Population']])

# Optionally, save the scaled data to a new CSV file
final_data.to_csv('standardized_population.csv', index=False)
