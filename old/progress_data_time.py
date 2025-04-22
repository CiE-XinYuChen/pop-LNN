import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('train.csv')
data = data[['ISO2_code', 'Year', 'Population']]  
scaler = StandardScaler()

scaled_data = []

# Process each country separately
for country_code, group in data.groupby('ISO2_code'):
    group['Population'] = scaler.fit_transform(group[['Population']])
    scaled_data.append(group)

# Concatenate all the scaled data back into a single DataFrame
final_data = pd.concat(scaled_data)

print(final_data[['ISO2_code', 'Year', 'Population']])

# Optionally, save the scaled data to a new CSV file
final_data.to_csv('standardized_population_time.csv', index=False)
