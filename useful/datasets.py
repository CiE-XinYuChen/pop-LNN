import pandas as pd
import torch
import os

# Load the dataset
file_path = 'train.csv'  # Path to the CSV file
data = pd.read_csv(file_path)

# Create a directory to save the tensors if it doesn't exist
output_dir = 'datasets'
os.makedirs(output_dir, exist_ok=True)

# Split the dataset by ISO2_code and create sequences of 10 years for prediction
for country_code, group in data.groupby('ISO2_code'):
    # Sort the data by year to ensure chronological order
    group = group.sort_values(by='Year')
    
    # Prepare lists to hold the features (X) and targets (y)
    X_data = []
    y_data = []
    
    # Create sequences of 10 years of data and use the 11th year as the target
    for i in range(len(group) - 10):
        X_data.append(group.iloc[i:i+10][['Year', 'Population']].values)  # 10 years data
        y_data.append(group.iloc[i+10]['Population'])  # Population of the next year (target)
    
    # Convert to tensors
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)  # Make y a column vector
    
    # Save the tensors to files
    torch.save((X_tensor, y_tensor), os.path.join(output_dir, f'{country_code}.pt'))

print("Tensors have been saved successfully.")
