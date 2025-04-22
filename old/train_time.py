import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the custom dataset (Replace 'train.csv' with your actual CSV file path)
data = pd.read_csv('standardized_population_time.csv')

# Data preprocessing
# Select relevant columns and ensure no missing values
data = data[['Year', 'Population']]  # Only keep 'Year' and 'Population' columns

# Check for missing values and handle them (for simplicity, let's drop them)
# data.dropna(inplace=True)

# Standardize the 'Year' feature and 'Population' target variable
# scaler_x = StandardScaler()  # For the 'Year' feature
# scaler_y = StandardScaler()  # For the 'Population' feature

# # Standardize the 'Year' column
# X_scaled = scaler_x.fit_transform(data[['Year']].values)

# # Standardize the 'Population' column (the target)
# y_scaled = scaler_y.fit_transform(data['Population'].values.reshape(-1, 1))
X_v = data[['Year']].values
y_v = data[['Population']].values
# Convert to torch tensors
X_tensor = torch.FloatTensor(X_v)
y_tensor = torch.FloatTensor(y_v)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.3, random_state=42)

# Define the LTCCell and LTCNetwork model
class LTCCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LTCCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.bias_ih = nn.Parameter(torch.Tensor(hidden_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))
        self.tau = nn.Parameter(torch.ones(hidden_size) * 1.0)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_ih)
        nn.init.kaiming_uniform_(self.weight_hh)
        nn.init.constant_(self.bias_ih, 0)
        nn.init.constant_(self.bias_hh, 0)

    def forward(self, input, hidden):
        pre_activation = torch.addmm(self.bias_ih, input, self.weight_ih.t()) + torch.addmm(self.bias_hh, hidden, self.weight_hh.t())
        dh = (pre_activation - hidden) / self.tau
        new_hidden = hidden + dh
        return new_hidden


class LTCNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LTCNetwork, self).__init__()
        self.ltc_cell = LTCCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        hidden = torch.zeros(input.size(0), self.ltc_cell.hidden_size)
        for i in range(input.size(1)):
            hidden = self.ltc_cell(input[:, i, :], hidden)
        output = self.fc(hidden)
        return output, hidden
# Initialize the model, loss function, and optimizer
model = LTCNetwork(input_size=1, hidden_size=10, output_size=1)  # For regression, output_size = 1
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Learning rate

# Train the model
num_epochs = 800
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs, _ = model(X_train.unsqueeze(2))  # Unsqueeze to match the input dimensions
    loss = criterion(outputs.squeeze(), y_train)  # Squeeze to match target dimensions
    loss.backward()
    
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients if necessary
    
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), 'ltc_population_model_time.pth')

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    outputs, _ = model(X_test.unsqueeze(2))
    test_loss = criterion(outputs.squeeze(), y_test)
    print(f"Test Loss: {test_loss.item():.4f}")
