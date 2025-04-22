import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

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
# Load the trained model
model = LTCNetwork(input_size=1, hidden_size=10, output_size=1)
model.load_state_dict(torch.load('ltc_population_model_time.pth'))
model.eval()

# Input data directly through lists
years_input = [1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989]  # Example years
population_input = [0,0.03,0.06,0.09,0.12,0.15,0.17,0.19,0.20,0.205]  # Example in millions

# Prepare tensor with raw population data (no normalization)
X_tensor = torch.FloatTensor(population_input).unsqueeze(0).unsqueeze(-1)

# Predict future population
predictions = []
nums = 4  # You can set this to any number of future steps you want to predict
for _ in range(nums):
    with torch.no_grad():
        output, _ = model(X_tensor)
    predicted_value = output[:, 0].item()
    predictions.append(predicted_value)

    next_input_data = torch.tensor([[[predicted_value]]])
    X_tensor = torch.cat((X_tensor[:, 1:, :], next_input_data), dim=1)
# Prepare future years
years_predicted = [years_input[-1] + i for i in range(1, nums + 1)]

# Plot results
plt.figure(figsize=(10, 6))
print(population_input)
# Actual population plot
plt.plot(years_input, population_input, label='Actual Population', color='blue', marker='o')
# Predicted future populations
plt.scatter(years_predicted, predictions, label='Predicted Population', color='red', marker='X', s=100)

# Add labels and grid
plt.xlabel('Year')
plt.ylabel('Population (millions)')
plt.title(f'Population Prediction ({nums} Steps Ahead)')
plt.legend()
plt.grid(True)

# Save and display
plt.savefig(f'population_{nums}_steps_prediction.png')
plt.show()