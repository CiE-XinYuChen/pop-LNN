import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model import LTCNetwork  # Import the model from model.py
import numpy as np

# Function to train and save models for each country
def train_and_save_models(data_dir, model_dir, num_epochs=800, batch_size=4, learning_rate=1e-4):
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.pt'):
            country_code = file_name.split('.')[0]
            print(f"Training model for {country_code}...")

            # Load data (X_tensor, y_tensor)
            X_data, y_data = torch.load(os.path.join(data_dir, file_name))
            print(X_data, '\n', y_data)
            # No need to reshape again, X_data already in shape (samples, 10, 2)
            X_tensor = torch.tensor(X_data, dtype=torch.float32)
            y_tensor = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)
            # Split into training and testing data
            X_train, X_test, y_train, y_test = train_test_split(
                X_tensor, y_tensor, test_size=0.3, random_state=42
            )
            # print(X_train[2], '\n', y_train[2])

            train_dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Initialize the model, loss, and optimizer with a fixed initial learning rate
            model = LTCNetwork(input_size=2, hidden_size=10, output_size=1)
            criterion = torch.nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Fixed initial learning rate

            # Introduce a dynamic learning rate scheduler (e.g., StepLR)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.5)  # Every 150 epochs, decay lr by a factor of 0.6

            # Training loop
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                    optimizer.zero_grad()

                    hidden = torch.zeros(x_batch.size(0), model.ltc_cell.hidden_size).to(x_batch.device)

                    outputs, _ = model(x_batch)
                    loss = criterion(outputs.squeeze(), y_batch.squeeze())  # Compute loss for this batch
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    optimizer.step()

                    running_loss += loss.item()

                scheduler.step()

                if (epoch + 1) % 5 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

            model_save_path = os.path.join(model_dir, f'{country_code}_model.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Model for {country_code} saved at {model_save_path}")

            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                all_outputs = []
                all_labels = []
                for x_batch, y_batch in test_loader:
                    outputs, _ = model(x_batch)
                    loss = criterion(outputs.squeeze(), y_batch.squeeze())
                    test_loss += loss.item()

                    all_outputs.append(outputs.squeeze().cpu().numpy())
                    all_labels.append(y_batch.squeeze().cpu().numpy())

                # Calculate average test loss
                avg_test_loss = test_loss / len(test_loader)
                print(f"Test Loss for {country_code}: {avg_test_loss:.4f}")

                # Optionally, print out some predictions vs true values for debugging
                all_outputs = np.concatenate(all_outputs)
                all_labels = np.concatenate(all_labels)
                print(f"Predictions vs True values for {country_code}:")
                print("Predictions:", all_outputs[:10])  # Print first 10 predictions
                print("True values:", all_labels[:10])  # Print first 10 true values

data_dir = 'datasets'
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

# Train models
train_and_save_models(data_dir, model_dir, num_epochs=2500, batch_size=256, learning_rate=1e-2)
