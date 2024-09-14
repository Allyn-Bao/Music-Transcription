import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import os

data_dir = "train_data"
train_X = pd.read_csv(os.path.join(data_dir, "train_x.csv"))
train_Y = pd.read_csv(os.path.join(data_dir, "train_y.csv"))
val_X = pd.read_csv(os.path.join(data_dir, "val_x.csv"))
val_Y = pd.read_csv(os.path.join(data_dir, "val_y.csv"))
test_X = pd.read_csv(os.path.join(data_dir, "test_x.csv"))
test_Y = pd.read_csv(os.path.join(data_dir, "test_y.csv"))

# drop timestamp column
train_X = train_X.drop(train_X.columns[0], axis=1)
val_X = val_X.drop(val_X.columns[0], axis=1)
test_X = test_X.drop(test_X.columns[0], axis=1)

# Convert DataFrames to PyTorch Tensors
train_X_tensor = torch.tensor(train_X.values, dtype=torch.float32)
train_Y_tensor = torch.tensor(train_Y['Y_Encoded'].values, dtype=torch.long)
val_X_tensor = torch.tensor(val_X.values, dtype=torch.float32)
val_Y_tensor = torch.tensor(val_Y['Y_Encoded'].values, dtype=torch.long)
test_X_tensor = torch.tensor(test_X.values, dtype=torch.float32)
test_Y_tensor = torch.tensor(test_Y['Y_Encoded'].values, dtype=torch.long)

# Create TensorDataset and DataLoader for batching
train_dataset = TensorDataset(train_X_tensor, train_Y_tensor)
val_dataset = TensorDataset(val_X_tensor, val_Y_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

"""
A simple Neural Network Model
"""


# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Model, loss function, and optimizer
input_size = train_X.shape[1]  # Number of input features
num_classes = len(train_Y['Y_Encoded'].unique())  # Number of output classes
model = SimpleNN(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

"""
Training Loop
"""
# Training loop
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track loss
        running_loss += loss.item()

    # Print average loss for this epoch
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')

print('Finished Training')


"""
evaluate model with testing sets
"""
# Set the model to evaluation mode
model.eval()

# Initialize counters for accuracy calculation
correct = 0
total = 0

# Disable gradient calculation
with torch.no_grad():
    # Get predictions for the test dataset
    test_outputs = model(test_X_tensor)

    # Get the predicted class (the one with the maximum score)
    _, predicted = torch.max(test_outputs.data, 1)

    # Calculate the total and correct predictions
    total = test_Y_tensor.size(0)
    correct = (predicted == test_Y_tensor).sum().item()

    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test data: {accuracy:.2f}%')


"""
Save Model
"""
# Define the directory to save the model
model_save_path = os.path.join('models', 'chord_detection_model.pth')

# Create the directory if it does not exist
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Save the model parameters
torch.save(model.state_dict(), model_save_path)
print(f'Model parameters saved to {model_save_path}')