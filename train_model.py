import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import shutil

# Define hyperparameters as variables before the run function
batch_size = 64
num_epochs = 10
learning_rate = 0.001
weight_decay = 1e-5
patience = 3  # Early stopping patience
max_grad_norm = 3.0  # Gradient clipping max norm
train_mode = False # True to split data between training and validation | False to do a final training on all data

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 1024)  # After convolution and pooling the size will be (3, 3)
        self.fc2 = nn.Linear(1024, 10)  # 10 classes for MNIST

        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        # Apply convolution layers with ReLU activation and max pooling
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)  # Max pooling with 2x2 kernel (halves the size)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)  # Max pooling with 2x2 kernel
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)  # Max pooling with 2x2 kernel

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)  # Automatically calculate the correct flattened size
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load and preprocess the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Split the training data into training and validation sets (80% training, 20% validation) if train_mode=True
if train_mode:
    train_size = int(0.8 * len(train_data))
else:
    train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

# Create data loaders for training, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # L2 regularization

# Learning rate scheduler with plateau reduction
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

# Function to train the model
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=10, train_mode=True):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    top_errors = []
    writer = SummaryWriter()  # Initialize TensorBoard writer

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        # Iterate over batches
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', ncols=100):
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)  # Gradient clipping
            optimizer.step()

            # Accuracy calculation
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        # Calculate and log the average loss and accuracy for this epoch
        avg_loss = running_loss / len(train_loader)
        avg_accuracy = correct_preds / total_preds
        train_losses.append(avg_loss)
        train_accuracies.append(avg_accuracy)

        # Validation phase
        if train_mode:
            val_loss, val_accuracy, val_top_errors = evaluate_model(model, val_loader, criterion)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            top_errors.extend(val_top_errors)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0  # reset patience counter
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        # Learning rate reduction based on training loss
        scheduler.step(avg_loss)

        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', avg_accuracy, epoch)

        if train_mode:
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        # Print training statistics        
        if train_mode:
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}, Training Accuracy: {avg_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}, Training Accuracy: {avg_accuracy:.4f}")


    writer.close()

    return train_losses, val_losses, train_accuracies, val_accuracies, top_errors

# Function to evaluate the model on the validation or test set
def evaluate_model(model, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    top_errors = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Accuracy calculation
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

            # Collect top errors (incorrect predictions with highest confidence)
            incorrect = (predicted != labels)
            top_errors.extend([(inputs[i], predicted[i], labels[i]) for i in range(len(incorrect)) if incorrect[i]])

    avg_loss = running_loss / len(val_loader)
    avg_accuracy = correct_preds / total_preds
    return avg_loss, avg_accuracy, top_errors[:6]  # Return top errors

# Function to plot the metrics
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, top_errors):
    # Plot Losses
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Plot Accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.savefig('plots/loss_accuracy_plot.png')
    print("Saved loss and accuracy in plots.")
    plt.show()

    # Plot Confusion Matrix
    all_preds = []
    all_labels = []
    for inputs, labels in tqdm(test_loader, desc="Generating Confusion Matrix"):
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('plots/confusion_matrix.png')
    print("Saved confusion matrix in plots.")
    plt.show()

    # Show Top Errors
    for i, (input_image, pred, true) in enumerate(top_errors):
        plt.imshow(input_image[0], cmap='gray')
        plt.title(f'Predicted: {pred.item()}, True: {true.item()}')
        plt.show()

# Run function to train the model
def run(train_mode=True, num_epochs=num_epochs):
    if train_mode:
        print("Training on 80% of the data with 20% for validation")
        train_losses, val_losses, train_accuracies, val_accuracies, top_errors = train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=num_epochs, train_mode=True)
    else:
        print("Training on 100% of the data")
        train_losses, val_losses, train_accuracies, val_accuracies, top_errors = train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=num_epochs, train_mode=False)

    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, top_errors)

    # Save the model for future use
    torch.save(model.state_dict(), 'cnn_mnist_model.pth')
    print("Model saved!")
    # Delete the TensorBoard logs if they are not needed
    if os.path.exists('runs'):
        shutil.rmtree('runs')

# Run the main function with the desired mode
if __name__ == "__main__":
    run(train_mode=train_mode)
