import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, test_loader, device, epochs=5, learning_rate=0.001):
    """Train the model and evaluate on test data."""
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Move model to the correct device (CPU/MPS/GPU)
    model.to(device)

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Flatten images from (batch_size, 1, 28, 28) -> (batch_size, 784)
            images = images.view(images.size(0), -1)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print loss after each epoch
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Evaluate the model after training
    evaluate_model(model, test_loader, device)

def evaluate_model(model, test_loader, device):
    """Evaluate the trained model on the test dataset."""
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1)  # Flatten images
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


