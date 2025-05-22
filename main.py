import torch
from data.dataset import get_mnist_dataloader
from models.model import SimpleNN
from train.train import train_model

# Hyperparameters
INPUT_SIZE = 28 * 28  # MNIST images are 28x28 pixels
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10  # Digits 0-9
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

def main():
    # Check for GPU availability
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    train_loader, test_loader = get_mnist_dataloader(batch_size=BATCH_SIZE)

    # Initialize Model
    model = SimpleNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)

    # Train the model
    train_model(model, train_loader, test_loader, device, epochs=EPOCHS, learning_rate=LEARNING_RATE)
    # Save the trained model after training
    torch.save(model.state_dict(), "mnist_model.pth")
    print("Model saved as mnist_model.pth")


if __name__ == "__main__":
    main()


