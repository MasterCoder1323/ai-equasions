import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm  # Import tqdm for progress bars
import numpy as np
from model import SimpleTransformer  # Import your transformer model from model.py

class EquationDataset(Dataset):
    def __init__(self, data_file="data.jsonl", start_idx=0, end_idx=None):
        with open(data_file, "r") as f:
            # Load the entire dataset
            self.data = [json.loads(line) for line in f.readlines()]
        
        # If end_idx is None, use the entire dataset from start_idx onwards
        if end_idx is None:
            end_idx = len(self.data)
        
        # Use the specified range of data
        self.data = self.data[start_idx:end_idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.data[idx], dtype=torch.long)
        return sequence, sequence  # The target is the same as the input for now.

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    # Wrap the dataloader with tqdm for a progress bar
    for batch_idx, (input_seq, target_seq) in enumerate(tqdm(dataloader, desc="Training batches", unit="batch")):
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)

        optimizer.zero_grad()

        output = model(input_seq)
        
        # Flatten both the output and target
        output = output.reshape(-1, output.size(-1))  # Flatten to [batch_size * sequence_length, vocab_size]
        target_seq = target_seq.view(-1)  # Flatten to [batch_size * sequence_length]
        
        # Compute loss
        loss = criterion(output, target_seq)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, (input_seq, target_seq) in enumerate(tqdm(dataloader, desc="Evaluating batches", unit="batch")):
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)

            output = model(input_seq)
            
            # Flatten both the output and target
            output = output.reshape(-1, output.size(-1))  # Flatten to [batch_size * sequence_length, vocab_size]
            target_seq = target_seq.view(-1)  # Flatten to [batch_size * sequence_length]
            
            # Compute loss
            loss = criterion(output, target_seq)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def main():
    # Hyperparameters
    vocab_size = 17  # Adjust according to your vocabulary size
    embedding_dim = 32
    hidden_dim = 64
    num_heads = 2
    num_layers = 2
    batch_size = 128
    epochs = 10
    learning_rate = 0.001
    start_idx = 0  # Start training from this index
    end_idx = 10000  # Train on this number of sequences
    early_stop_patience = 3  # Number of epochs to wait before early stopping

    # Initialize model, dataset, and dataloader
    model = SimpleTransformer(vocab_size=vocab_size, embedding_dim=embedding_dim, 
                              hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the subset of the dataset
    dataset = EquationDataset(data_file="data.jsonl", start_idx=start_idx, end_idx=end_idx)

    # Split the dataset into training and validation sets (80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Training loop with tqdm for overall epoch progress
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Training step
        avg_train_loss = train(model, train_dataloader, criterion, optimizer, device)
        print(f"Training Loss: {avg_train_loss}")

        # Validation step
        avg_val_loss = evaluate(model, val_dataloader, criterion, device)
        print(f"Validation Loss: {avg_val_loss}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # Save model if it improves
            model_save_path = f"transformeV1.pt"
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(f"Early stopping after {epoch + 1} epochs due to no improvement in validation loss.")
                break

if __name__ == "__main__":
    main()
