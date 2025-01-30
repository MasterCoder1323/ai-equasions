import torch
import json
from torch.utils.data import DataLoader
from model import SimpleTransformer
from tokenizer import EquasionTokenizer  # Import the tokenizer
import numpy as np

class EquationDataset(torch.utils.data.Dataset):
    def __init__(self, data_file="data.jsonl", start_idx=0, end_idx=None):
        with open(data_file, "r") as f:
            # Load the entire dataset
            self.data = [json.loads(line) for line in f.readlines()]
        
        if end_idx is None:
            end_idx = len(self.data)
        
        self.data = self.data[start_idx:end_idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.data[idx], dtype=torch.long)
        return sequence  # No target, we are testing.

def test(model, dataloader, device, tokenizer):
    model.eval()
    predictions = []
    with torch.no_grad():
        for input_seq in dataloader:
            input_seq = input_seq.to(device)
            
            # Get the model's output
            output = model(input_seq)
            
            # Convert output to predicted solutions (e.g., the last value)
            output = output.argmax(dim=-1)  # Get the predicted index
            
            # Decode the predicted tokens back to the equation format
            for seq in output:
                decoded_seq = tokenizer.decode(seq.cpu().numpy())
                predictions.append(decoded_seq)

    return predictions

def main():
    # Hyperparameters
    vocab_size = 17  # Adjust according to your vocabulary size
    embedding_dim = 32
    hidden_dim = 64
    num_heads = 2
    num_layers = 2
    batch_size = 64
    start_idx = 10000  # Test dataset starts after training data
    end_idx = 11000  # Use the next 1000 lines for testing

    # Initialize model and dataset
    model = SimpleTransformer(vocab_size=vocab_size, embedding_dim=embedding_dim,
                              hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load trained model
    model.load_state_dict(torch.load("transformeV1.pt"))  # Load the best model after training

    # Initialize tokenizer
    tokenizer = EquasionTokenizer()
    tokenizer.load_vocab()

    # Load the test dataset
    dataset = EquationDataset(data_file="data.jsonl", start_idx=start_idx, end_idx=end_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Test the model
    predictions = test(model, dataloader, device, tokenizer)

    # Print the first 10 predictions for inspection
    print("Predictions: ")
    for idx, pred in enumerate(predictions[:10]):
        print(f"Test case {idx + 1}: Predicted solution {pred}")

if __name__ == "__main__":
    main()
