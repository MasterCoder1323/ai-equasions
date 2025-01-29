import torch
import torch.nn as nn

class EquationSolverLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64, output_dim=1, num_layers=2):
        """
        Initialize the LSTM model.
        
        Parameters:
        - vocab_size: The number of unique tokens (vocabulary size).
        - embedding_dim: The dimensionality of the token embeddings.
        - hidden_dim: The number of features in the hidden state of the LSTM.
        - output_dim: The dimensionality of the output (in this case, a single value).
        - num_layers: The number of LSTM layers.
        """
        super(EquationSolverLSTM, self).__init__()

        # Embedding layer to convert token IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=0.3)

        # Fully connected (Linear) layer for output
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        - x: A tensor of token IDs with shape (batch_size, sequence_length)
        
        Returns:
        - output: The predicted result (equation solution).
        """
        # Get embeddings for the input tokens
        embedded = self.embedding(x)
        
        # Pass through LSTM layer
        lstm_out, (hn, cn) = self.lstm(embedded)
        
        # We use the last hidden state to make the prediction
        final_hidden_state = hn[-1]  # Get the last layer's hidden state
        
        # Pass the last hidden state through the fully connected layer
        output = self.fc(final_hidden_state)
        
        return output

# Example of model creation
if __name__ == "__main__":
    vocab_size = 16  # Change this according to your token vocabulary size
    model = EquationSolverLSTM(vocab_size=vocab_size)

    # Print the model architecture
    print(model)
