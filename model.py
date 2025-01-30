import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64, num_heads=2, num_layers=2):
        super(SimpleTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Transformer block configuration
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            dropout=0.1
        )
        
        self.fc_out = nn.Linear(embedding_dim, vocab_size)  # Output layer

    def forward(self, src):
        # src is a tensor of shape (batch_size, seq_len)
        
        embedded = self.embedding(src)  # Shape: (batch_size, seq_len, embedding_dim)
        
        # Transformer expects shape (seq_len, batch_size, embedding_dim)
        embedded = embedded.permute(1, 0, 2)
        
        # Passing through transformer
        transformer_out = self.transformer(embedded, embedded)
        
        # Pass through the final linear layer for each token in the sequence
        output = self.fc_out(transformer_out)  # Shape: (seq_len, batch_size, vocab_size)
        
        # Return the output back to [batch_size, seq_len, vocab_size]
        output = output.permute(1, 0, 2)  # Now shape: (batch_size, seq_len, vocab_size)
        
        return output
