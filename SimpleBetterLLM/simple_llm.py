import torch
import torch.nn as nn

class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, device="cpu"):
        super(SimpleLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.device = device

        # Move model to the selected device
        self.to(self.device)
    
    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])
        return output
    
    def generate_text(self, dataset, seed_text, length=100, temperature=1.0):
        self.eval()
        
        if len(seed_text) % dataset.token_size != 0:
            padding_needed = dataset.token_size - (len(seed_text) % dataset.token_size)
            seed_text = seed_text + ' ' * padding_needed
        
        seed_tokens = [seed_text[i:i+dataset.token_size] for i in range(0, len(seed_text), dataset.token_size)]
        
        indices = torch.tensor([[dataset.token_to_idx.get(token, 0) for token in seed_tokens]]).to(self.device)
        
        generated_text = seed_text
        
        num_tokens_to_generate = length // dataset.token_size
        for _ in range(num_tokens_to_generate):
            with torch.no_grad():
                output = self(indices)
                
                # The rest remains similar, just make sure any new tensors are on the device
                if temperature != 1.0:
                    output = output / temperature
                
                probabilities = torch.softmax(output, dim=1)
                next_token_idx = torch.multinomial(probabilities, 1)
                
                indices = torch.cat([indices[:, 1:], next_token_idx], dim=1)
                
                next_token = dataset.idx_to_token[next_token_idx.cpu().item()]
                generated_text += next_token
        
        return generated_text
