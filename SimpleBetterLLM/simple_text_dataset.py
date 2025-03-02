import torch

class SimpleTextDataset(torch.utils.data.Dataset):
    def __init__(self, text, seq_length=10, device="cpu"):
        self.token_size = 1
        self.text = text.lower()
        self.seq_length = seq_length
        self.device = device

        padding_needed = (self.token_size - len(self.text) % self.token_size) % self.token_size
        self.padded_text = self.text + ' ' * padding_needed
        self.tokens = [self.padded_text[i:i+self.token_size] for i in range(0, len(self.padded_text), self.token_size)]

        unique_tokens = sorted(list(set(self.tokens)))
        self.token_to_idx = {token: i for i, token in enumerate(unique_tokens)}
        self.idx_to_token = {i: token for i, token in enumerate(unique_tokens)}
        self.vocab_size = len(unique_tokens)
        
        # Vectorize the text and store vectors in self.data
        self.data = []
        for i in range(0, len(self.tokens) - seq_length):
            input_seq = self.tokens[i:i+seq_length]
            target_token = self.tokens[i+seq_length]

            input_indices = [self.token_to_idx[token] for token in input_seq]
            target_index = self.token_to_idx[target_token]
            
            self.data.append((input_indices, target_index))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_indices, target_index = self.data[idx]
        return torch.tensor(input_indices).to(self.device), torch.tensor(target_index).to(self.device)
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def decode(self, indices):
        return ''.join([self.idx_to_token[idx.item()] for idx in indices])
    
    def encode(self, text):
        text = text.lower()
        tokens = [text[i:i+self.token_size] for i in range(0, len(text), self.token_size)]
        
        if len(tokens[-1]) < self.token_size:
            tokens[-1] = tokens[-1] + ' ' * (self.token_size - len(tokens[-1]))
        
        default_idx = 0
        indices = [self.token_to_idx.get(token, default_idx) for token in tokens]

        return indices