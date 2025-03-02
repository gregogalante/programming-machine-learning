import torch

class TextGenerator:
    def __init__(self, model, dataset_info, device="cpu"):
        self.model = model
        self.token_to_idx = dataset_info["token_to_idx"]
        self.idx_to_token = dataset_info["idx_to_token"]
        self.token_size = dataset_info["token_size"]
        self.seq_length = dataset_info["seq_length"]
        self.device = device
    
    def generate_text(self, seed_text, length=100, temperature=1.0):
        self.model.eval()
        
        if len(seed_text) % self.token_size != 0:
            padding_needed = self.token_size - (len(seed_text) % self.token_size)
            seed_text = seed_text + ' ' * padding_needed
        
        seed_tokens = [seed_text[i:i+self.token_size] for i in range(0, len(seed_text), self.token_size)]

        if len(seed_tokens) < self.seq_length:
            padding_tokens = ['   '] * (self.seq_length - len(seed_tokens))
            seed_tokens = padding_tokens + seed_tokens

        if len(seed_tokens) > self.seq_length:
            seed_tokens = seed_tokens[-self.seq_length:]
        
        default_idx = 0
        indices = torch.tensor([[self.token_to_idx.get(token, default_idx) for token in seed_tokens]]).to(self.device)
        
        generated_text = seed_text
        
        num_tokens_to_generate = length // self.token_size
        for _ in range(num_tokens_to_generate):
            with torch.no_grad():
                output = self.model(indices)
                
                if temperature != 1.0:
                    output = output / temperature
                
                probabilities = torch.softmax(output, dim=1)
                next_token_idx = torch.multinomial(probabilities, 1).to(self.device)
                
                indices = torch.cat([indices[:, 1:], next_token_idx], dim=1)
                
                # Move back to CPU to get the item
                next_token = self.idx_to_token[next_token_idx.cpu().item()]
                generated_text += next_token
        
        return generated_text