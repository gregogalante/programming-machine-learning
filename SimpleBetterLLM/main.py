import os
import re
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from simple_text_dataset import SimpleTextDataset
from simple_llm import SimpleLLM
from text_generator import TextGenerator
from torch.utils.data import DataLoader

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    device = mps_device
print(f"Using device: {device}")

# HELPERS
##

def save_model(model, dataset, path="simple_llm_model"):
    os.makedirs(path, exist_ok=True)
    
    model_path = os.path.join(path, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Modello salvato in: {model_path}")
    
    dataset_info = {
        "token_to_idx": dataset.token_to_idx,
        "idx_to_token": dataset.idx_to_token,
        "vocab_size": dataset.vocab_size,
        "token_size": dataset.token_size,
        "seq_length": dataset.seq_length
    }
    
    dataset_path = os.path.join(path, "dataset_info.pkl")
    with open(dataset_path, "wb") as f:
        pickle.dump(dataset_info, f)
    
    model_config = {
        "vocab_size": dataset.vocab_size,
        "embedding_dim": model.embedding.embedding_dim,
        "hidden_dim": model.lstm.hidden_size
    }
    
    config_path = os.path.join(path, "model_config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(model_config, f)
    
    return path

def load_model(path="simple_llm_model"):
    config_path = os.path.join(path, "model_config.pkl")
    with open(config_path, "rb") as f:
        model_config = pickle.load(f)
    
    dataset_path = os.path.join(path, "dataset_info.pkl")
    with open(dataset_path, "rb") as f:
        dataset_info = pickle.load(f)
    
    model = SimpleLLM(
        vocab_size=model_config["vocab_size"],
        embedding_dim=model_config["embedding_dim"],
        hidden_dim=model_config["hidden_dim"]
    )
    
    model_path = os.path.join(path, "model.pt")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model, dataset_info

def train_model(model, dataset, epochs=10, batch_size=32, lr=0.001):
    print(f"Loading dataloader with batch size {batch_size}...")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Creating optimizer with learning rate {lr}...")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting training...")
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        percent = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            percent += 100 / len(dataloader)
            print(f"\rEpoch {epoch+1}/{epochs} - {percent:.2f}%", end="")
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            sample_text = model.generate_text(dataset, "Nel mezzo", length=50)
            print(f"Sample text: {sample_text}")
    
    return model

def create_model():
    sample_text = ''

    print("Loading text...")
    with open("divina-commedia.txt", "r") as f:
        sample_text += f.read()

    print("Preprocessing text...")
    sample_text = re.sub(r'\s+', ' ', sample_text).strip()
    
    print("Creating dataset...")
    dataset = SimpleTextDataset(sample_text, seq_length=10, device=device)
    print("Creating model...")
    model = SimpleLLM(vocab_size=dataset.vocab_size, embedding_dim=48, hidden_dim=96, device=device)
    print("Training model...")
    trained_model = train_model(model, dataset, epochs=50, batch_size=16, lr=0.001)
    print("Saving model...")
    save_model(trained_model, dataset, "divina-commedia-model")
    
    return True

if __name__ == "__main__":
    # create_model()
    
    model, dataset_info = load_model("divina-commedia-model")
    text_gen = TextGenerator(model, dataset_info, device=device)
    text = text_gen.generate_text("Nel mezzo", length=150, temperature=0.8)
    print(text)
