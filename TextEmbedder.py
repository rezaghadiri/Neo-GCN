import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

from huggingface_hub import login
hf_token = "?"
login(token=hf_token)

# 1. Data Preprocessing
def preprocess_cora_data(text):
    # Clean and prepare the text data
    text = text.replace("Abstract:", "")
    text = text.replace("Title:", "")
    text = ' '.join(text.split())
    return text

# 2. Dataset Class
class CoraDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts  # Now expects numpy array
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 3. Model Configuration
def setup_model_and_tokenizer():
    # Initialize tokenizer and model
    model_name = "microsoft/deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=7  # Number of classes in Cora dataset
    )

    # Print available module names
    # print("Available modules:")
    # for name, _ in model.named_modules():
        # print(name)

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,  # rank of the LoRA matrices
        lora_alpha=32,  # scaling factor
        lora_dropout=0.1,
        target_modules=["query_proj", "key_proj", "value_proj"]  # Apply LoRA to attention layers
    )
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    
    return model, tokenizer

# 4. Training Function
def train_model(model, train_loader, val_loader, device, num_epochs=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"\nEpoch {epoch + 1} - Validation loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained("best_model")

# 5. Generate Embeddings Function
def generate_embeddings(model, loader, device):
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get the last hidden states
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # Use the [CLS] token embedding from the last layer
            embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings)

# 6. Main Execution
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and preprocess data
    df = pd.read_csv('Cora.csv', header=None)
    texts = df[0].apply(preprocess_cora_data).reset_index(drop=True)  # Reset index
    
    # Convert labels to numerical format
    label_map = {label: idx for idx, label in enumerate(df[1].unique())}
    labels = df[1].map(label_map).values

    print("The number of labels:", len(labels))
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts.values,  # Convert to numpy array
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels  # Ensure balanced split
    )
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    model.to(device)
    
    # Create datasets
    train_dataset = CoraDataset(train_texts, train_labels, tokenizer)
    val_dataset = CoraDataset(val_texts, val_labels, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8, # 4
        shuffle=True,
        num_workers=0,  # Set to 0 for Google Colab
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8, # 4
        shuffle=False,
        num_workers=0,  # Set to 0 for Google Colab
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Train model
    train_model(model, train_loader, val_loader, device)
    
    # Generate embeddings for all data
    all_dataset = CoraDataset(texts.values, labels, tokenizer)
    all_loader = DataLoader(all_dataset, batch_size=8, shuffle=False) # 4
    embeddings = generate_embeddings(model, all_loader, device)

    print("Embedding Shape:", embeddings.shape)
    
    # Save embeddings
    np.save('cora_embeddings.npy', embeddings)

if __name__ == "__main__":
    main()