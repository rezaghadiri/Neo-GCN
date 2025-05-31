# !huggingface-cli login
# ?

# !pip install torch torchvision torchaudio
# !pip install transformers peft bitsandbytes accelerate pandas scikit-learn datasets sentencepiece

import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AdamW,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import numpy as np

# --- Configuration ---
FILE_PATH = "Cora.csv"  # Path to your Cora.csv file
MODEL_NAME = "meta-llama/Llama-3.1-8B"
OUTPUT_DIR = "output_llama3_cora"
EMBEDDINGS_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cora_embeddings.pt")
ADAPTERS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "lora_adapters")

# Training Hyperparameters
NUM_EPOCHS = 5 # Adjust as needed
BATCH_SIZE = 4  # Adjust based on GPU memory. Effective batch size will be BATCH_SIZE for contrastive loss.
LEARNING_RATE = 2e-5 # Typical for LoRA
WARMUP_STEPS = 100
MAX_SEQ_LENGTH = 512 # Max sequence length for tokenizer
TEMPERATURE = 0.05 # Temperature for InfoNCE loss
POOLING_STRATEGY = 'mean' # 'mean' or 'last'

# QLoRA Configuration
LORA_R = 16  # Rank of LoRA matrices
LORA_ALPHA = 32 # Alpha scaling factor for LoRA
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [ # Common target modules for Llama models
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

class TextDataset(Dataset):
    """PyTorch Dataset for text data."""
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0)
        }

def load_cora_data(file_path):
    """Loads text data from the Cora CSV file."""
    try:
        df = pd.read_csv(file_path, header=None) # Assuming no header
        # The first column contains the titles and abstracts
        # The last column contains the labels (not used for unsupervised fine-tuning here)
        texts = df.iloc[:, 0].tolist()
        print(f"Loaded {len(texts)} texts from {file_path}")
        return texts
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        print("Please ensure Cora.csv is in the correct location.")
        exit()
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit()


def pool_embeddings(last_hidden_state, attention_mask, strategy='mean'):
    """Pools token embeddings to get a single sentence embedding."""
    if strategy == 'mean':
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    elif strategy == 'last':
        batch_size = last_hidden_state.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
    else:
        raise ValueError(f"Unsupported pooling strategy: {strategy}")

def info_nce_loss(embeddings1, embeddings2, temperature):
    """
    Calculates InfoNCE loss for contrastive learning.
    embeddings1 and embeddings2 are two views of the same original batch.
    Shape: (batch_size, hidden_dim)
    """
    batch_size = embeddings1.shape[0]
    device = embeddings1.device

    # Normalize embeddings
    embeddings1_norm = F.normalize(embeddings1, p=2, dim=1)
    embeddings2_norm = F.normalize(embeddings2, p=2, dim=1)

    # Cosine similarity matrix
    # sim_matrix[i, j] is the similarity between embeddings1[i] and embeddings2[j]
    sim_matrix = torch.matmul(embeddings1_norm, embeddings2_norm.T) / temperature

    # Positive pairs are on the diagonal (i.e., sim(embeddings1[i], embeddings2[i]))
    labels = torch.arange(batch_size, device=device)

    # Calculate cross-entropy loss
    # Loss for view1 predicting view2
    loss1 = F.cross_entropy(sim_matrix, labels)
    # Loss for view2 predicting view1 (symmetric)
    loss2 = F.cross_entropy(sim_matrix.T, labels)

    return (loss1 + loss2) / 2.0


def main():
    set_seed()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not torch.cuda.is_available() and BATCH_SIZE > 1 :
        print("Warning: CUDA not available. Training on CPU will be very slow. Consider reducing batch size or not using QLoRA.")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded.")

    # 2. Load Data
    texts = load_cora_data(FILE_PATH)
    dataset = TextDataset(texts, tokenizer, MAX_SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=min(4, os.cpu_count() // 2 if os.cpu_count() else 1))
    print(f"Data loaded. Dataset size: {len(dataset)}, DataLoader ready.")

    # 3. Configure BitsAndBytes for QLoRA and determine compute dtype
    bnb_config = None
    compute_dtype = torch.float16 # Default for CPU or if BF16 not supported

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
            print("Using compute dtype: torch.bfloat16 (BF16 is supported on CUDA)")
        else:
            print("Using compute dtype: torch.float16 (BF16 not supported on CUDA, falling back to FP16)")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        print("BitsAndBytesConfig configured for CUDA.")
    else:
        print("Warning: BitsAndBytes 4-bit quantization is not enabled for CPU. Model will be loaded in default precision (FP32 or FP16).")
        # compute_dtype remains float16 or could be set to float32 for CPU if preferred

    # 4. Load Base Model
    print(f"Loading base model: {MODEL_NAME}...")

    model_load_kwargs = {
        # "trust_remote_code": True # Uncomment if model requires it
    }

    if bnb_config and device.type == "cuda": # Only apply quantization if on CUDA and bnb_config is set
        model_load_kwargs["quantization_config"] = bnb_config
        # Explicitly map all layers to the current CUDA device to avoid disallowed offloading
        current_cuda_device_index = device.index if device.index is not None else 0
        model_load_kwargs["device_map"] = {"": current_cuda_device_index}
        # Suggested by accelerate warning, helps if some non-module buffers need offloading
        model_load_kwargs["offload_buffers"] = True
        print(f"Attempting to load quantized model onto cuda:{current_cuda_device_index} with explicit device_map.")
    elif device.type == "cuda": # No BNB config but on CUDA (e.g. if user disables BNB)
        current_cuda_device_index = device.index if device.index is not None else 0
        model_load_kwargs["device_map"] = {"": current_cuda_device_index}
        model_load_kwargs["torch_dtype"] = compute_dtype # Load in specified compute_dtype
        print(f"Attempting to load model in {compute_dtype} onto cuda:{current_cuda_device_index} (no quantization).")
    else: # CPU loading
        model_load_kwargs["device_map"] = {"": "cpu"}
        model_load_kwargs["torch_dtype"] = compute_dtype # Load in specified compute_dtype (usually float32 for CPU)
        print(f"Attempting to load model in {compute_dtype} onto CPU (no quantization).")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        **model_load_kwargs
    )

    base_model.config.use_cache = False # Important for training with LoRA
    if base_model.config.pad_token_id is None:
         base_model.config.pad_token_id = tokenizer.pad_token_id
    print("Base model loaded.")

    # 5. Prepare model for k-bit training and apply LoRA
    if bnb_config and device.type == "cuda": # Only prepare for kbit if actually quantized
        base_model = prepare_model_for_kbit_training(base_model)
        print("Model prepared for k-bit training.")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    print("PEFT model created with LoRA.")

    # 6. Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_training_steps = len(dataloader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_training_steps
    )
    print("Optimizer and scheduler configured.")

    # 7. Training Loop
    print("Starting fine-tuning...")
    model.train()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs1 = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_state1 = outputs1.hidden_states[-1]
            embeddings1 = pool_embeddings(last_hidden_state1, attention_mask, strategy=POOLING_STRATEGY)

            outputs2 = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_state2 = outputs2.hidden_states[-1]
            embeddings2 = pool_embeddings(last_hidden_state2, attention_mask, strategy=POOLING_STRATEGY)

            loss = info_nce_loss(embeddings1, embeddings2, TEMPERATURE)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item(), "avg_loss": total_loss / (batch_idx + 1)})

        avg_epoch_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Average Loss: {avg_epoch_loss:.4f}")

    print("Fine-tuning completed.")

    # 8. Save LoRA Adapters
    print(f"Saving LoRA adapters to {ADAPTERS_OUTPUT_DIR}...")
    model.save_pretrained(ADAPTERS_OUTPUT_DIR)
    tokenizer.save_pretrained(ADAPTERS_OUTPUT_DIR)
    print("LoRA adapters saved.")

    # 9. Generate Embeddings for all texts
    print("Generating embeddings for all texts...")
    model.eval()
    all_embeddings = []

    generation_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=min(4, os.cpu_count() // 2 if os.cpu_count() else 1))

    with torch.no_grad():
        for batch in tqdm(generation_dataloader, desc="Generating Embeddings"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            embeddings = pool_embeddings(last_hidden_state, attention_mask, strategy=POOLING_STRATEGY)
            all_embeddings.append(embeddings.cpu())

    final_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Embeddings generated. Shape: {final_embeddings.shape}")

    # 10. Save Embeddings
    torch.save(final_embeddings, EMBEDDINGS_OUTPUT_FILE)
    print(f"Embeddings saved to {EMBEDDINGS_OUTPUT_FILE}")
    print(f"Script finished. Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()