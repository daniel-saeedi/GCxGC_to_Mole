# %%
# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

sample_SMILES = "<s>C"

t = tokenizer(sample_SMILES, return_tensors="pt")

output = model(**t)


def log_epoch(log_file, msg):
    # Get the current timestamp    
    # Format the log message
    log_entry = f"{msg}\n"
    
    # Write the log message to a file
    try:
        with open(log_file, 'a') as file:
            file.write(log_entry)
    except IOError as e:
        # Handle file write errors
        print(f"Error writing to log file: {e}")


# %%
import torch
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

class BERT_GCxGC(nn.Module):
    def __init__(self, base_model, hidden_dim, output_dim):
        super(BERT_GCxGC, self).__init__()
        self.base_model = base_model

        # Predicts Molecular Weight or M/Z
        self.m_z = nn.Sequential(
            nn.Linear(base_model.config.vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)
        # Predicting retention time 1
        self.rt1 = nn.Sequential(
            nn.Linear(base_model.config.vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)
        # Predicting retention time 2
        self.rt2 = nn.Sequential(
            nn.Linear(base_model.config.vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        cls_token = outputs.logits[:, 0, :]  # Get the CLS token(<s> token)
        m_z = self.m_z(cls_token)
        rt1 = self.rt1(cls_token)
        rt2 = self.rt2(cls_token)
        return m_z, rt1, rt2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the base model
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(device)

# Create the model with MLPs
hidden_dim = 128  # Hidden dimension size for each MLP
output_dim = 1  # Output dimension for each MLP
custom_model = BERT_GCxGC(model, hidden_dim, output_dim).to(device)


# Example usage
# sample_SMILES = "<s>CC1=CC(=CC=C1)S(=O)(=O)NC2=CC=C(C=C2)S(=O)(=O)NC(C)C"
# inputs = tokenizer(sample_SMILES, return_tensors="pt")
# outputs = custom_model(**inputs)
# print(outputs)  # Outputs from the three MLPs


# %%
all_dataset = pd.read_csv('results/training_set_march20.csv')


# %%
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# Assuming you have a dataset in the form of a list of tuples [(smiles, m_z, rt1, rt2), ...]
class SMILESDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = '<s>'+self.data.iloc[idx]['Canonical_SMILES']
        m_z = self.data.iloc[idx]['m_z']
        rt1 = self.data.iloc[idx]['1st Dimension Time (s)']
        rt2 = self.data.iloc[idx]['2nd Dimension Time (s)']

        # Tokenize the SMILES and pad to the max length
        inputs = self.tokenizer(smiles, padding='max_length', truncation=True, return_tensors="pt")

        # Remove the batch dimension that the tokenizer adds by default
        input_ids = inputs.input_ids.squeeze(0)

        # Your targets as a tensor
        targets = torch.tensor([m_z, rt1, rt2], dtype=torch.float32, device=device)
        
        return input_ids.to(device), targets.to(device)


from sklearn.model_selection import train_test_split

# Split the dataset into training and test sets
train_val_data, test_data = train_test_split(all_dataset, test_size=0.1, random_state=42)

# Then split the training+validation set into separate training and validation sets
train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42) # 0.25 x 0.8 = 0.2


# Create instances of SMILESDataset for each set
train_dataset = SMILESDataset(train_data, tokenizer)
val_dataset = SMILESDataset(val_data, tokenizer)
test_dataset = SMILESDataset(test_data, tokenizer)

# Create DataLoaders for each set
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Loss Function
criterion = nn.MSELoss()

# Optimizer
optimizer = Adam(custom_model.parameters(), lr=1e-4)

# Training Loop
# Training, Validation, and Testing Loop
for epoch in range(1000):
    # Training phase
    custom_model.train()
    train_loss = 0
    for input_ids, targets in train_dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        m_z_pred, rt1_pred, rt2_pred = custom_model(input_ids)
        loss = criterion(m_z_pred, targets[:, 0]) + criterion(rt1_pred, targets[:, 1]) + criterion(rt2_pred, targets[:, 2])
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Average loss for the training set
    avg_train_loss = train_loss / len(train_dataloader)
    
    # Validation phase
    custom_model.eval()
    val_loss = 0
    with torch.no_grad():
        for input_ids, targets in val_dataloader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            m_z_pred, rt1_pred, rt2_pred = custom_model(input_ids)
            loss = criterion(m_z_pred, targets[:, 0]) + criterion(rt1_pred, targets[:, 1]) + criterion(rt2_pred, targets[:, 2])
            val_loss += loss.item()
    
    # Average loss for the validation set
    avg_val_loss = val_loss / len(val_dataloader)
    
    # Print training and validation loss

    log_epoch('results/chemberta_finetuned.log', f"Epoch {epoch} | Training loss: {avg_train_loss} | Validation loss: {avg_val_loss}")
    print(f"Epoch {epoch} | Training loss: {avg_train_loss} | Validation loss: {avg_val_loss}")
    
# Test phase would be similar to validation but after all epochs are done to test the final model performance.
custom_model.eval()
test_loss = 0
with torch.no_grad():
    for input_ids, targets in test_dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        m_z_pred, rt1_pred, rt2_pred = custom_model(input_ids)
        loss = criterion(m_z_pred, targets[:, 0]) + criterion(rt1_pred, targets[:, 1]) + criterion(rt2_pred, targets[:, 2])
        test_loss += loss.item()

# Average loss for the validation set
avg_test_loss = test_loss / len(val_dataloader)

print(f'Test Loss: {avg_test_loss}')

# Save the fine-tuned model
# torch.save(custom_model.state_dict(), "chemberta_with_mlps_finetuned.pth")






