import numpy as np
import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import csv
import sys
from datasets import load_dataset
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = load_dataset("ealvaradob/phishing-dataset", "combined_reduced", trust_remote_code=True)
df = dataset['train'].to_pandas()
df = df.drop_duplicates().reset_index(drop=True)

text_col = "text"
label_col = "label"

plt.figure(figsize=(8, 6))
label_counts = df[label_col].value_counts()
label_counts.plot(kind='bar', color=['green', 'red'])
plt.title("Label Distribution")
plt.xlabel("Label (0=Legit, 1=Phishing)")
plt.ylabel("Count")
plt.savefig("label_distribution.png")
print(f"Label distribution: {label_counts.to_dict()}")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class PhishingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return (
            encoding["input_ids"].squeeze(),
            encoding["attention_mask"].squeeze(),
            torch.tensor(float(label), dtype=torch.float32)
        )

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df[text_col], df[label_col], 
    test_size=0.2, 
    random_state=42, 
    stratify=df[label_col]
)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"Class weights: {class_weights}")

train_ds = PhishingDataset(train_texts.values, train_labels.values, tokenizer)
val_ds = PhishingDataset(val_texts.values, val_labels.values, tokenizer)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

class ImprovedPhishingModel(torch.nn.Module):
    def __init__(self, hidden_size=768, dropout_rate=0.5):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 512),
            torch.nn.LayerNorm(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        pooled = outputs.last_hidden_state[:, 0]
        
        features = self.feature_extractor(pooled)
        logits = self.classifier(features).squeeze(1)
        
        return logits

model = ImprovedPhishingModel().to(device)

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights[1]/class_weights[0])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

num_epochs = 10
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0.1 * total_steps,
    num_training_steps=total_steps
)

train_losses, val_losses, accuracies = [], [], []
best_accuracy = 0
patience = 3
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
    for msg, mask, lbl in progress_bar:
        msg, mask, lbl = msg.to(device), mask.to(device), lbl.to(device)
        
        optimizer.zero_grad()
        logits = model(msg, mask)
        loss = criterion(logits, lbl)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    val_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for msg, mask, lbl in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
            msg, mask, lbl = msg.to(device), mask.to(device), lbl.to(device)
            logits = model(msg, mask)
            loss = criterion(logits, lbl)
            val_loss += loss.item()
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == lbl).sum().item()
            total += lbl.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())
            
    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total

    tp = np.sum((np.array(all_preds) == 1) & (np.array(all_labels) == 1))
    fp = np.sum((np.array(all_preds) == 1) & (np.array(all_labels) == 0))
    fn = np.sum((np.array(all_preds) == 0) & (np.array(all_labels) == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    accuracies.append(accuracy)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, Acc: {accuracy:.2f}%, "
          f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), "sentra_model_best.pth")
        tokenizer.save_pretrained("./sentra_tokenizer")
        print(f"New best model saved with accuracy: {best_accuracy:.2f}%")
        patience_counter = 0
    else:
        patience_counter += 1
        
    if (epoch + 1) % 2 == 0:
        torch.save(model.state_dict(), f"sentra_model_epoch_{epoch+1}.pth")
    
    if patience_counter >= patience:
        print(f"Early stopping after {epoch+1} epochs!")
        break

plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracies, label="Validation Accuracy", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy Over Time")
plt.legend()

plt.tight_layout()
plt.savefig("training_results.png")
plt.show()

print(f"Training completed! Best accuracy: {best_accuracy:.2f}%")