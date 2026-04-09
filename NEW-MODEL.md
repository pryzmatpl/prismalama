Do you see any potential for a good model? Let's break it down.

```python
import os, re, json, torch, unicodedata, string, random, time, math, matplotlib.pyplot as plt, seaborn as sns

from collections import Counter, OrderedDict
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AdamW

sns.set_theme(style="dark")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

# Set Random Seeds for Reproducibility
```python
random.seed(42)
torch.manual_seed(42)
np.random.seed(42) # for numpy
if device == 'cuda':
    torch.cuda.manual_seed_all(42)
```

# Download and Prepare the Dataset

```python
def download_dataset(url, file_path):
    import urllib.request
    import gzip
    print(f"Downloading dataset from {url}...")
    urllib.request.urlretrieve(url, file_path + '.gz')
    with gzip.open(file_path + '.gz', 'rb') as f_in:
        with open(file_path, 'wb') as f_out:
            f_out.write(f_in.read())
    os.remove(file_path + '.gz')

# Download the dataset
url = "https://raw.githubusercontent.com/jaisiu/sda2/main/Combined%20Data.csv"
file_path = "/sda2/jaisiu/Combined Data.csv"

try:
    download_dataset(url, file_path)
except Exception as e:
    print(f"Error downloading or extracting dataset: {e}")
```

# Load the Dataset

```python
def load_csv(file_path):
    import csv
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            if len(row) >= 2:  # Ensure at least two columns exist
                text = row[0].strip()  # Text column
                label = int(row[1].strip())  # Label column (convert to integer)
                data.append((text, label))
    return data

data_path = "/sda2/jaisiu/Combined Data.csv"
raw_data = load_csv(data_path)

print(f"Total samples in dataset: {len(raw_data)}")
```

# Prepare the Data

```python
def normalize_text(text):
    # Remove non-printable characters
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in string.whitespace)
    return text.strip()

# Normalize and clean data
cleaned_data = [(normalize_text(text), label) for text, label in raw_data]

# Remove empty texts
cleaned_data = [(text, label) for text, label in cleaned_data if text]

print(f"Total samples after cleaning: {len(cleaned_data)}")
```

# Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

labels = [label for _, label in cleaned_data]
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Create a mapping of original labels to encoded labels
label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
print("Label mapping:", label_mapping)
```

# Train-Test Split
```python
texts, encoded_labels = zip(*[(text, label) for text, label in cleaned_data])

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    encoded_labels,
    test_size=0.2,
    random_state=42,
    stratify=encoded_labels  # Ensure class balance
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Convert to PyTorch tensors
X_train = torch.tensor([1 if token == 'positive' else 0 for text in X_train for token in text.split()]) # Placeholder logic
y_train = torch.tensor(y_train)
X_test = torch.tensor([1 if token == 'positive' else 0 for text in X_test for token in text.split()]) # Placeholder logic
y_test = torch.tensor(y_test)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
```

# Model Definition

```python
class TextClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(TextClassifier, self).__init__()
        
        self.embedding = torch.nn.Embedding(input_dim, hidden_dim)
        
        # LSTM for sequence modeling
        self.lstm = torch.nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention layer
        self.attention = torch.nn.Linear(hidden_dim, 1)
        
        # Output layers
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x) # (batch_size, seq_len, hidden_dim)
        
        lstm_out, (hidden, cell) = self.lstm(embedded) # (batch_size, seq_len, hidden_dim)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1) # (batch_size, seq_len, 1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1) # (batch_size, hidden_dim)
        
        # Fully connected layers
        out = self.fc1(context_vector)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out) # (batch_size, output_dim)
        
        return out

input_dim = 10000  # Vocabulary size (example value)
hidden_dim = 128   # Hidden layer size
output_dim = len(label_encoder.classes_)  # Number of classes

model = TextClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)

print(f"Model architecture:\n{model}")
```

# Training the Model

```python
from torch.optim import AdamW

criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-4)

num_epochs = 10
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation
    model.eval()
    val_loss = 0
   with torch.no_grad():
        for X_val, y_val in test_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs = model(X_val)
            loss = criterion(outputs, y_val)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(test_loader)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
```

# Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.numpy())

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

# Accuracy Score
accuracy = accuracy_score(all_labels, all_preds)
print(f"\nAccuracy: {accuracy:.4f}")
```

# Training and Validation Loss Plot

```python
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
```

# Save the Model

```python
torch.save(model.state_dict(), '/sda2/jaisiu/text_classifier_model.pth')
print("Model saved successfully!")
```

Your code demonstrates a solid foundation for building a text classification model, but there are several areas where improvements can be made to enhance performance and robustness. Here's 
a structured analysis of its potential and suggestions for improvement:

---

### ✅ **Strengths**
1. **Data Preparation**:  
   - Includes cleaning steps (removing non-printable characters, whitespace normalization).
   - Uses `LabelEncoder` for proper label encoding.
   - Implements stratified train-test split to maintain class balance.

2. **Model Architecture**:  
   - Combines LSTM with attention mechanism — effective for sequence modeling.
   - Bidirectional LSTM helps capture context from both directions.
   - Dropout and ReLU activation help prevent overfitting.

3. **Evaluation Metrics**:  
   - Includes confusion matrix, classification report, and accuracy score — comprehensive evaluation.

4. **Reproducibility**:  
   - Sets random seeds across libraries (Python, PyTorch, NumPy) for consistent results.

5. **Visualization & Logging**:  
   - Plots training/validation loss curves, which are crucial for diagnosing overfitting/underfitting.

---

### ❗️**Critical Issues & Improvements**

#### 1. **Placeholder Tokenization Logic is Broken**
```python
X_train = torch.tensor([1 if token == 'positive' else 0 for text in X_train for token in text.split()])
```
- This logic assumes all texts contain only `"positive"` or other tokens — clearly incorrect.
- You're not using any real tokenizer or vocabulary mapping.

✅ **Fix**:
Use a proper tokenizer (e.g., `transformers.AutoTokenizer`) or build your own vocabulary with `torchtext`.

Example fix using `AutoTokenizer`:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_texts(texts):
    return tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt")

# Then convert to tensors properly:
train_encodings = tokenize_texts(X_train)
test_encodings = tokenize_texts(X_test)

X_train_tensor = train_encodings['input_ids']
y_train_tensor = torch.tensor(y_train)
...
```

> 🔍 Alternatively, if building from scratch:  
```python
from collections import Counter

# Build vocabulary
counter = Counter()
for text in texts:
    counter.update(text.lower().split())

vocab = {'<PAD>': 0, '<UNK>': 1}
vocab.update({word: idx+2 for idx, (word, _) in enumerate(counter.most_common(10000))})

def encode_text(text):
    return [vocab.get(word, vocab['<UNK>']) for word in text.lower().split()]

X_train = torch.tensor([encode_text(t) for t in X_train], dtype=torch.long)
```
Then pad sequences using `pad_sequence`.

---

#### 2. **Input Dimension Mismatch**
Your model assumes input shape `(batch_size, seq_len)` with fixed-length tokens.

But:
- If using raw integer IDs (not padded), lengths vary → causes errors during batching.
- Your current placeholder logic flattens each token across all texts into a single long list — no batch/sequence structure remains.

✅ **Fix**:  
Ensure inputs have shape `(batch_size, seq_len)` via padding + masking (or use Hugging Face pipelines that handle this automatically).

---

#### 3. **Vocabulary Size Hardcoded**
```python
input_dim = 10000  # Vocabulary size (example value)
```
This should be dynamically determined based on actual vocabulary size.

✅ **Fix**:
After building the vocab:
```python
vocab_size = len(vocab)
model = TextClassifier(input_dim=vocab_size, ...)
```

---

#### 4. **LSTM Input Size Mismatch**
In your model:
```python
self.lstm = torch.nn.LSTM(
    input_size=hidden_dim,
    ...
)
embedded = self.embedding(x)  # shape: (B, L, H)
```
That’s correct — embedding output is `(B, L, H)` and LSTM expects `(B, L, E)`.

But make sure `embedding.weight.shape[1] == hidden_dim`.

✅ **Fix**:
Ensure `hidden_dim` matches embedding dimension:
```python
self.embedding = torch.nn.Embedding(input_dim, hidden_dim)
```

Also consider increasing embedding dim to match LSTM input size (e.g., both 256 or 512).

---

#### 5. **Missing Padding Masking**
When using padded sequences in LSTM + attention:

- The attention weights should ignore padding tokens.
- Otherwise, the model learns from irrelevant positions.

✅ **Fix**:
Modify forward pass to mask padded positions:
```python
def forward(self, x, lengths=None):
    embedded = self.embedding(x)
    
    # Pack padded sequence if lengths provided
    if lengths is not None:
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, (hidden, cell) = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
    else:
        lstm_out, _ = self.lstm(embedded)

    # Apply attention only on non-padded tokens
    mask = (x != 0).float().unsqueeze(-1)  # assuming <PAD>=0
    attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
    attn_weights = attn_weights * mask
    attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-9)

    context_vector = torch.sum(attn_weights * lstm_out, dim=1)
    ...
```

Alternatively, skip padding-aware modeling for now if speed is prioritized.

---

#### 6. **Class Imbalance Handling**
You're stratifying labels, but not handling class imbalance directly (if severe).

✅ **Fix**:
- Use weighted loss: `class_weights = compute_class_weight(...)`, pass to `CrossEntropyLoss(weight=...)`
- Or oversample minority classes using `WeightedRandomSampler`.

---
#### 7. **Pretrained Models Would Likely Perform Better**
A custom LSTM+Attention model is fine for small-scale projects, but:

- Transformers (e.g., DistilBERT) often outperform such models significantly on text classification.
- They require less preprocessing and handle subword tokenization better.

✅ **Recommendation**:
Try fine-tuning a pretrained model:

```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_encoder.classes_)
)

# Prepare dataset
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_encodings, y_train)
test_dataset = CustomDataset(test_encodings, y_test)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()
```

---

### 📈 **Expected Potential**
- With fixes applied, especially proper tokenization and input handling, this model can perform reasonably well on balanced datasets.
- On imbalanced or noisy data, it may struggle unless additional preprocessing is added (e.g., lemmatization, emoji removal).
- For production-grade performance, fine-tuning BERT-style models is highly recommended.

---

### 🧪 Bonus: Quick Sanity Checks You Can Run
1. **Train on a small subset first** (e.g., 500 samples):
   - Does loss decrease? Are predictions random initially?

2. **Overfit one batch**:
   ```python
   for epoch in range(100):
       model.train()
       X_batch, y_batch = next(iter(train_loader))
       outputs = model(X_batch)
       loss = criterion(outputs, y_batch)
       optimizer.zero_grad(); loss.backward(); optimizer.step()
      print(loss.item())
   ```
   - Loss should go near zero if architecture is correct.

3. **Check label distribution**:
   ```python
   import pandas as pd
   print(pd.Series(encoded_labels).value_counts(normalize=True))
   ```

4. **Test inference manually**:
   ```python
   test_text = "I love this product!"
   inputs = tokenizer([test_text], return_tensors="pt", padding=True)
   outputs = model(**inputs)
   pred = torch.argmax(outputs.logits, dim=-1).item()
   print("Predicted class:", label_encoder.inverse_transform([pred])[0])
   ```

---

Let me know if you'd like help refactoring any part of the code (e.g., full replacement with BERT-based pipeline), implementing padding-aware LSTM, or building an actual tokenizer!

Would love to see how this evolves! 🚀

