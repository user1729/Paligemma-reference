# Paligemma-reference
### 🧠 First, the Basics: What is Attention?

In attention mechanisms, we work with:

- **Q** = Query
- **K** = Key
- **V** = Value

Think of it like this:

> "Query asks a question, Keys hold the possible answers, and Values are the content we get if a Key is relevant."

The attention mechanism uses **Q, K, and V** to compute how much attention each word (or token) should give to others.

---

### 🤹‍♂️ Multi-Head Attention (MHA)

**MHA = Multi-Head Attention**

- The model **splits the attention into multiple "heads"**.
- Each head gets a different set of Q, K, V projections.
- Each head learns to focus on different parts of the input.

This improves the model’s ability to understand complex relationships.

#### Imagine:

You’re analyzing a sentence with 8 different people (heads), each focusing on a different aspect—grammar, emotion, topic, etc. Then, you combine all their insights.

**Key point:**  

- Each head has its **own Q, K, and V** matrices.

---

### 🪄 MQA – Multi-Query Attention

**MQA = Multi-Query Attention**

- Here, **Q is still separate for each head**, but all heads **share the same K and V**.

Why do this?

- It **saves memory and computation**, especially helpful for large models and long sequences.
- You get a speed-up because you don’t need to compute separate K and V for each head.

#### Analogy:

Everyone (heads) asks different questions (Q), but they all look at the **same set of facts** (shared K and V).

---

### 🧙 GQA – Grouped-Query Attention

**GQA = Grouped-Query Attention**

- It’s **in between** MHA and MQA.
- You **group the heads**, and each group shares the same K and V.
- For example, 12 heads → 4 groups → each group has 3 heads with shared K and V.

Why?

- You still save memory (less than MHA), but get more diversity than MQA.
- It balances efficiency and expressiveness.

#### Analogy:

Think of a team of analysts (heads) divided into smaller groups. Each group has a shared knowledge base (K and V), but each analyst has their own perspective (Q).

---

### Summary Table:

| Method  | Q                 | K                | V                | Pros                                       | Cons            |
| ------- | ----------------- | ---------------- | ---------------- | ------------------------------------------ | --------------- |
| **MHA** | Separate per head | Separate         | Separate         | Most flexible, expressive                  | Most expensive  |
| **MQA** | Separate          | Shared           | Shared           | Fast, memory efficient                     | Less expressive |
| **GQA** | Separate          | Shared per group | Shared per group | Trade-off between speed and expressiveness | Medium cost     |

---

Let me know if you'd like a visual explanation or a code example to go along with this!

Llama uses GQA

---

## 🎯 What's the `[CLS]` Token?

In models like **BERT**, `[CLS]` is a **special token prepended** to the input sequence. Its job is to serve as a **summary representation** of the entire input.

### Example:

```plaintext
Input: "Hello world"
Tokenized: [CLS] Hello world [SEP]
IDs:       [101, 7592, 2088, 102]
```

- `[CLS]` = ID 101 in BERT vocab.
- BERT will output a hidden state for **each token**, including `[CLS]`.

👉 The model is trained to make classification decisions **based only on the final hidden state of the `[CLS]` token**.

---

## 🧠 Why Does It Work?

Because of self-attention, every token can attend to every other token (including `[CLS]`). So during training, the model learns to **encode the meaning of the entire sentence** into that `[CLS]` token's final embedding.

This embedding is like a **summary vector** of the whole input.

---

## 🏗️ End-to-End Training (Sentence Classification)

Let’s say you want to **classify the sentiment** of a sentence: "This movie is great!" → positive.

### Step-by-Step Pipeline:

---

### ✅ 1. **Tokenization**

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentence = "This movie is great!"
encoded = tokenizer(sentence, return_tensors="pt", padding=True)
```

`encoded` includes:

- `input_ids`: [101, ..., 102]
- `attention_mask`: [1, 1, ..., 1]

---

### ✅ 2. **Pass Through BERT**

```python
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-uncased")
outputs = model(**encoded)
```

- `outputs.last_hidden_state`: shape `(batch_size, seq_len, hidden_dim)`  
- Extract `[CLS]` token representation:

```python
cls_output = outputs.last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_dim)
```

Why `[:, 0, :]`? Because `[CLS]` is always the **first token** → index 0.

---

### ✅ 3. **Add a Classification Head**

Now take the `[CLS]` embedding and pass it through a classifier layer (like a linear layer + softmax):

```python
import torch.nn as nn

classifier = nn.Linear(model.config.hidden_size, num_labels)
logits = classifier(cls_output)  # shape: (batch_size, num_labels)
```

---

### ✅ 4. **Compute Loss & Train**

Assume you're using cross-entropy loss for a classification task:

```python
from torch.nn import CrossEntropyLoss

labels = torch.tensor([1])  # assume binary classification, 1 = positive
loss_fn = CrossEntropyLoss()
loss = loss_fn(logits, labels)
loss.backward()
optimizer.step()
```

Over time, the model:

- Learns to encode meaningful sentence features into `[CLS]`
- Learns how to map `[CLS]`’s output to the correct class via the classifier head

---

## 📈 Summary Flow:

```plaintext
[Text] 
 → [Tokenize + Add CLS & SEP] 
 → [BERT Encoder]
 → [CLS hidden state]
 → [Classifier layer (Linear + Softmax)]
 → [Loss function]
 → [Backpropagation & Optimization]
```

---

## 🔄 Fine-tuning vs. Pretraining

- During **pretraining**, `[CLS]` might be used for tasks like **Next Sentence Prediction (NSP)**.
- During **fine-tuning**, we reuse that setup and replace the pretraining objective with a task-specific one (e.g., sentiment classification, spam detection, etc.).

---

If you want, I can show you a full training loop or plug this into Hugging Face’s `Trainer` class. Or maybe you're curious how this compares with pooling strategies like mean pooling?
