# Spam_Classification_using_Encoder_LLMs_with_Linear_Probing-
we will use encoder Large Language Models (LLMs) for spam classification. We will leverage the rich features of pre-trained LLMs without fine-tuning them. Instead, we will freeze the LLM weights and train a lightweight classifier head (MLP) on top for spam classification.
# Spam Classification using Encoder LLMs with Linear Probing

This repository accompanies the notebook **`notebooks/Spam Classification using Encoder LLMs with Linear Probing.ipynb`**.  
It demonstrates **feature extraction with pre-trained encoder-only Transformers** (BERT / DistilBERT / ALBERT) and **linear probing** (Logistic Regression (linear probe)) for **spam vs ham** text classification.

---

## 🚀 What’s inside
- Load raw SMS/email text, clean it, and split into train/valid/test.
- Encode texts with **pre-trained Transformer encoders** (frozen) to obtain **fixed-size embeddings** (e.g., CLS token or mean pooling).
- Train **lightweight linear classifiers** (a “linear probe”) on top of those embeddings.
- Evaluate with accuracy, precision/recall/F1, plus confusion matrix.
- Compare encoders and pooling strategies.

---

## 📦 Data
- **Dataset(s):** Enron Email, Hugging Face Datasets (loaded via `datasets`)

**Expected columns:**
- `text` – the message body
- `label` – categorical class, e.g., `ham`/`spam` or `0`/`1`

You can point the notebook to your CSV path at the top (look for a `DATA_PATH` or a similar variable).

---

## 🧠 Models & Approach
- **Encoders:** BERT, DistilBERT, ALBERT
- **Pooling:** CLS representation or mean pooling over token embeddings (as implemented in the notebook).
- **Linear Probe:** Logistic Regression (linear probe)
- **Why linear probing?** We keep the encoder **frozen** and train a small linear head for speed and strong baselines without full fine-tuning.

---

## 📊 Results
| Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 85 | 99 | 98.6 | 0 |

> Tip: Rerun the notebook to reproduce metrics; set seeds for reproducibility and ensure consistent train/valid/test split.

---

## 🛠️ Setup

### 1) Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Run the notebook
```bash
jupyter notebook notebooks/Spam\ Classification\ using\ Encoder\ LLMs\ with\ Linear\ Probing.ipynb
```
Run cells in order to prepare data, build embeddings, train the linear probe(s), and evaluate.

-
