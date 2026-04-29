# i23-2552-NLP-Assignment3
**Muhammad Moiz Khalid  23i-2552  BDS-6C**

A three-stage NLP pipeline built entirely from scratch on the Amazon Reviews dataset. No pretrained models. No `nn.Transformer`, `nn.MultiheadAttention`, or `nn.TransformerEncoder`.

---

## Overview

| Stage | Component | Task |
|-------|-----------|------|
| A | Encoder-Only Transformer | Multi-task sentiment + length classification |
| B | Retrieval Module | Cosine similarity search over learned embeddings |
| C | Decoder-Only Transformer | RAG-enhanced explanation generation |

---

## Dataset

- **Categories:** Cell Phones, Beauty, Home & Kitchen
- **Size:** ~36,000 reviews (12,000 per category)
- **Split:** 70% train / 15% val / 15% test
- **Labels:** Star rating (1–5) → Sentiment (Negative / Neutral / Positive)

---

## Project Structure

```
├── i232552_NLP_Assignment3.ipynb   # Main notebook
├── Report
├── models/
│   ├── encoder.pt                  # Trained encoder weights
│   ├── decoderRag.pt               # Trained decoder (with RAG)
│   └── decoderNoRag.pt             # Trained decoder (no RAG baseline)
└── results/
    ├── train_embeddings.pkl        # Encoder CLS embeddings for retrieval
    ├── encoderLearning.png         # Encoder training curves
    └── decoderLearning.png         # Decoder training curves
```

---

## How to Run

1. Upload the three dataset files (`cellphones.json.gz`, `beauty.json.gz`, `home.json.gz`) to Google Drive.
2. Open `i232552_NLP_Assignment3.ipynb` in Google Colab.
3. Mount Drive and run all cells top to bottom.

> GPU runtime recommended (Runtime → Change runtime type → T4 GPU).

---

## Model Details

**Encoder**
- 2 encoder blocks, `d_model=128`, 4 attention heads, `d_ff=256`, dropout=0.1
- CLS token embedding used for classification and retrieval
- Combined cross-entropy loss over both tasks

**Retrieval**
- L2-normalised CLS embeddings, cosine similarity via dot product
- Configurable top-k (default k=3)

**Decoder**
- 2 decoder blocks, same dimensions as encoder
- Causal (triangular) mask prevents attention to future tokens
- Input template: `[REVIEW] ... [SENT] ... [LEN] ... [CTX] ... [EXP] <BOS>`
- Greedy autoregressive generation, stops at `<EOS>` or 30 new tokens

---

## Results

| Task | Metric |
|------|--------|
| Sentiment Classification | Per-class F1 reported on test set |
| Length Classification | Per-class F1 reported on test set |
| Decoder Perplexity (RAG) | Reported on test set |
| Decoder Perplexity (No RAG) | Reported on test set |

---

## Dependencies

```
torch, numpy, matplotlib, scikit-learn
```
All other logic (tokenization, attention, positional encoding) is implemented from scratch.
