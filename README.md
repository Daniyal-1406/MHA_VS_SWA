# Transformers from Scratch — AG News Classifier

End-to-end implementation of two Transformer variants for text classification on the [AG News dataset](https://huggingface.co/datasets/ag_news).

## What's in this repo

| File / Folder | Description |
|---|---|
| `model.py` | Multi-Head Attention (MHA) Transformer encoder, built from scratch in PyTorch |
| `model_swa.py` | Sliding Window Attention (SWA) Transformer — true O(N·w·d) gather-based implementation |
| `generate_notebook.py` | Script that generates `ag_news_experiment.ipynb` for Google Colab training |
| `api/main.py` | Minimal FastAPI backend — loads both models, exposes `/predict` and `/health` |
| `api/requirements.txt` | Python dependencies for the backend |
| `frontend/` | React + Vite frontend with model toggle (MHA / SWA), example inputs, and confidence bars |

## Architecture

Both models share the same skeleton (embedding → positional encoding → N encoder layers → mean pool → classifier), differing only in the attention block:

- **MHA**: Full `N × N` attention matrix — O(N² · d) time, O(N²) space
- **SWA**: Each token attends only to its `w`-token neighbourhood via advanced indexing gather — O(N · w · d) time, O(N · w) space

## Results on AG News (20k train / 7.6k test, 5 epochs, T4 GPU)

| Model | Test Acc | Train Loss | Avg Epoch Time |
|-------|----------|-----------|----------------|
| MHA   | 86.64%   | 0.2514    | ~21.1s         |
| SWA (w=32) | 85.46% | 0.2452 | ~23.2s        |

## Running locally

### 1. Train on Colab

```bash
python generate_notebook.py   # creates ag_news_experiment.ipynb
# Upload to Google Colab, run all cells, then download:
#   mha_model.pt, swa_model.pt, vocab.json
# Place all three in api/
```

### 2. Start the backend

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Start the frontend

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

## References

1. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
2. Jolicoeur-Martineau et al., *Sliding-Window Beats Linear Attention*, arXiv 2026
3. Zhang, Zhao & LeCun, *Character-level Convolutional Networks for Text Classification*, NeurIPS 2015
