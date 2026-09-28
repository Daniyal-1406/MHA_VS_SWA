import os, sys, re, json
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── import model building blocks from parent directory ────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model import (InputEmbedding, PositionalEncoding, EncoderBlock,
                   MultiHeadAttention, FeedForwardBlock, Encoder, ClassificationHead)
from model_swa import SlidingWindowAttention

# ── app setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="AG News Classifier")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── constants ─────────────────────────────────────────────────────────────────
CLASSES    = ["World", "Sports", "Business", "Sci/Tech"]
PAD_TOKEN  = "<pad>"
UNK_TOKEN  = "<unk>"
VOCAB_PATH = os.path.join(os.path.dirname(__file__), "vocab.json")
MHA_PATH   = os.path.join(os.path.dirname(__file__), "mha_model.pt")
SWA_PATH   = os.path.join(os.path.dirname(__file__), "swa_model.pt")

# ── vocab ─────────────────────────────────────────────────────────────────────
if not os.path.exists(VOCAB_PATH):
    print(f"[!] vocab.json not found. Serving /health only until you add it.")
    vocab = {"<pad>": 0, "<unk>": 1}   # dummy so app can still start
else:
    vocab = json.load(open(VOCAB_PATH))

def encode(text: str, max_len: int = 256) -> list[int]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    ids    = [vocab.get(t, vocab[UNK_TOKEN]) for t in tokens]
    ids    = ids[:max_len] if len(ids) > max_len else ids + [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return ids

# ── model definitions ─────────────────────────────────────────────────────────
class MHAClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, h=8, d_ff=512,
                 num_layers=2, seq_len=256, num_classes=4, dropout=0.1):
        super().__init__()
        self.embedding  = InputEmbedding(d_model, vocab_size)
        self.pos_enc    = PositionalEncoding(d_model, seq_len, dropout)
        self.encoder    = Encoder(d_model, [
            EncoderBlock(d_model, MultiHeadAttention(d_model, h, dropout),
                         FeedForwardBlock(d_model, d_ff, dropout), dropout)
            for _ in range(num_layers)
        ])
        self.classifier = ClassificationHead(d_model, num_classes)

    def forward(self, x):
        return self.classifier(self.encoder(self.pos_enc(self.embedding(x)), None))


class SWAClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, h=8, d_ff=512,
                 num_layers=2, seq_len=256, num_classes=4, dropout=0.1, window_size=32):
        super().__init__()
        self.embedding  = InputEmbedding(d_model, vocab_size)
        self.pos_enc    = PositionalEncoding(d_model, seq_len, dropout)
        self.encoder    = Encoder(d_model, [
            EncoderBlock(d_model, SlidingWindowAttention(d_model, h, window_size, dropout),
                         FeedForwardBlock(d_model, d_ff, dropout), dropout)
            for _ in range(num_layers)
        ])
        self.classifier = ClassificationHead(d_model, num_classes)

    def forward(self, x):
        return self.classifier(self.encoder(self.pos_enc(self.embedding(x)), None))


# ── load models ───────────────────────────────────────────────────────────────
vocab_size = len(vocab)
models     = {}

for name, cls, path in [("mha", MHAClassifier, MHA_PATH),
                         ("swa", SWAClassifier, SWA_PATH)]:
    m = cls(vocab_size)
    if os.path.exists(path):
        m.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"[✓] {name.upper()} weights loaded from {path}")
    else:
        print(f"[!] {name.upper()} weights not found — using random init ({path})")
    m.eval()
    models[name] = m

# ── API schemas ───────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text:  str
    model: str = "mha"   # "mha" or "swa"

# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "available_models": list(models)}

@app.post("/predict")
def predict(req: PredictRequest):
    if req.model not in models:
        raise HTTPException(status_code=400, detail=f"model must be 'mha' or 'swa'")

    x      = torch.tensor([encode(req.text)], dtype=torch.long)
    with torch.no_grad():
        probs  = torch.softmax(models[req.model](x), dim=-1)[0].tolist()

    pred   = int(torch.tensor(probs).argmax())
    return {
        "model":           req.model.upper(),
        "predicted_class": CLASSES[pred],
        "class_index":     pred,
        "probabilities":   {c: round(p, 4) for c, p in zip(CLASSES, probs)},
    }
