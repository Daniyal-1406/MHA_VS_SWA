import json

def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

def md(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [source]
    }

cells = []

# ── 1. GPU check ──────────────────────────────────────────────────────────────
cells.append(code(
    "import torch\n"
    "\n"
    "print(torch.cuda.is_available())\n"
    "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\")"
))

# ── 2. Install datasets ───────────────────────────────────────────────────────
cells.append(code("!pip install -q datasets"))

# ── 3. Load dataset ───────────────────────────────────────────────────────────
cells.append(code(
    "from datasets import load_dataset\n"
    "\n"
    "dataset = load_dataset(\"fancyzhx/ag_news\")"
))

# ── 4. Print dataset ──────────────────────────────────────────────────────────
cells.append(code("print(dataset)"))

# ── 5. Explore ────────────────────────────────────────────────────────────────
cells.append(code(
    "print(dataset[\"train\"][0])\n"
    "print(dataset[\"train\"].features)\n"
    "print(dataset[\"train\"].features[\"label\"])"
))

# ── 6. Split ──────────────────────────────────────────────────────────────────
cells.append(code(
    "train_data = dataset[\"train\"].shuffle(seed=42).select(range(20000))\n"
    "test_data  = dataset[\"test\"]\n"
    "print(len(train_data))\n"
    "print(len(test_data))"
))

# ── 7. Tokenizer ──────────────────────────────────────────────────────────────
cells.append(code(
    "import re\n"
    "\n"
    "def tokenize(text):\n"
    "    return re.findall(r\"[a-z0-9]+\", text.lower())"
))

# ── 8. Build vocab ────────────────────────────────────────────────────────────
cells.append(code(
    "from collections import Counter\n"
    "counter = Counter()\n"
    "for example in train_data:\n"
    "    counter.update(tokenize(example[\"text\"]))\n"
    "print(\"unique tokens before filtering:\", len(counter))\n"
    "\n"
    "PAD_TOKEN, UNK_TOKEN = \"<pad>\", \"<unk>\"\n"
    "min_freq = 2\n"
    "vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}\n"
    "for word, freq in counter.items():\n"
    "    if freq >= min_freq:\n"
    "        vocab[word] = len(vocab)\n"
    "print(\"unique tokens after filtering:\", len(vocab))"
))

# ── 9. Encode function ────────────────────────────────────────────────────────
cells.append(code(
    "def encode(text, vocab, max_len=256):\n"
    "    tokens = tokenize(text)\n"
    "    ids = [vocab.get(t, vocab[UNK_TOKEN]) for t in tokens]\n"
    "    if len(ids) > max_len:\n"
    "        ids = ids[:max_len]\n"
    "    elif len(ids) < max_len:\n"
    "        ids += [vocab[PAD_TOKEN]] * (max_len - len(ids))\n"
    "    return ids\n"
    "\n"
    "sample  = train_data[0][\"text\"]\n"
    "encoded = encode(sample, vocab)\n"
    "print(sample)\n"
    "print(encoded[:20])\n"
    "assert len(encoded) == 256"
))

# ── 10. Save vocab ────────────────────────────────────────────────────────────
cells.append(code(
    "import json\n"
    "with open(\"vocab.json\", \"w\") as f:\n"
    "    json.dump(vocab, f)"
))

# ── 11. Dataset class ─────────────────────────────────────────────────────────
cells.append(code(
    "import torch\n"
    "from torch.utils.data import Dataset\n"
    "\n"
    "class AGNewsDataset(Dataset):\n"
    "    def __init__(self, hf_split, vocab, max_len=256):\n"
    "        self.data    = hf_split\n"
    "        self.vocab   = vocab\n"
    "        self.max_len = max_len\n"
    "\n"
    "    def __len__(self):\n"
    "        return len(self.data)\n"
    "\n"
    "    def __getitem__(self, idx):\n"
    "        example = self.data[idx]\n"
    "        ids     = encode(example[\"text\"], self.vocab, self.max_len)\n"
    "        label   = example[\"label\"]\n"
    "        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)"
))

# ── 12. DataLoaders ───────────────────────────────────────────────────────────
cells.append(code(
    "from torch.utils.data import DataLoader\n"
    "\n"
    "train_dataset = AGNewsDataset(train_data, vocab, max_len=256)\n"
    "test_dataset  = AGNewsDataset(test_data,  vocab, max_len=256)\n"
    "\n"
    "train_loader = DataLoader(\n"
    "    train_dataset,\n"
    "    batch_size=64,\n"
    "    shuffle=True\n"
    ")\n"
    "\n"
    "test_loader = DataLoader(\n"
    "    test_dataset,\n"
    "    batch_size=64,\n"
    "    shuffle=False\n"
    ")"
))

# ── 13. Batch shapes ──────────────────────────────────────────────────────────
cells.append(code(
    "x, y = next(iter(train_loader))\n"
    "print(x.shape)\n"
    "print(y.shape)"
))

# ── 14. Inspect first sample ──────────────────────────────────────────────────
cells.append(code("train_dataset[0]"))

# ── SECTION HEADER ─────────────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## Part 1 — Multi-Head Attention Transformer\n"
    "*(Make sure to upload `model.py` and `model_swa.py` to the Colab environment first)*"
))

# ── 15. Import from model.py ───────────────────────────────────────────
cells.append(code(
    "import torch.nn as nn\n"
    "from model import (\n"
    "    InputEmbedding,\n"
    "    PositionalEncoding,\n"
    "    EncoderBlock,\n"
    "    MultiHeadAttention,\n"
    "    FeedForwardBlock,\n"
    "    Encoder,\n"
    "    ClassificationHead\n"
    ")\n"
    "\n"
    "class MHATransformerClassifier(nn.Module):\n"
    "\n"
    "    def __init__(self, vocab_size, d_model=128, h=8, d_ff=512,\n"
    "                 num_layers=2, seq_len=256, num_classes=4, dropout=0.1):\n"
    "        super().__init__()\n"
    "        self.embedding  = InputEmbedding(d_model, vocab_size)\n"
    "        self.pos_enc    = PositionalEncoding(d_model, seq_len, dropout)\n"
    "        layers = [\n"
    "            EncoderBlock(\n"
    "                d_model,\n"
    "                MultiHeadAttention(d_model, h, dropout),\n"
    "                FeedForwardBlock(d_model, d_ff, dropout),\n"
    "                dropout\n"
    "            )\n"
    "            for _ in range(num_layers)\n"
    "        ]\n"
    "        self.encoder    = Encoder(d_model, layers)\n"
    "        self.classifier = ClassificationHead(d_model, num_classes)\n"
    "\n"
    "    def forward(self, x, mask=None):\n"
    "        x = self.embedding(x)\n"
    "        x = self.pos_enc(x)\n"
    "        x = self.encoder(x, mask)\n"
    "        return self.classifier(x)\n"
    "\n"
    "print(\"MHATransformerClassifier defined from model.py!\")"
))

# ── 16. Build + count MHA params ─────────────────────────────────────────────
cells.append(code(
    "device     = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    "vocab_size = len(vocab)\n"
    "\n"
    "mha_model = MHATransformerClassifier(\n"
    "    vocab_size = vocab_size,\n"
    "    d_model    = 128,\n"
    "    h          = 8,\n"
    "    d_ff       = 512,\n"
    "    num_layers = 2,\n"
    "    seq_len    = 256,\n"
    "    num_classes = 4,\n"
    "    dropout    = 0.1\n"
    ")\n"
    "\n"
    "mha_params = sum(p.numel() for p in mha_model.parameters())\n"
    "print(f\"MHA  |  parameters: {mha_params:,}\")"
))

# ── 17. Train + evaluate helpers ──────────────────────────────────────────────
cells.append(code(
    "import time\n"
    "\n"
    "def evaluate(model, loader, device):\n"
    "    model.eval()\n"
    "    correct = 0\n"
    "    total   = 0\n"
    "    with torch.no_grad():\n"
    "        for x, y in loader:\n"
    "            x, y    = x.to(device), y.to(device)\n"
    "            logits  = model(x)\n"
    "            correct += (logits.argmax(dim=-1) == y).sum().item()\n"
    "            total   += y.size(0)\n"
    "    return correct / total\n"
    "\n"
    "\n"
    "def train_model(model, train_loader, test_loader, epochs=5, lr=1e-3, device='cuda'):\n"
    "    model     = model.to(device)\n"
    "    optimizer = torch.optim.Adam(model.parameters(), lr=lr)\n"
    "    criterion = nn.CrossEntropyLoss()\n"
    "\n"
    "    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'epoch_time': []}\n"
    "\n"
    "    for epoch in range(epochs):\n"
    "        model.train()\n"
    "        total_loss = 0\n"
    "        correct    = 0\n"
    "        total      = 0\n"
    "        start      = time.time()\n"
    "\n"
    "        for x, y in train_loader:\n"
    "            x, y   = x.to(device), y.to(device)\n"
    "            optimizer.zero_grad()\n"
    "            logits = model(x)\n"
    "            loss   = criterion(logits, y)\n"
    "            loss.backward()\n"
    "            optimizer.step()\n"
    "\n"
    "            total_loss += loss.item() * y.size(0)\n"
    "            correct    += (logits.argmax(dim=-1) == y).sum().item()\n"
    "            total      += y.size(0)\n"
    "\n"
    "        epoch_time = time.time() - start\n"
    "        train_loss = total_loss / total\n"
    "        train_acc  = correct / total\n"
    "        test_acc   = evaluate(model, test_loader, device)\n"
    "\n"
    "        history['train_loss'].append(train_loss)\n"
    "        history['train_acc'].append(train_acc)\n"
    "        history['test_acc'].append(test_acc)\n"
    "        history['epoch_time'].append(epoch_time)\n"
    "\n"
    "        print(\n"
    "            f\"Epoch {epoch+1}/{epochs}  \"\n"
    "            f\"| Loss: {train_loss:.4f}  \"\n"
    "            f\"| Train Acc: {train_acc:.4f}  \"\n"
    "            f\"| Test Acc: {test_acc:.4f}  \"\n"
    "            f\"| Time: {epoch_time:.1f}s\"\n"
    "        )\n"
    "\n"
    "    return history"
))

# ── 18. Train MHA ─────────────────────────────────────────────────────────────
cells.append(code(
    "print(\"--- Training MHA Transformer ---\\n\")\n"
    "mha_history = train_model(\n"
    "    mha_model, train_loader, test_loader, epochs=5, device=device\n"
    ")"
))

# ── SECTION HEADER ─────────────────────────────────────────────────────────
cells.append(md("---\n## Part 2 — Sliding Window Attention Transformer"))

# ── 19. Import SWA + classifier ───────────────────────────────────────────
cells.append(code(
    "from model_swa import SlidingWindowAttention\n"
    "# Re-using components from model.py, but substituting SlidingWindowAttention\n"
    "\n"
    "class SWATransformerClassifier(nn.Module):\n"
    "\n"
    "    def __init__(self, vocab_size, d_model=128, h=8, d_ff=512,\n"
    "                 num_layers=2, seq_len=256, num_classes=4, dropout=0.1, window_size=32):\n"
    "        super().__init__()\n"
    "        self.embedding  = InputEmbedding(d_model, vocab_size)\n"
    "        self.pos_enc    = PositionalEncoding(d_model, seq_len, dropout)\n"
    "        layers = [\n"
    "            EncoderBlock(\n"
    "                d_model,\n"
    "                SlidingWindowAttention(d_model, h, window_size, dropout),\n"
    "                FeedForwardBlock(d_model, d_ff, dropout),\n"
    "                dropout\n"
    "            )\n"
    "            for _ in range(num_layers)\n"
    "        ]\n"
    "        self.encoder    = Encoder(d_model, layers)\n"
    "        self.classifier = ClassificationHead(d_model, num_classes)\n"
    "\n"
    "    def forward(self, x, mask=None):\n"
    "        x = self.embedding(x)\n"
    "        x = self.pos_enc(x)\n"
    "        x = self.encoder(x, mask)\n"
    "        return self.classifier(x)\n"
    "\n"
    "print(\"SWATransformerClassifier defined using model_swa.py!\")"
))

# ── 20. Build + count SWA params ─────────────────────────────────────────────
cells.append(code(
    "swa_model = SWATransformerClassifier(\n"
    "    vocab_size  = vocab_size,\n"
    "    d_model     = 128,\n"
    "    h           = 8,\n"
    "    d_ff        = 512,\n"
    "    num_layers  = 2,\n"
    "    seq_len     = 256,\n"
    "    num_classes = 4,\n"
    "    dropout     = 0.1,\n"
    "    window_size = 32    # each token sees 65 neighbours (w=32 each side)\n"
    ")\n"
    "\n"
    "swa_params = sum(p.numel() for p in swa_model.parameters())\n"
    "print(f\"SWA  |  parameters: {swa_params:,}\")"
))

# ── 21. Train SWA ─────────────────────────────────────────────────────────────
cells.append(code(
    "print(\"--- Training SWA Transformer ---\\n\")\n"
    "swa_history = train_model(\n"
    "    swa_model, train_loader, test_loader, epochs=5, device=device\n"
    ")"
))

# ── SECTION HEADER ─────────────────────────────────────────────────────────
cells.append(md("---\n## Comparison — MHA vs SWA"))

# ── 22. Results table ─────────────────────────────────────────────────────────
cells.append(code(
    "EPOCHS = 5\n"
    "\n"
    "print(\"=\" * 58)\n"
    "print(f\"{'Metric':<28} {'MHA':>13} {'SWA':>13}\")\n"
    "print(\"=\" * 58)\n"
    "print(f\"{'Parameters':<28} {mha_params:>13,} {swa_params:>13,}\")\n"
    "print(f\"{'Final Test Accuracy':<28} {mha_history['test_acc'][-1]:>13.4f} {swa_history['test_acc'][-1]:>13.4f}\")\n"
    "print(f\"{'Final Train Loss':<28} {mha_history['train_loss'][-1]:>13.4f} {swa_history['train_loss'][-1]:>13.4f}\")\n"
    "print(f\"{'Avg Epoch Time (s)':<28} {sum(mha_history['epoch_time'])/EPOCHS:>13.1f} {sum(swa_history['epoch_time'])/EPOCHS:>13.1f}\")\n"
    "print(f\"{'Attention window':<28} {'global (n)':>13} {'local (65)':>13}\")\n"
    "print(\"=\" * 58)"
))

# ── 23. Loss + accuracy curves ────────────────────────────────────────────────
cells.append(code(
    "import matplotlib.pyplot as plt\n"
    "\n"
    "epochs_range = range(1, EPOCHS + 1)\n"
    "\n"
    "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
    "\n"
    "axes[0].plot(epochs_range, mha_history['train_loss'], marker='o', label='MHA')\n"
    "axes[0].plot(epochs_range, swa_history['train_loss'], marker='s', label='SWA')\n"
    "axes[0].set_title('Training Loss')\n"
    "axes[0].set_xlabel('Epoch')\n"
    "axes[0].legend()\n"
    "axes[0].grid(True)\n"
    "\n"
    "axes[1].plot(epochs_range, mha_history['test_acc'], marker='o', label='MHA')\n"
    "axes[1].plot(epochs_range, swa_history['test_acc'], marker='s', label='SWA')\n"
    "axes[1].set_title('Test Accuracy')\n"
    "axes[1].set_xlabel('Epoch')\n"
    "axes[1].legend()\n"
    "axes[1].grid(True)\n"
    "\n"
    "axes[2].bar(\n"
    "    ['MHA', 'SWA'],\n"
    "    [sum(mha_history['epoch_time'])/EPOCHS, sum(swa_history['epoch_time'])/EPOCHS],\n"
    "    color=['steelblue', 'coral']\n"
    ")\n"
    "axes[2].set_title('Avg Epoch Time (s)')\n"
    "axes[2].set_ylabel('Seconds')\n"
    "\n"
    "plt.suptitle('MHA vs SWA — AG News (20k train / 7.6k test)', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.savefig('mha_vs_swa.png', dpi=150)\n"
    "plt.show()"
))

# ── assemble notebook ─────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        },
        "accelerator": "GPU"
    },
    "cells": cells
}

out_path = "/Users/daniyal/Desktop/Transformers from Scratch/ag_news_experiment.ipynb"
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written  ->  {out_path}")
print(f"Total cells: {len(cells)}")
