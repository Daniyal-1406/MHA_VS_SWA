import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)

        # eps prevents division by zero
        return self.alpha * (x - mean) / torch.sqrt(variance + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(
            self.dropout(torch.relu(self.linear_1(x)))
        )


class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)

        position = torch.arange(
            0, seq_len, dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class SlidingWindowAttention(nn.Module):
    """
    Multi-Head Sliding Window Attention — true O(N·w·d) implementation.

    For each query at position i, we physically gather only the 2w+1
    neighbouring keys/values using torch.gather.  The N×N score matrix
    is NEVER created, so both compute and memory scale as O(N·w).
    """

    def __init__(self, d_model: int, h: int, window_size: int, dropout: float) -> None:
        super().__init__()

        self.d_model = d_model
        self.h = h
        self.w = window_size          # one-sided radius; full window = 2w + 1

        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _gather_window(x, indices):
        # x       : (B, h, N, d_k)
        # indices : (N, win)  — positions of the 2w+1 neighbours for each token
        B, h, N, d_k = x.shape
        win = indices.shape[1]

        # Flatten indices to 1-D, do one advanced index into the N axis,
        # then reshape — NO N×N tensor is ever allocated.
        #
        # flat_idx : (N*win,)
        # x[:, :, flat_idx, :] reads exactly N*win rows  →  O(N·w·d) memory
        flat_idx = indices.reshape(-1)                      # (N*win,)
        gathered = x[:, :, flat_idx, :]                    # (B, h, N*win, d_k)
        return gathered.view(B, h, N, win, d_k)            # (B, h, N, win, d_k)

    def forward(self, q, k, v, mask):

        query = self.w_q(q)
        key   = self.w_k(k)
        value = self.w_v(v)

        # (batch, seq_len, d_model)
        # -> (batch, seq_len, h, d_k)
        # -> (batch, h, seq_len, d_k)

        B, T, _ = query.shape

        query = query.view(B, T, self.h, self.d_k).transpose(1, 2)
        key   = key.view(B, T, self.h, self.d_k).transpose(1, 2)
        value = value.view(B, T, self.h, self.d_k).transpose(1, 2)

        # ── build gather indices: for position i, window = [i-w .. i+w] ──────
        # shape: (T, 2w+1) — clamped so edges don't go out of bounds
        center  = torch.arange(T, device=query.device)                    # (T,)
        offsets = torch.arange(-self.w, self.w + 1, device=query.device)  # (2w+1,)
        indices = (center.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, T - 1)
        # indices[i, k] = position of the k-th neighbour of token i

        # ── gather only the window keys and values — O(N·w·d) ────────────────
        # gathered shape: (B, h, T, 2w+1, d_k)   NO N×N matrix ever exists
        gathered_k = self._gather_window(key,   indices)
        gathered_v = self._gather_window(value, indices)

        # ── local dot-product: each query i vs its 2w+1 neighbours only ──────
        # query      : (B, h, T, d_k)       -> (B, h, T, 1,    d_k)
        # gathered_k : (B, h, T, 2w+1, d_k) -> (B, h, T, d_k, 2w+1)
        # scores     : (B, h, T, 1, 2w+1)   -> (B, h, T, 2w+1)
        scores = (query.unsqueeze(3) @ gathered_k.transpose(-2, -1)).squeeze(3)
        scores = scores / math.sqrt(self.d_k)

        # softmax over 2w+1 — NOT over N
        scores = scores.softmax(dim=-1)
        self.attention_scores = scores

        if self.dropout is not None:
            scores = self.dropout(scores)

        # ── weighted sum over window values ───────────────────────────────────
        # scores     : (B, h, T, 2w+1)      -> (B, h, T, 1,    2w+1)
        # gathered_v : (B, h, T, 2w+1, d_k)
        # output     : (B, h, T, 1, d_k)    -> (B, h, T, d_k)
        x = (scores.unsqueeze(3) @ gathered_v).squeeze(3)

        # (batch, h, seq_len, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(B, T, self.h * self.d_k)

        return self.w_o(x)


class EncoderBlock(nn.Module):
    def __init__(self, features, self_attention_block, feed_forward_block, dropout):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_connections = nn.ModuleList([
            ResidualConnection(features, dropout),
            ResidualConnection(features, dropout)
        ])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x,
            lambda x: self.self_attention_block(x, x, x, src_mask)
        )

        x = self.residual_connections[1](
            x,
            self.feed_forward_block
        )

        return x


class Encoder(nn.Module):
    def __init__(self, features, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


# d_model = 128
# d_ff    = 512
# h       = 8
# w       = 32   # window radius; each token sees 65 neighbours
# dropout = 0.1
# seq_len = 256

# attention     = SlidingWindowAttention(d_model, h, w, dropout)
# feed_forward  = FeedForwardBlock(d_model, d_ff, dropout)

# encoder_block = EncoderBlock(
#     d_model,
#     attention,
#     feed_forward,
#     dropout
# )

# encoder = Encoder(
#     d_model,
#     [encoder_block]
# )

# x = torch.randn(4, seq_len, d_model)

# output = encoder(x, None)

# print("Input:", x.shape)
# print("Output:", output.shape)


class ClassificationHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.mean(dim=1)
        return self.linear(x)


classifier = ClassificationHead(128, 4)

x = torch.randn(64, 256, 128)

output = classifier(x)

print(x.shape)
print(output.shape)
