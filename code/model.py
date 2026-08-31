"""Decoder-only Transformer, with optional Adaptive Latent State Bottleneck.

Baseline and ALSB share every line of code except the alsb flag, so a
difference between conditions cannot come from an incidental code path.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, nh, ffn_mult=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nh, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        hid = int(ffn_mult * d)
        self.ffn = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Linear(hid, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.ffn(self.ln2(x))


class ALSB(nn.Module):
    """Gated diagonal recurrence of width d_s, read back into the residual stream.

        z_t = (1-a_t) * A z_{t-1} + a_t * phi(W_in h_t)
        h_t = h_t + W_out z_t

    Diagonal A => O(1) per-token state at inference.  Computed with a sequential
    scan here; a parallel associative scan is a throughput optimisation only and
    does not change the function computed.
    """

    def __init__(self, d, d_s):
        super().__init__()
        self.d_s = d_s
        self.ln = nn.LayerNorm(d)
        self.w_in = nn.Linear(d, d_s)
        self.w_g = nn.Linear(d + d_s, d_s)
        self.w_out = nn.Linear(d_s, d)
        # diagonal transition, parameterised so A = sigmoid(a_log) in (0,1)
        self.a_log = nn.Parameter(torch.zeros(d_s))
        nn.init.zeros_(self.w_out.weight)   # start as identity map
        nn.init.zeros_(self.w_out.bias)

    def forward(self, x):
        B, T, _ = x.shape
        h = self.ln(x)
        u = torch.tanh(self.w_in(h))
        A = torch.sigmoid(self.a_log)
        z = x.new_zeros(B, self.d_s)
        zs = []
        for t in range(T):
            g = torch.sigmoid(self.w_g(torch.cat([h[:, t], z], dim=-1)))
            z = (1 - g) * (A * z) + g * u[:, t]
            zs.append(z)
        Z = torch.stack(zs, dim=1)
        return x + self.w_out(Z), Z


class ChessLM(nn.Module):
    def __init__(self, vocab, d=256, n_layer=8, n_head=8, max_len=161,
                 alsb=False, d_s=64, k=4, ffn_mult=4.0, n_state_cls=13,
                 n_state_slots=64):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, n_head, ffn_mult) for _ in range(n_layer)])
        self.alsb = alsb
        if alsb:
            # one ALSB module after every k-th block
            self.alsb_at = {i for i in range(k - 1, n_layer, k)}
            self.alsb_mods = nn.ModuleDict(
                {str(i): ALSB(d, d_s) for i in sorted(self.alsb_at)})
            last = max(self.alsb_at)
            # auxiliary state decoder reads the LAST ALSB channel:
            # n_state_slots x n_state_cls  (chess: 64 squares x 13 pieces;
            # synthetic: 8 variables x 100 values)
            self.state_head = nn.Linear(d_s, n_state_slots * n_state_cls)
            self.last_alsb = last
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.max_len = max_len

    def forward(self, idx, return_hidden=False):
        B, T = idx.shape
        mask = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), 1)
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        hiddens, Zlast = [], None
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask)
            if self.alsb and i in self.alsb_at:
                x, Z = self.alsb_mods[str(i)](x)
                if i == self.last_alsb:
                    Zlast = Z
            if return_hidden:
                hiddens.append(x)
        logits = self.head(self.ln_f(x))
        state_logits = self.state_head(Zlast) if (self.alsb and Zlast is not None) else None
        return logits, hiddens, state_logits


def param_count(m):
    return sum(p.numel() for p in m.parameters())
