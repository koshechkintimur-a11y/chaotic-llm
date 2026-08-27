"""
chaotic_torch.py — Torch model definitions for the Chaotic LLM project (v2).

Models take (B, N, d_in) input and produce (B, d_out) logits.
d_out = number of classes (cross-entropy).
"""
import torch
import torch.nn as nn

from chaos_lib import permute_indices, square_grid_size


# ============ ChaoticMixer ============

class ChaoticBlock(nn.Module):
    def __init__(self, d, n_tokens, coupling='sym', sigma_cache=None):
        super().__init__()
        self.n_tokens = n_tokens
        self.N = square_grid_size(n_tokens)
        self.coupling = coupling
        self.gate = nn.Parameter(torch.tensor(0.5))
        self._sigma_cache = sigma_cache

    def forward(self, x, step, gsf_control=None):
        B, N, d = x.shape
        device = x.device
        if self._sigma_cache is not None and step in self._sigma_cache:
            sigma_t = self._sigma_cache[step]
        else:
            sigma = permute_indices(N, step)
            sigma_t = torch.as_tensor(sigma, dtype=torch.long, device=device)
        x = x[:, sigma_t, :]

        g = torch.sigmoid(self.gate)
        # handle odd N: leave the last token unchanged (coupling needs pairs)
        if N % 2 == 1:
            tail = x[:, -1:, :]
            x = x[:, :-1, :]
            if gsf_control is not None:
                gsf_control = gsf_control[:, :-1, :]
            B, Ne, d = x.shape
        else:
            tail = None
            Ne = N
        even = x[:, 0::2, :]
        odd = x[:, 1::2, :]
        if gsf_control is not None:
            g_even = gsf_control[:, 0::2, :]
            g_odd = gsf_control[:, 1::2, :]
        else:
            g_even = g_odd = torch.ones_like(even[:, :, :1]) * g
        x_even_new = even + g_even * odd
        x_odd_new = odd + g_odd * even
        x = torch.stack([x_even_new, x_odd_new], dim=2).reshape(B, Ne, d)
        if tail is not None:
            x = torch.cat([x, tail], dim=1)
        return x


class ChaoticMixer(nn.Module):
    def __init__(self, n_tokens, d_in, d_out, d_model=32, n_blocks=6,
                 gsf_hidden=None, adaptive_depth=False):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.adaptive_depth = adaptive_depth

        self.embed = nn.Linear(d_in, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        # precompute all permutation tensors once (they depend only on N, step)
        self._sigmas = {}
        for step in range(1, n_blocks + 1):
            sigma = permute_indices(n_tokens, step)
            self._sigmas[step] = torch.as_tensor(sigma, dtype=torch.long)
        self.blocks = nn.ModuleList([
            ChaoticBlock(d_model, n_tokens, sigma_cache=self._sigmas)
            for _ in range(n_blocks)
        ])

        if gsf_hidden is not None:
            self.gsf = nn.Sequential(
                nn.Linear(d_model, gsf_hidden), nn.ReLU(),
                nn.Linear(gsf_hidden, 1), nn.Sigmoid(),
            )
        else:
            self.gsf = None

        self.readout = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(),
            nn.Linear(32, d_out),
        )

    def forward(self, x, query_mask=None, return_states=False):
        B, N, _ = x.shape
        x = self.embed(x) + self.pos_embed
        states = [x]
        for step, block in enumerate(self.blocks):
            gsf_control = None
            if self.gsf is not None:
                pooled = x.mean(dim=1, keepdim=True)
                gsf_control = self.gsf(x + pooled)
            x = block(x, step + 1, gsf_control=gsf_control)
            if return_states:
                states.append(x)
        if query_mask is not None:
            m = query_mask.float().unsqueeze(-1)
            out = (x * m).sum(dim=1) / (m.sum(dim=1) + 1e-8)
        else:
            out = x.mean(dim=1)
        logits = self.readout(out)
        return (logits, states) if return_states else logits


class ChaoticAttnReadout(nn.Module):
    """ChaoticMixer + content-based readout AT THE QUERY ONLY.

    The chaotic blocks spread information (O(N log N)); then the query token
    selects the relevant content via attention over ALL final token states:
        out = softmax(q^T k_i) v_i
    This is O(N) per query (not O(N^2) — only ONE query attends), so the
    total remains O(N log N + N) = O(N log N).
    """
    def __init__(self, n_tokens, d_in, d_out, d_model=32, n_blocks=6):
        super().__init__()
        self.mixer = ChaoticMixer(n_tokens, d_in, d_out, d_model=d_model,
                                  n_blocks=n_blocks)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.readout = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(),
                                     nn.Linear(32, d_out))

    def forward(self, x, query_mask=None):
        B, N, _ = x.shape
        h = self.mixer.embed(x) + self.mixer.pos_embed
        for step, block in enumerate(self.mixer.blocks):
            h = block(h, step + 1)
        # query state
        m = query_mask.float().unsqueeze(-1)
        q = (h * m).sum(dim=1, keepdim=True) / (m.sum(dim=1, keepdim=True) + 1e-8)
        # content-based selection over all tokens (O(N) per query)
        keys = self.k_proj(h)          # (B, N, d)
        vals = self.v_proj(h)          # (B, N, d)
        scores = (q * keys).sum(-1, keepdim=True) / (self.mixer.d_model ** 0.5)
        attn = scores.softmax(dim=1)   # (B, N, 1)
        out = (attn * vals).sum(dim=1)
        return self.readout(out)


# ============ Baselines ============

class GRUModel(nn.Module):
    def __init__(self, n_tokens, d_in, d_out, d_model=32):
        super().__init__()
        self.embed = nn.Linear(d_in, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.readout = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, d_out))

    def forward(self, x, query_mask=None):
        B, N, _ = x.shape
        x = self.embed(x)
        out, _ = self.gru(x)
        if query_mask is not None:
            m = query_mask.float().unsqueeze(-1)
            out = (out * m).sum(dim=1) / (m.sum(dim=1) + 1e-8)
        else:
            out = out[:, -1, :]
        return self.readout(out)


class LocalAttn(nn.Module):
    def __init__(self, n_tokens, d_in, d_out, d_model=32, window=5):
        super().__init__()
        self.n_tokens = n_tokens
        self.window = window
        self.embed = nn.Linear(d_in, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(),
                                 nn.Linear(d_model * 4, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.readout = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, d_out))

    def forward(self, x, query_mask=None):
        B, N, _ = x.shape
        x = self.embed(x) + self.pos_embed
        mask = torch.zeros(N, N, device=x.device)
        for i in range(N):
            lo = max(0, i - self.window)
            hi = min(N, i + self.window + 1)
            mask[i, lo:hi] = 1.0
        qkv = self.qkv(x).reshape(B, N, 3, -1).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.embed.out_features ** 0.5)
        attn = attn.masked_fill(mask.unsqueeze(0) == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        x = x + self.proj(attn @ v)
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        if query_mask is not None:
            m = query_mask.float().unsqueeze(-1)
            out = (x * m).sum(dim=1) / (m.sum(dim=1) + 1e-8)
        else:
            out = x.mean(dim=1)
        return self.readout(out)


class FullAttn(nn.Module):
    def __init__(self, n_tokens, d_in, d_out, d_model=32, n_layers=2):
        super().__init__()
        self.embed = nn.Linear(d_in, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=d_model * 4,
                                       batch_first=True, dropout=0.0)
            for _ in range(n_layers)
        ])
        self.readout = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, d_out))

    def forward(self, x, query_mask=None):
        B, N, _ = x.shape
        x = self.embed(x) + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        if query_mask is not None:
            m = query_mask.float().unsqueeze(-1)
            out = (x * m).sum(dim=1) / (m.sum(dim=1) + 1e-8)
        else:
            out = x.mean(dim=1)
        return self.readout(out)


class MLPModel(nn.Module):
    def __init__(self, n_tokens, d_in, d_out, d_model=32):
        super().__init__()
        self.embed = nn.Linear(d_in, d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, d_out),
        )

    def forward(self, x, query_mask=None):
        B, N, _ = x.shape
        x = self.embed(x)
        if query_mask is not None:
            m = query_mask.float().unsqueeze(-1)
            x = (x * m).sum(dim=1) / (m.sum(dim=1) + 1e-8)
        else:
            x = x.mean(dim=1)
        return self.net(x)