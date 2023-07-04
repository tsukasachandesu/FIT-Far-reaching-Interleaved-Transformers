import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat

# helper functions

def exists(val):
    return val is not None

# feedforward

def FeedForward(dim, mult = 4, dropout = 0.):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias = False),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim, bias = False)
    )

# rotary positional embedding
# https://arxiv.org/abs/2104.09864

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device = device, dtype = self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim = -1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(pos, t):
    seq_len, rotate_dim = t.shape[-2], pos.shape[-1]
    pos = pos[..., -seq_len:, :]
    t, t_pass = t[..., :rotate_dim], t[..., rotate_dim:]
    t = (t * pos.cos()) + (rotate_half(t) * pos.sin())
    return torch.cat((t, t_pass), dim = -1)

# attention


class PerceiverAR(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        cross_attn_seq_len,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        cross_attn_dropout = 0.,
        ff_mult = 4,
        perceive_depth = 1,
        perceive_max_heads_process = 2 # processes the heads in the perceiver layer in chunks to lower peak memory, in the case the prefix is really long
    ):
        super().__init__()
        assert max_seq_len > cross_attn_seq_len, 'max_seq_len must be greater than cross_attn_seq_len, the length of the sequence for which to cross attend to "perceiver" style'
        self.max_seq_len = max_seq_len
        self.cross_attn_seq_len = cross_attn_seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.rotary_pos_emb = RotaryEmbedding(dim = max(32, dim_head // 2))

        self.perceive_layers  = nn.ModuleList([])

        for _ in range(perceive_depth):
            self.perceive_layers.append(nn.ModuleList([
                CausalPrefixAttention(dim = dim, dim_head = dim_head, heads = heads, max_heads_process = perceive_max_heads_process, dropout = dropout, cross_attn_dropout = cross_attn_dropout),
                FeedForward(dim, mult = ff_mult, dropout = dropout)
            ]))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CausalAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim, mult = ff_mult, dropout = dropout),
            ]))

        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

    def forward(
        self,
        x,
        prefix_mask = None,
        labels = None
    ):
        batch_size, seq_len, emb_size, device = x.shape, x.device
        
        x = x + self.pos_emb(torch.arange(seq_len, device = device))
        
        rotary_pos_emb1 = self.rotary_pos_emb(16, device = device)
        rotary_pos_emb2 = self.rotary_pos_emb(seq_len/4, device = device)

        latents = self.pos_emb(torch.arange(seq_len / 4, device = device)).unsqueeze(0)
        latents = latents.repeat(batch_size, 1, 1)
        latents = latents.reshape(-1, 4, emb_size)
        x = x.reshape(-1, 16, emb_size)

        for i in range(len(self.layer_configs)):
            for attn, ff in self.layers:
                x = attn(x, rotary_pos_emb = rotary_pos_emb1) + x
                x = ff(x) + x
                
            for cross_attn, ff in self.perceive_layers:
                latents = cross_attn(latents, x, context_mask = prefix_mask, rotary_pos_emb = rotary_pos_emb1) + latents
                latents = ff(latents) + latents
                
            latents = latents.reshape(batch_size, -1, emb_size)
            
            for attn, ff in self.layers:
                latents = attn(latents, rotary_pos_emb = rotary_pos_emb2) + latents
                latents = ff(latents) + latents
                
            latents_leading, latents_last = latents[:, :-1,:], latents[:, -1:,:]
            latents = torch.cat([torch.zeros_like(latents_last), latents_leading], dim=1)
            latents = latents.reshape(-1, 4, emb_size)
            
            for cross_attn, ff in self.perceive_layers:
                x = cross_attn(x, latents, context_mask = prefix_mask, rotary_pos_emb = rotary_pos_emb1) + x
                x = ff(x) + x

            for attn, ff in self.layers:
                x = attn(x, rotary_pos_emb = rotary_pos_emb1) + x
                x = ff(x) + x

            latents = latents.reshape(batch_size, -1, emb_size)
            latents = torch.cat([latents[:, 1:,:], latents_last], dim=1)
            
        x = x.reshape(batch_size, -1, emb_size)


    self.x_mask = 1. - get_ar_mask(x_per_group, dtype=tf.float32)
    self.l_mask = 1. - get_chunk_ar_mask(
        num_groups * latents_per_group, latents_per_group, dtype=tf.float32)
    todo マスクはまだ未実装

A Domain-Knowledge-Inspired Music Embedding Space and a Novel Attention
Mechanism for Symbolic Music Modeling

Controlling Perceived Emotion in Symbolic Music Generation
with Monte Carlo Tree Search



