def exists(val):
    return val is not None

def FeedForward(dim, mult = 4, dropout = 0.):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias = False),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim, bias = False)
    )

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

class cross(nn.Module):
    def __init__(
        self,
        *,
        dim=512,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
        perceive_depth = 4,
        max_seq_len
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pos_emb = nn.Embedding(max_seq_len, dim)
      
        self.rotary_pos_emb = RotaryEmbedding(dim = max(32, dim_head // 2))
      
        self.perceive_layers  = nn.ModuleList([])
      
        for _ in range(perceive_depth):
            self.perceive_layers.append(nn.ModuleList([
                CausalPrefixAttention(dim = dim, dim_head = dim_head, heads = heads, dropout = dropout),
                FeedForward(dim, mult = ff_mult, dropout = dropout)
            ]))

    def forward(
        self,
        x
    ):
        
        rotary_pos_emb = self.rotary_pos_emb(16, device = x.device)

        latents = self.pos_emb(torch.arange(self.max_seq_len, device = x.device))
        latents = latents.repeat(x.shape[0], 1, 1)
      
        for cross_attn, ff in self.perceive_layers:
          latents = cross_attn(latents, x, context_mask = prefix_mask, rotary_pos_emb = rotary_pos_emb) + latents
          latents = ff(latents) + latents
        return latents
