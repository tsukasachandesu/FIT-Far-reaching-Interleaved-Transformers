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

class CausalPrefixAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.max_heads_process = 2

        inner_dim = heads * dim_head

        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.cross_attn_dropout = dropout
        
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context, context_mask = None, rotary_pos_emb = None):
        batch, context_len, device = x.shape[0], context.shape[-2], x.device

        q_rotary_pos_emb = rotary_pos_emb
        k_rotary_pos_emb = rotary_pos_emb

        # take care of cross attention dropout

        if self.training and self.cross_attn_dropout > 0.:
            rand = torch.zeros((batch, context_len), device = device).uniform_()
            keep_context_len = context_len - int(context_len * self.cross_attn_dropout)
            keep_indices = rand.topk(keep_context_len, dim = -1).indices
            keep_mask = torch.zeros_like(rand).scatter_(1, keep_indices, 1).bool()

            context = rearrange(context[keep_mask], '(b n) d -> b n d', b = batch)

            if exists(context_mask):
                context_mask = rearrange(context_mask[keep_mask], '(b n) -> b n', b = batch)

            # operate on rotary position embeddings for keys

            k_rotary_pos_emb = repeat(k_rotary_pos_emb, '... -> b ...', b = batch)
            k_rotary_pos_emb_context, k_rotary_pos_emb_seq = k_rotary_pos_emb[:, :context_len], k_rotary_pos_emb[:, context_len:]
            k_rotary_pos_emb_context = rearrange(k_rotary_pos_emb_context[keep_mask], '(b n) d -> b n d', b = batch)

            k_rotary_pos_emb = torch.cat((k_rotary_pos_emb_context, k_rotary_pos_emb_seq), dim = 1)
            k_rotary_pos_emb = rearrange(k_rotary_pos_emb, 'b n d -> b 1 n d')

        # normalization

        x = self.norm(x)
        context = self.context_norm(context)

        # derive queries, keys, values

        q = self.to_q(x)

        k_input, v_input = self.to_kv(x).chunk(2, dim = -1)
        k_context, v_context = self.to_kv(context).chunk(2, dim = -1)

        k = torch.cat((k_context, k_input), dim = 1)
        v = torch.cat((v_context, v_input), dim = 1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        q = q * self.scale

        # rotate queries and keys with rotary embeddings

        if exists(rotary_pos_emb):
            q = apply_rotary_pos_emb(q_rotary_pos_emb, q)
            k = apply_rotary_pos_emb(k_rotary_pos_emb, k)

        # take care of masking

        i, j = q.shape[-2], k.shape[-2]
        mask_value = -torch.finfo(q.dtype).max

        if exists(context_mask):
            mask_len = context_mask.shape[-1]
            context_mask = F.pad(context_mask, (0, max(j - mask_len, 0)), value = True)
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')

        causal_mask = torch.ones((i, j), device = x.device, dtype = torch.bool).triu(j - i + 1)

        # process in chunks of heads

        out = []

        max_heads = self.max_heads_process

        for q_chunk, k_chunk, v_chunk in zip(q.split(max_heads, dim = 1), k.split(max_heads, dim = 1), v.split(max_heads, dim = 1)):
            sim = einsum('b h i d, b h j d -> b h i j', q_chunk, k_chunk)

            if exists(context_mask):
                sim = sim.masked_fill(~context_mask, mask_value)

            sim = sim.masked_fill(causal_mask, mask_value)

            attn = sim.softmax(dim = -1)
            attn = self.dropout(attn)

            out_chunk = einsum('b h i j, b h j d -> b h i d', attn, v_chunk)
            out.append(out_chunk)

        # concat all the heads together

        out = torch.cat(out, dim = 1)

        # merge heads and then combine with linear

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

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
        
        rotary_pos_emb = self.rotary_pos_emb(self.max_seq_len, device = x.device)
        latents = self.pos_emb(torch.arange(self.max_seq_len, device = x.device))
        latents = latents.repeat(x.shape[0], 1, 1)
        letents = latents.reshape(-1,1,512)
      
        for cross_attn, ff in self.perceive_layers:
          latents = cross_attn(latents, x, rotary_pos_emb = rotary_pos_emb) + latents
          latents = ff(latents) + latents
        return latents
