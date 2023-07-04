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
      
        pos = self.pos_emb(torch.arange(seq_len, device = device))
      
        x = x + pos
    
        latents = pos.reshape(-1, 16, emb_size)
      
        x = x.reshape(-1, 16, emb_size)

        for attn, ff in self.layers:
            x = attn(x, rotary_pos_emb = rotary_pos_emb) + x
            x = ff(x) + x

        for cross_attn, ff in self.perceive_layers:
            latents = cross_attn(latents, x, context_mask = prefix_mask, rotary_pos_emb = rotary_pos_emb) + latents
            latents = ff(latents) + latents

        latents = latents.reshape(batch_size, -1, emb_size)
        latents_leading, latents_last = latents[:, :-1,:], latents[:, -1:,:]
        latents = torch.cat([torch.zeros_like(latents_last), latents_leading], dim=1)

        for attn, ff in self.layers:
            latents = attn(latents, rotary_pos_emb = rotary_pos_emb) + latents
            latents = ff(latents) + latents
          
        latents = latents.reshape(-1, 16, emb_size)
      
        for cross_attn, ff in self.perceive_layers:
            x = cross_attn(x, latents, context_mask = prefix_mask, rotary_pos_emb = rotary_pos_emb) + x
            x = ff(x) + x

        latents = latents.reshape(batch_size, -1, emb_size)
        latents = torch.cat([latents[:, 1:,:], latents_last], dim=1)

        x = x.reshape(batch_size, -1, emb_size)



