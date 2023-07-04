class FITAR(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

  def __init__(self,
               layers,  # str format: (local-layers)global-layers.. eg '(2)4(2)'
               x_size,
               num_groups,
               latents_per_group,
               x_dim,
               latent_dim,
               x_num_heads,
               latent_num_heads,
               mlp_ratio,
               vocab_size,
               shared_embedding=True,
               output_bias=True,
               drop_path=0.0,
               drop_units=0.0,
               drop_att=0.0,
               x_pos_encoding='learned',
               latent_pos_encoding='learned',
               **kwargs):
    super().__init__(**kwargs)
    if x_size % num_groups != 0:
      raise ValueError(
          f'x_size={x_size} is not divisible by num_groups={num_groups}')
    x_per_group = x_size // num_groups
    self.num_groups = num_groups
    self.latents_per_group = latents_per_group
    self.shared_embedding = shared_embedding
    self.output_bias = output_bias
    self.layer_configs = get_layer_config(layers)
    self.x_mask = 1. - get_ar_mask(x_per_group, dtype=tf.float32)
    self.l_mask = 1. - get_chunk_ar_mask(
        num_groups * latents_per_group, latents_per_group, dtype=tf.float32)

    self.latent_pos_emb = add_vis_pos_emb(
        self, latent_pos_encoding, num_groups, latents_per_group, latent_dim,
        name_prefix=f'{self.name}/latent_pos_emb/kernel',
        return_only=True, normalization_max=1000.)
    self.x_pos_emb = add_vis_pos_emb(
        self, x_pos_encoding, x_per_group, 1, x_dim,
        name_prefix=f'{self.name}/x_pos_emb/kernel',
        return_only=True, normalization_max=1000.)
    add_vocab_token_emb(
        self, vocab_size, x_dim, shared_embedding, output_bias)

    self.x2l_cross_attn = {}
    self.l2x_cross_attn = {}
    self.x_network = {}
    self.l_network = {}
    for i, (x_layers, l_layers) in enumerate(self.layer_configs):
      self.x_network[str(i)] = TransformerDecoder(
          x_layers,
          dim=x_dim,
          mlp_ratio=mlp_ratio,
          num_heads=x_num_heads,
          drop_path=drop_path,
          drop_units=drop_units,
          drop_att=drop_att,
          cross_attention=False,
          name='x_network' + suffix_id(i))
      if l_layers > 0:
        self.l2x_cross_attn[str(i)] = TransformerDecoderLayer(
            dim=0,
            mlp_ratio=0,
            num_heads=min(latent_num_heads, x_num_heads),
            drop_path=0.,
            drop_units=0.,
            drop_att=0.,
            dim_x_att=min(latent_dim, x_dim),
            self_attention=False,
            cross_attention=True,
            use_mlp=False,
            use_enc_ln=True,
            name='l2x_cross_attn' + suffix_id(i))
        self.l_network[str(i)] = TransformerDecoder(
            l_layers,
            dim=latent_dim,
            mlp_ratio=mlp_ratio,
            num_heads=latent_num_heads,
            drop_path=drop_path,
            drop_units=drop_units,
            drop_att=drop_att,
            cross_attention=False,
            name='l_network' + suffix_id(i))
      if i < len(self.layer_configs) - 1:
        self.x2l_cross_attn[str(i)] = TransformerDecoderLayer(
            dim=0,
            mlp_ratio=0,
            num_heads=min(x_num_heads, latent_num_heads),
            drop_path=0.,
            drop_units=0.,
            drop_att=0.,
            dim_x_att=min(latent_dim, x_dim),
            self_attention=False,
            cross_attention=True,
            use_mlp=False,
            use_enc_ln=True,
            name='x2l_cross_attn' + suffix_id(i))
    self.x_output_ln = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name='x_output_ln')
    self.l_output_ln = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name='l_output_ln')

  def _latent_shift(self, latents, s_len):
    """latents shape change: b t m d -> (b t) m d."""
    latents_leading, latents_last = latents[:, :-1], latents[:, -1:]
    latents = tf.concat([tf.zeros_like(latents_last), latents_leading], axis=1)
    latents = einops.rearrange(latents, 'b t m d -> (b t) m d', t=s_len)
    return latents, latents_last

  def _latent_shift_back(self, latents, latents_last, s_len):
    """latents shape change: (b t) m d -> b t m d."""
    latents = einops.rearrange(latents, '(b t) m d -> b t m d', t=s_len)
    latents = tf.concat([latents[:, 1:], latents_last], axis=1)
    return latents

  def call(self, x, encoded=None, training=True):
    """x (e.g. token id) is an integer tensor with shape of [bsz, t, n]."""
    del encoded  # not implemented.
    bsz = tf.shape(x)[0]
    t = self.num_groups
    x_mask, l_mask = self.x_mask, self.l_mask
    if self.shared_embedding:
      inp_embedding = outp_embedding = self.token_embedding
    else:
      inp_embedding = self.inp_token_embedding
      outp_embedding = self.outp_token_embedding
    x = tf.gather(inp_embedding, x)
    x = x + self.x_pos_emb[tf.newaxis, tf.newaxis, ...]
    latents = tf.reshape(self.latent_pos_emb,
                         [1, t, self.latents_per_group, -1])
    latents = tf.tile(latents, [bsz, 1, 1, 1])

    x = einops.rearrange(x, 'b t n c -> (b t) n c')
    for i in range(len(self.layer_configs)):
      x = self.x_network[str(i)](
          x, None, None, x_mask, None, training=training)[0]

      if self.layer_configs[i][-1] > 0:
        latents = einops.rearrange(latents, 'b t m d -> (b t) m d')
        latents = self.l2x_cross_attn[str(i)](
            latents, x, None, None, None, training=training)[0]
        latents = einops.rearrange(latents, '(b t) m d -> b (t m) d', t=t)
        latents = self.l_network[str(i)](
            latents, None, None, l_mask, None, training=training)[0]
        latents = einops.rearrange(latents, 'b (t m) d -> b t m d', t=t)
        if i < len(self.layer_configs) - 1:
          latents, latents_last = self._latent_shift(latents, t)
          x = self.x2l_cross_attn[str(i)](
              x, latents, None, None, None, training=training)[0]
          latents = self._latent_shift_back(latents, latents_last, t)

    x = einops.rearrange(x, '(b t) n d -> b t n d', t=t)
    logits = tf.einsum('btnd,kd->btnk', self.x_output_ln(x), outp_embedding)
    if self.output_bias:
      logits = tf.nn.bias_add(logits, self.outp_bias)
    return logits

  def _latent_shift(self, latents, s_len):
    """latents shape change: b t m d -> (b t) m d."""
    latents_leading, latents_last = latents[:, :-1], latents[:, -1:]
    latents = tf.concat([tf.zeros_like(latents_last), latents_leading], axis=1)
    latents = einops.rearrange(latents, 'b t m d -> (b t) m d', t=s_len)
    return latents, latents_last

  def _latent_shift_back(self, latents, latents_last, s_len):
    """latents shape change: (b t) m d -> b t m d."""
    latents = einops.rearrange(latents, '(b t) m d -> b t m d', t=s_len)
    latents = tf.concat([latents[:, 1:], latents_last], axis=1)
    return latents
