from onmt.decoders.transformer import TransformerDecoder


class ConvTransformerDecoder(TransformerDecoder):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 copy_attn, self_attn_type, dropout, attention_dropout,
                 embeddings, max_relative_positions, aan_useffn,
                 full_context_alignment, alignment_layer,
                 alignment_heads):
        super().__init__(num_layers, d_model, heads, d_ff,
                         copy_attn, self_attn_type, dropout, attention_dropout,
                         embeddings, max_relative_positions, aan_useffn,
                         full_context_alignment, alignment_layer,
                         alignment_heads)

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        self.state["src"] = enc_hidden
        self.state["cache"] = None
