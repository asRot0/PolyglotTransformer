import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import LayerNormalization, Dropout
from .multi_head_attention import MultiHeadSelfAttention
from .feed_forward import FeedForwardNetwork


class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads, ff_expansion=4, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_size = embed_size
        self.heads = heads
        self.dense_dim = embed_size * ff_expansion
        # self.self_attention = MultiHeadSelfAttention(embed_size, heads)
        # self.enc_dec_attention = MultiHeadSelfAttention(embed_size, heads)
        self.self_attention = layers.MultiHeadAttention(num_heads=heads, key_dim=embed_size)
        self.enc_dec_attention = layers.MultiHeadAttention(num_heads=heads, key_dim=embed_size)

        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.norm3 = LayerNormalization(epsilon=1e-6)

        self.ffn = FeedForwardNetwork(embed_size, ff_expansion, dropout_rate)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)
        self.supports_masking = True

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),
            tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)

    def call(self, x, enc_output, padding_mask=None, training=False):
        causal_mask = self.get_causal_attention_mask(x)

        if padding_mask is not None:
            padding_mask = tf.cast(padding_mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        # Self-attention with causal mask (decoder-only)
        self_attn_output = self.self_attention(
            query=x,
            value=x,
            key=x,
            attention_mask=causal_mask,
            )
        self_attn_output = self.dropout1(self_attn_output, training=training)
        out1 = self.norm1(x + self_attn_output)

        # Cross-attention with optional padding mask (encoder-decoder)
        enc_dec_attn_output = self.enc_dec_attention(
            query=out1,
            value=enc_output,
            key=enc_output,
            attention_mask=padding_mask,
            )
        enc_dec_attn_output = self.dropout2(enc_dec_attn_output, training=training)
        out2 = self.norm2(out1 + enc_dec_attn_output)

        # Feed-forward network
        ffn_output = self.ffn(out2, training=training)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.norm3(out2 + ffn_output)

        return out3

    def get_config(self):
        config = super().get_config()
        config.update({
        "embed_dim": self.embed_size,
        "num_heads": self.heads,
        "dense_dim": self.dense_dim,
        })
        return config