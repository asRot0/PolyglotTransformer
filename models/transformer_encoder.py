import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import LayerNormalization, Dropout
from .multi_head_attention import MultiHeadSelfAttention
from .feed_forward import FeedForwardNetwork


class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads, ff_expansion=4, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_size = embed_size
        self.heads = heads
        self.dense_dim = embed_size * ff_expansion
        # self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.attention = layers.MultiHeadAttention(num_heads=heads, key_dim=embed_size)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.ffn = FeedForwardNetwork(embed_size, ff_expansion, dropout_rate)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, mask=None, training=False):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        # Self-attention layer + Add & Norm
        attention_output = self.attention(x, x, attention_mask=mask)
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.norm1(x + attention_output)

        # Feed-forward layer + Add & Norm
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.norm2(out1 + ffn_output)

        return out2

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_size,
            "num_heads": self.heads,
            "dense_dim": self.dense_dim,
        })
        return config