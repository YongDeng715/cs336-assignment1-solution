import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from cs336_basics.transformer import TransformerLM, TransformerBlock, MultiHeadSelfAttention, \
    Embedding, RoPE, SwiGLU, RMSNorm, Linear, scaled_dot_product_attn, softmax
    

all = ["TransformerBlock", "RMSNorm", "SwiGLU", "scaled_dot_product_attn", "RoPE"]