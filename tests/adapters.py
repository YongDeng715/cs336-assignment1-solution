from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from cs336_basics.bpe_tokenizer import BPETokenizer, train_bpe
from jaxtyping import Bool, Float, Int
from torch import Tensor

#%% Transformer LM Architecture
def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    # 使用torch.matmul执行线性变换: y = xW^T
    # 其中in_features的形状是(..., d_in),weights.T的形状是(d_in, d_out)
    return in_features @ weights.T


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    return weights[token_ids]


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight

    # step1:将输入in_features投影到高维度,作为上采样
    value = in_features@w1_weight.T  # 形状: [..., d_ff]
    # step2:将输入in_features投影到高维度,作为门控信号
    gate = in_features@w3_weight.T  # 形状: [..., d_ff]
    # step3:对value应用swish激活函数
    swish_value = value * torch.sigmoid(value)  # 形状: [..., d_ff]
    # step4:门控激活 (Swish(xW₁) ⊙ (xW₃))
    GLU = swish_value * gate  # 形状: [..., d_ff]
    # step5:下采样回原始维度
    output = GLU@w2_weight.T  # 形状: [..., d_model]
    return output


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    # step1: calculate attention scores
    attnScores = Q @ K.transpose(-2,-1)

    #step2: normalize
    d_k=Q.shape[-1]
    attnScores = attnScores / torch.sqrt(torch.tensor(d_k, dtype=attnScores.dtype))

    #step3: apply mask
    if mask is not None:
        attnScores = attnScores.masked_fill(~mask, -1e9)
        
    #step4: softmax and output
    softmaxAttnScores=torch.softmax(attnScores,dim=-1)
    ans = softmaxAttnScores @ V
    return ans


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    print(f"\nd_model: {d_model}, num_heads: {num_heads}")
    print(f"q_proj_weight: {q_proj_weight.shape}")
    print(f"k_proj_weight: {k_proj_weight.shape}")
    print(f"v_proj_weight: {v_proj_weight.shape}")
    print(f"o_proj_weight:{o_proj_weight.shape}")
    print(f"in_features:{in_features.shape}")

    """
    tests/test_model.py::test_multihead_self_attention 
    num_heads: 4, d_k_per_head: 16，d_model: 64, 
    q_proj_weight: torch.Size([64, 64])
    k_proj_weight: torch.Size([64, 64])
    v_proj_weight: torch.Size([64, 64])
    o_proj_weight:torch.Size([64, 64])
    in_features:torch.Size([4, 12, 64])

    第一个64:d_k - 每个头的查询/键维度
    第二个64:d_in - 输入特征维度
    关键理解:虽然 q_proj_weight 形状是 [64, 64],但这实际上是所有头的投影权重合并在一起。因为 num_heads = 4,所以:
    每个头的实际维度:d_k_per_head = 64 / 4 = 16
    总投影权重大小:d_k = 16 x 4 = 64

    2. 输入特征形状:[4, 12, 64]
    in_features: torch.Size([4, 12, 64])
    第一个4:batch_size - 批次大小(4个样本)
    第二个12:sequence_length - 序列长度(12个token)
    第三个64:d_in - 输入特征维度(与投影权重的第二个维度匹配）

    1. q_proj_weight - 投影权重矩阵
    类型：权重参数（可训练的参数）
    作用：将输入特征映射到查询向量：用于将输入特征转换为查询向量
    形状：[d_k, d_in]（输出维度 × 输入维度）
    2. Q 向量 - 查询向量
    类型：计算得到的张量
    作用：实际的查询向量，用于注意力计算
    形状：[batch_size, seq_len, d_k]
    """
    batch_size, seq_len, d_in = in_features.shape
    d_q = q_proj_weight.shape[0]
    d_k = k_proj_weight.shape[0]
    d_v = v_proj_weight.shape[0]

    d_q_per_head = d_q // num_heads
    d_k_per_head = d_k // num_heads
    d_v_per_head = d_v // num_heads

    #step1:calcluate Q,K,V
    Q = run_linear(d_in=d_in, d_out=d_k, weights=q_proj_weight, in_features=in_features) #4,12,64
    K = run_linear(d_in=d_in, d_out=d_k, weights=k_proj_weight, in_features=in_features)
    V = in_features @ v_proj_weight.T

    #step2:reshape
    Q = Q.view(batch_size, seq_len, num_heads, d_k_per_head)#4,12,4,16
    K = K.view(batch_size, seq_len, num_heads, d_k_per_head)#4,12,4,16
    V = V.view(batch_size, seq_len, num_heads, d_v_per_head)#4,12,4,16

    #step3:Transformer standard format of Q,K,V
    Q = Q.transpose(1,2)#4,4,12,16
    K = K.transpose(1,2)
    V = V.transpose(1,2)

    #step4:attention scores
    attnScores = Q @ K.transpose(-2,-1)
    #step5:scale and mask
    attnScores = attnScores/torch.sqrt(torch.tensor(d_k_per_head, dtype=attnScores.dtype))
    mask = torch.tril(torch.ones(seq_len, seq_len, device=attnScores.device)).bool()
    attnScores = attnScores.masked_fill(mask == 0, float('-inf'))

    #step6:softmax and output
    softmaxAttnScores = torch.softmax(attnScores, dim=-1)
    tmp_output = softmaxAttnScores @ V

    # step7: 合并多头输出
    # 当前形状: [batch_size, num_heads, seq_len, d_v_per_head] → 目标形状: [batch_size, seq_len, d_v]

    output = tmp_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_v)
    #print(f"合并多头输出之后output.shape={output.shape}")

    # step8: 投影到输出维度
    output = output @ o_proj_weight.T
    #print(f"投影到输出维度之后output.shape={output.shape}")
    return output


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    #step1:确认嵌入维度是偶数
    d_k = in_query_or_key.shape[-1]
    seq_len = in_query_or_key.shape[-2]
    if d_k % 2 != 0:
        raise ValueError("d_k must be even for RoPE")
    
    #step2:确认旋转角度theta, 频率计算: θ^(-2i/d_k) 
    i = torch.arange(0, d_k//2, dtype=torch.float32, device=in_query_or_key.device)
    freqs = theta ** (-2 * i / d_k)  # 形状 [d_k//2]

    # step3:扩展 freqs 以匹配最大序列长度
    freqs = freqs.unsqueeze(0).expand(max_seq_len, -1)  # (max_seq_len, d_k // 2)
    
    # step4:创建位置索引
    positions = torch.arange(max_seq_len).unsqueeze(1)          # (max_seq_len, 1)
    
    """
    角度计算: 位置 x 频率
    Token "I" (位置0): [0x1.0=0°, 0x0.01=0°]
    Token "love" (位置1): [1x1.0=1°, 1x0.01=0.01°]  
    Token "apple" (位置2): [2x1.0=2°, 2x0.01=0.02°]
    """
    # step5: 计算角度 (向量化操作)
    angles = positions * freqs  # 广播：形状 [*batch_dims, seq_len, d_k//2]

    #step6:创建旋转矩阵
    cos_emb = torch.cos(angles)
    sin_emb = torch.sin(angles)
    cos_emb, sin_emb = cos_emb[:seq_len], sin_emb[:seq_len]

    #step7:x转换成复数
    x_even = in_query_or_key[..., ::2]
    x_odd = in_query_or_key[..., 1::2]

    #step8:计算旋转
    rotated_even = x_even * cos_emb - x_odd * sin_emb
    rotated_odd = x_even * sin_emb + x_odd * cos_emb

    # step9:重新组合维度
    result = torch.zeros_like(in_query_or_key)
    result[..., ::2] = rotated_even
    result[..., 1::2] = rotated_odd
    return result


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """

    # 输入 → RMSNorm → 多头自注意力 → 残差连接 → RMSNorm → 前馈网络 → 残差连接 → 输出
    batch_size, seq_len, d_model = in_features.shape

    #step1:第一个RMSNorm
    in_feat_1stRMSNorm=run_rmsnorm(d_model=d_model, eps=1e-5, weights=weights['ln1.weight'], in_features=in_features)

    #step2:多头注意力
    token_positions = torch.arange(seq_len).expand(batch_size,-1)
    MHA = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=seq_len,
        theta=theta,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=in_feat_1stRMSNorm,
        token_positions=token_positions
    )
    
    #step3:第一个残差连接
    MHA_fristResidual = in_features + MHA

    #step4:第二个RMSNorm
    MHA_1stResidual = run_rmsnorm(d_model=d_model,
                                eps=1e-5,
                                weights=weights['ln2.weight'],
                                in_features=MHA_fristResidual)
    
    #step5:前馈网络
    FFN = run_swiglu(d_model=d_model,
                    d_ff=d_ff,
                    w1_weight=weights['ffn.w1.weight'],
                    w2_weight=weights['ffn.w2.weight'],
                    w3_weight=weights['ffn.w3.weight'],
                    in_features=MHA_1stResidual)
    
    return MHA_1stResidual + FFN


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    #step1:获取token_embeddings, 即将token转换为embedding之后的向量
    token_embeddings = run_embedding(vocab_size, d_model, weights['token_embeddings.weight'], in_indices) #[batch_size,sequence_length,d_model]=[4,12,64]
    batch_size, seq_len, d_model = token_embeddings.shape
    #step2:构建多层transformer block，将embedding之后的token输入transformer block
    for layer_idx in range(num_layers):
        #构建当前层的权重字典
        layer_weights={
        'attn.q_proj.weight':weights[f'layers.{layer_idx}.attn.q_proj.weight'],
        'attn.k_proj.weight':weights[f'layers.{layer_idx}.attn.k_proj.weight'],
        'attn.v_proj.weight':weights[f'layers.{layer_idx}.attn.v_proj.weight'],
        'attn.output_proj.weight':weights[f'layers.{layer_idx}.attn.output_proj.weight'],
        'ln1.weight':weights[f'layers.{layer_idx}.ln1.weight'],
        'ffn.w1.weight':weights[f'layers.{layer_idx}.ffn.w1.weight'],
        'ffn.w2.weight':weights[f'layers.{layer_idx}.ffn.w2.weight'],
        'ffn.w3.weight':weights[f'layers.{layer_idx}.ffn.w3.weight'],
        'ln2.weight':weights[f'layers.{layer_idx}.ln2.weight'],
        }
        # 注意这里将当前层的输出赋值给token_embeddings，作为下一层的输入，因为左边只能是token_embeddings
        token_embeddings = run_transformer_block(d_model,num_heads,d_ff,context_length,rope_theta,layer_weights,token_embeddings)

    #step3:将多层transformer输出通过RMSNorm
    final_output = run_rmsnorm(d_model=d_model,eps=1e-5,weights=weights['ln_final.weight'],in_features=token_embeddings)
    logits = run_linear(d_in=d_model,d_out=vocab_size,weights=weights['lm_head.weight'],in_features=final_output) #[batch_size, sequence_length, vocab_size]=[4,12,10000]

    return logits


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    output=in_features.clone()
    
    for i in range(in_features.shape[0]):   # each batch

        for j in range(in_features.shape[1]):   # each token
            #step1:计算均方根rms
            token_feature=in_features[i][j][:]
            token_feature_squared=token_feature**2
            mean_token_feature_squared=torch.mean(token_feature_squared)
            rms=torch.sqrt(mean_token_feature_squared+eps)
            #step2:归一化
            output[i][j][:]=(in_features[i][j][:]/rms)*weights
    return output


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    output = in_features * torch.sigmoid(in_features)
    return output


#%% Transformer LM Training 
def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    print(f"\nin_features.shape={in_features.shape}") #[3,5]

    #step1:计算每行最大值
    max_values = torch.max(in_features, dim=dim, keepdim=True)[0]
    print(f"max_values.shape={max_values.shape}") #[3,1]
    #step2:减去最大值
    exp_vals = torch.exp(in_features - max_values)
    #step3:计算softmax
    softmax_vals = exp_vals / torch.sum(exp_vals, dim=dim, keepdim=True)
    return softmax_vals
    #return torch.softmax(in_features, dim=dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


# %% BPE Tokenizer
def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return BPETokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    return train_bpe(input_path, vocab_size, special_tokens)
