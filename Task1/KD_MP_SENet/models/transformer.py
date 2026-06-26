import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import MultiheadAttention, GRU, Linear, LayerNorm, Dropout


class FFN(nn.Module):
    """
    原版：GRU(d_model, 2*d_model, 1, bidirectional=True)
    现在：hidden = int(d_model * ffn_mult)，默认 ffn_mult=2.0 -> 完全等价原版
    """
    def __init__(self, d_model, bidirectional=True, dropout=0, ffn_mult: float = 2.0):
        super(FFN, self).__init__()
        hidden = int(round(d_model * ffn_mult))
        hidden = max(1, hidden)

        self.gru = GRU(d_model, hidden, 1, bidirectional=bidirectional, batch_first=True)
        out_dim = hidden * (2 if bidirectional else 1)
        self.linear = Linear(out_dim, d_model)

        self.dropout = Dropout(dropout)

    def forward(self, x):
        # x: [B, T, C] because TransformerBlock uses batch_first=True
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class TransformerBlock(nn.Module):
    """
    最小侵入改动：
    - 增加 ffn_mult 和 ffn_bidirectional 两个可控超参
    - 默认 (ffn_mult=2.0, ffn_bidirectional=True) 与原版完全一致
    """
    def __init__(
        self,
        d_model,
        n_heads,
        bidirectional=True,   # 保持兼容（attention 不用它，但你原来传了）
        dropout=0,
        ffn_mult: float = 2.0,
        ffn_bidirectional: bool = True,
    ):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(d_model)
        self.attention = MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = Dropout(dropout)

        self.norm2 = LayerNorm(d_model)
        self.ffn = FFN(
            d_model,
            bidirectional=ffn_bidirectional,
            dropout=dropout,
            ffn_mult=ffn_mult,
        )
        self.dropout2 = Dropout(dropout)

        self.norm3 = LayerNorm(d_model)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        xt = self.norm1(x)
        xt, _ = self.attention(
            xt, xt, xt,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x = x + self.dropout1(xt)

        xt = self.norm2(x)
        xt = self.ffn(xt)
        x = x + self.dropout2(xt)

        x = self.norm3(x)
        return x
