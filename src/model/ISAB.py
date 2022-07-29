from typing import Optional
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

"""
Multihead Attention Block (MAB), Set Attention Block (SAB) and Induced Set Attention Block (ISAB) from
Set Transformer Paper (https://arxiv.org/pdf/1810.00825.pdf)
code taken and modified from the paper's github repo
"""


class MAB(nn.Module):
    """
    Multihead attention Block (MAB). Performs multihead attention with a residual connection followed by
    a row-wise feedworward layer with a residual connection:

        MAB(X,Y) = LayerNorm(H(X,Y) + rFF(H(X,Y)))

    where

        H(X,Y) = LayerNorm(X + Multihead(X, Y, Y))

    for matrices X, Y. The three arguments for Multihead stand for Query, Value and Key matrices.
    Setting X = Y i.e. MAB(X, X) would result in the type of multi-headed self attention that was used in the original
    transformer paper 'attention is all you need'.
    Furthermore in the original transformer paper a type of positional encoding is used which is not present here.
    """

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=True):
        super(MAB, self).__init__()

        assert (dim_V % num_heads) == 0, 'dimension must be divisible by number of heads'

        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)

        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, mask: Optional[torch.BoolTensor] = None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), dim=0)
        K_ = torch.cat(K.split(dim_split, 2), dim=0)
        V_ = torch.cat(V.split(dim_split, 2), dim=0)

        soft_mask_input = Q_.bmm(K_.transpose(1, 2))/math.sqrt(self.dim_V)

        if mask is not None:
            _mask = mask.repeat((self.num_heads, 1))
            attn_mask = torch.zeros_like(_mask, dtype=torch.float)
            attn_mask.masked_fill_(_mask, float("-inf"))
            soft_mask_input+= attn_mask.unsqueeze(1)

        A = torch.softmax(soft_mask_input, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.gelu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)

        return O, A


class SAB(nn.Module):
    """
    The Set Attention Block (SAB) is defined as

        SAB(X) := MAB(X,X)

    i.e. standard self attention as described in the 'attention is all you need' paper without positional encoding.
    """

    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()

        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, mask: Optional[torch.BoolTensor] = None):
        return self.mab(X, X, mask)[0]


class ISAB(nn.Module):
    """
    The Induced Set Attention Block (ISAB) uses learnable 'inducing points' to reduce the complexity from O(n^2)
    to O(nm) where m is the number of inducing points. While the number of these inducing points is a fixed parameter
    the points themselves are learnable parameters.
    The ISAB is then defined as

        ISAB(X) = MAB(X,H)

    where

        H = MAB(I,X)

    i.e. ISAB(X) = MAB(X, MAB(I, X)).
    """

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.FloatTensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, mask)[0]

        return self.mab1(X, H)[0]
