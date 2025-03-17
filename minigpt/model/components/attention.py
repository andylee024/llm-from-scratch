import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """Implementation of multihead attention w/ parallel matrix processing"""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        # validate input dimensions
        if (d_out % num_heads != 0):
            raise ValueError("d_out must be divisible by num_heads")

        self.d_out = d_out
        self.num_heads = num_heads
        self.attention_dim = d_out // num_heads

        # setup attention matrices
        self.W_query = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.W_key = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.W_value = nn.Linear(in_features=d_in, out_features=d_out, bias=qkv_bias)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

        # setup dropout
        self.dropout = nn.Dropout(dropout)

        # setup linear
        self.out_proj = nn.Linear(d_out, d_out)

    
    def forward(self, x):
        n, seq_length, _ = x.shape

        # compute Q, K, V matrices
        x_query = self.W_query(x)
        x_key = self.W_key(x)
        x_value = self.W_value(x)

        # reshape to separate into Q = [Q1, Q2, ...], K = [K1, K2, ...]
        x_query = x_query.view(n, seq_length, self.num_heads, self.attention_dim)
        x_query = x_query.transpose(1, 2) # (n, num_heads, seq_length, attention_dim)

        x_key = x_key.view(n, seq_length, self.num_heads, self.attention_dim)
        x_key = x_key.transpose(1, 2) # (n, num_heads, seq_length, attention_dim)
        x_key = x_key.transpose(2, 3) # (n, num_heads, attention_dim, seq_length)

        x_value = x_value.view(n, seq_length, self.num_heads, self.attention_dim)
        x_value = x_value.transpose(1, 2) # (n, num_heads, seq_length, attention_dim)

        # compute attention scores (per-head)
        dk_constant = x_key.shape[-1] ** -0.5
        mask_context = self.mask.bool()[:seq_length, :seq_length] 
        attention_scores = (x_query @ x_key) # (n, num_heads, seq_length, seq_length)
        attention_scores.masked_fill_(mask_context, -torch.inf)

        # compute attention weights 
        # note : no dropout on scores (b/c dropout on -inf is not well-defined)
        attention_weights = torch.softmax(attention_scores * dk_constant, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # compute context
        context = attention_weights @ x_value # (n, num_heads, seq_length, attn_dim)
        context = context.transpose(1, 2) # (n, seq_length, num_heads, attn_dim) , done by convention
        context = context.contiguous().view(n, seq_length, self.d_out) # (n, seq_length, d_out)
        
        # apply linear layer
        return self.out_proj(context)
