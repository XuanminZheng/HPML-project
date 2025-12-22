import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedAttentionBlock(nn.Module):
    """融合LayerNorm + Attention的块"""
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.ln = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, residual=None):
        B, N, C = x.shape

        # Fused: LayerNorm + Linear (QKV projection)
        x_norm = self.ln(x)
        qkv = self.qkv(x_norm)

        # Reshape: (B, N, 3*C) -> (B, N, 3, num_heads, head_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Fused: Attention + Output projection
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2)  # (B, N, num_heads, head_dim)
        out = out.reshape(B, N, C)  # (B, N, C)
        out = self.out_proj(out)

        # Add residual
        if residual is not None:
            out = out + residual

        return out


class FusedFFNBlock(nn.Module):
    """融合LayerNorm + FFN的块"""
    def __init__(self, hidden_dim, ff_ratio=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        ff_dim = int(hidden_dim * ff_ratio)

        self.ln = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.act = nn.GELU()

    def forward(self, x, residual=None):
        # Fused: LayerNorm + Linear1 + GELU + Linear2
        x_norm = self.ln(x)
        hidden = self.fc1(x_norm)
        hidden = self.act(hidden)
        out = self.fc2(hidden)

        # Add residual
        if residual is not None:
            out = out + residual

        return out


class FusedTransformerBlock(nn.Module):
    """融合的完整Transformer Block"""
    def __init__(self, hidden_dim, num_heads, ff_ratio=4):
        super().__init__()
        self.attn_block = FusedAttentionBlock(hidden_dim, num_heads)
        self.ffn_block = FusedFFNBlock(hidden_dim, ff_ratio)

    def forward(self, x):
        # Attention with residual
        x = self.attn_block(x, residual=x)
        # FFN with residual
        x = self.ffn_block(x, residual=x)
        return x


class StandardTransformerBlock(nn.Module):
    """标准Transformer Block（非融合）"""
    def __init__(self, hidden_dim, num_heads, ff_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.ln2 = nn.LayerNorm(hidden_dim)
        ff_dim = int(hidden_dim * ff_ratio)
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.act = nn.GELU()

    def forward(self, x):
        # Attention
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # FFN
        x_norm = self.ln2(x)
        hidden = self.fc1(x_norm)
        hidden = self.act(hidden)
        ffn_out = self.fc2(hidden)
        x = x + ffn_out

        return x


if __name__ == "__main__":
    hidden_dim = 768
    num_heads = 12
    batch_size = 2
    seq_len = 128

    x = torch.randn(batch_size, seq_len, hidden_dim)

    # Benchmark
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    fused_block = FusedTransformerBlock(hidden_dim, num_heads).to(device)
    std_block = StandardTransformerBlock(hidden_dim, num_heads).to(device)
    x = x.to(device)

    # Warmup
    for _ in range(3):
        _ = fused_block(x)
        _ = std_block(x)

    if device == "cuda":
        # Fused
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            _ = fused_block(x)
        end.record()
        torch.cuda.synchronize()
        fused_time = start.elapsed_time(end)

        # Standard
        start.record()
        for _ in range(100):
            _ = std_block(x)
        end.record()
        torch.cuda.synchronize()
        std_time = start.elapsed_time(end)

        print(f"Fused:    {fused_time:.2f}ms")
        print(f"Standard: {std_time:.2f}ms")
        print(f"Speedup:  {std_time/fused_time:.2f}x")
    else:
        import time

        # Fused
        start = time.perf_counter()
        for _ in range(100):
            _ = fused_block(x)
        fused_time = (time.perf_counter() - start) * 1000

        # Standard
        start = time.perf_counter()
        for _ in range(100):
            _ = std_block(x)
        std_time = (time.perf_counter() - start) * 1000

        print(f"Fused:    {fused_time:.2f}ms")
        print(f"Standard: {std_time:.2f}ms")
        print(f"Speedup:  {std_time/fused_time:.2f}x")
