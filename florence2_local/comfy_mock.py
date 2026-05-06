import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Mock comfy module
class MockOps:
    Linear = nn.Linear
    Embedding = nn.Embedding
    LayerNorm = nn.LayerNorm
    Conv2d = nn.Conv2d
    def cast_to_input(self, tensor, target): return tensor.to(target.dtype)

class MockAttention:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, q, k, v, num_heads, mask=None, **kwargs):
        # Accept both [B, N, C] and [B, H, N, D] formats used by the local Florence code.
        if q.dim() == 4:
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return attn.transpose(1, 2).reshape(q.shape[0], q.shape[2], q.shape[1] * q.shape[3])

        if q.dim() == 3:
            bsz, q_len, dim = q.shape
            head_dim = dim // num_heads
            q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(bsz, -1, num_heads, head_dim).transpose(1, 2)
            v = v.view(bsz, -1, num_heads, head_dim).transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return attn.transpose(1, 2).reshape(bsz, q_len, dim)

        raise ValueError(f"Formato de tensor inesperado no MockAttention: q.shape={tuple(q.shape)}")

def optimized_attention_for_device(*args, **kwargs):
    return MockAttention()

class MockComfy:
    ops = MockOps()
    class ldm:
        class modules:
            class attention:
                optimized_attention_for_device = optimized_attention_for_device
    class utils:
        class ProgressBar:
            def __init__(self, *args): pass
            def update(self, *args): pass

def inject_comfy_mock():
    sys.modules['comfy'] = MockComfy
    sys.modules['comfy.ops'] = MockComfy.ops
    sys.modules['comfy.ldm'] = MockComfy.ldm
    sys.modules['comfy.ldm.modules'] = MockComfy.ldm.modules
    sys.modules['comfy.ldm.modules.attention'] = MockComfy.ldm.modules.attention
    sys.modules['comfy.utils'] = MockComfy.utils
