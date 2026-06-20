import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import warnings
from timm.models.layers import DropPath, to_2tuple


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):   
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C) 
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


# ---------------------------------------------------------
# RMSNorm (as used in the differential attention paper)
# ---------------------------------------------------------
class RMSNorm(nn.Module):
    """
    RMSNorm normalizes the input tensor by its root-mean-square (RMS) value.

    Given an input x ∈ ℝ^(...×d), it computes:

        RMS(x) = sqrt(mean(x², dim=-1, keepdim=True) + ε)
        output = x / RMS(x)

    Optionally, a learnable weight is applied if elementwise_affine is True.

    Args:
        dim (int): Dimension to normalize.
        eps (float): A value added for numerical stability.
        elementwise_affine (bool): If True, multiply by a learnable weight.
    """
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.weight is not None}'



# ---------------------------------------------------------
# SPSA
# ---------------------------------------------------------
class SpatiallyProjectedSpectralAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., lambda_init=0.8):
        super().__init__()
        if num_heads % 2 != 0:
            raise ValueError("num_heads must be even for Differential Attention.")
        self.dim = dim
        num_heads = 1
        self.num_heads = num_heads
        self.head_dim = window_size[0] * window_size[0] // num_heads 


        # Spatial Varing Projection
        self.q_proj = nn.Linear(window_size[0] * window_size[0], window_size[0] * window_size[0], bias=qkv_bias) 
        self.k_proj = nn.Linear(window_size[0] * window_size[0], window_size[0] * window_size[0], bias=qkv_bias)
        self.v_proj = nn.Linear(window_size[0] * window_size[0], window_size[0] * window_size[0], bias=qkv_bias)

        self.out_proj = nn.Linear(window_size[0] * window_size[0], window_size[0] * window_size[0], bias=True)  # final output projection

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        self.window_size = window_size
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)



        # Channel Position Bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * self.dim - 1, num_heads)
        )


        coords = torch.arange(self.dim) 
        relative_coords = coords[:, None] - coords[None, :]
        relative_coords += self.dim - 1
        relative_position_index = relative_coords
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, window_size*window_size, C).

        Returns:
            Tensor of shape (B, window_size*window_size, C) after applying Window Spectral Attention.
        """
        B, N, c = x.shape
        x = x.transpose(-1,-2) # Spatial-Channel Dimension Transformation

        # Spatial Varing Projection
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, c, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, c, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, c, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))

        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01)).to(self.logit_scale.device)).exp()

        attn_scores = attn_scores * logit_scale
        # Add position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.dim, self.dim, -1
        ).permute(2, 0, 1).contiguous()
        attn_scores = attn_scores + relative_position_bias.unsqueeze(0)


        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).reshape(B, c, N)
        x_out = self.out_proj(attn_output).transpose(-1,-2)
        x_out = self.proj_drop(x_out)
        return x_out


class SpectralSwinTransformerBlock(nn.Module):
    """ Spectral Swin Transformer Block.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
       
        self.attn = SpatiallyProjectedSpectralAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        # x: [b,h,w,c]
        B, H, W, C = x.shape
        x = self.norm1(x)
        

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W/SW
        attn_windows = self.attn(x_windows)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W/SW
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class SpecHSA(nn.Module):
    def __init__(
            self,
            dim,
            stage=1,
    ):
        super().__init__()
        self.wa = SpectralSwinTransformerBlock(dim=dim, input_resolution=256 // (2 ** stage),
                                     num_heads=2 ** stage, window_size=8,
                                     shift_size=0)
        self.swa = SpectralSwinTransformerBlock(dim=dim, input_resolution=256 // (2 ** stage),
                                     num_heads=2 ** stage, window_size=8,
                                     shift_size=4)
        self.pn = PreNorm(dim, FeedForward(dim=dim))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        x = self.wa(x) + x
        x = self.swa(x) + x
        x = self.pn(x) + x
        out = x.permute(0, 3, 1, 2)
        return out



# ---------------------------------------------------------
# VDSA
# ---------------------------------------------------------
class VisualDifferentialSpatialAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., lambda_init=0.8):
        super().__init__()
        if num_heads % 2 != 0:
            raise ValueError("num_heads must be even for Differential Attention.")
        self.dim = dim
        self.num_heads = num_heads
        self.effective_heads = num_heads // 2
        self.head_dim = dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.out_proj = nn.Linear(dim, dim, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.diff_norm = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # Learnable lambda parameters (shared across all heads)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_init = lambda_init
        self.window_size = window_size
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)



    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, N, c).

        Returns:
            Tensor of shape (B, N, c) after applying differential attention.
        """
        B, N, c = x.shape

        # Compute Q, K, V projections.
        q = self.q_proj(x)+x
        k = self.k_proj(x)+x
        v = self.v_proj(x)+x

        # Reshape Q and K into (B, N, 2 * h_effective, head_dim)
        q = q.view(B, N, 2 * self.effective_heads, self.head_dim)
        k = k.view(B, N, 2 * self.effective_heads, self.head_dim)
        # Reshape V into (B, N, h_effective, 2 * head_dim)
        v = v.view(B, N, self.effective_heads, 2 * self.head_dim)

        # Transpose to bring head dimension forward.
        # q, k: (B, 2 * h_effective, N, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        # v: (B, h_effective, N, 2 * head_dim)
        v = v.transpose(1, 2)

        # Scale Q.
        q = q * self.scaling

        # Compute raw attention scores: (B, 2 * h_effective, N, N)
        attn_scores = torch.matmul(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))

        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01)).to(self.logit_scale.device)).exp()
        attn_scores = attn_scores * logit_scale

        # Add position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        attn_scores = attn_scores + relative_position_bias.unsqueeze(0)

        # Compute attention probabilities.
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        # Reshape to separate the two halves: (B, h_effective, 2, N, N)
        attn_probs = attn_probs.view(B, self.effective_heads, 2, N, N)

        # Compute lambda via re-parameterization.
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # Differential attention: subtract the second attention map scaled by lambda_full.
        diff_attn = attn_probs[:, :, 0, :, :] - lambda_full * attn_probs[:, :, 1, :, :]  # shape: (B, h_effective, N, N)
 
        # Multiply the differential attention weights with V.
        attn_output = torch.matmul(diff_attn, v)  # shape: (B, h_effective, N, 2 * head_dim)

        # Apply RMSNorm (headwise normalization) and scale by (1 - lambda_init)
        attn_output = self.diff_norm(attn_output) * (1 - self.lambda_init)

        # Concatenate heads: reshape from (B, h_effective, N, 2 * head_dim) → (B, N, 2 * h_effective * head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, 2 * self.effective_heads * self.head_dim)

        # Final linear projection.
        x_out = self.out_proj(attn_output)
        x_out = self.proj_drop(x_out)
        return x_out

class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
       
        self.attn = VisualDifferentialSpatialAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, 
                                  attn_drop=attn_drop, proj_drop=drop, lambda_init=0.8)


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        # x: [b,h,w,c]
        B, H, W, C = x.shape
        x = self.norm1(x)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W/SW
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W/SW
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops
    
class HSA(nn.Module):
    def __init__(
            self,
            dim,
            stage=1,
    ):
        super().__init__()
        self.wa = SwinTransformerBlock(dim=dim, input_resolution=256 // (2 ** stage),
                                     num_heads=2 ** stage, window_size=8,
                                     shift_size=0)
        self.swa = SwinTransformerBlock(dim=dim, input_resolution=256 // (2 ** stage),
                                     num_heads=2 ** stage, window_size=8,
                                     shift_size=4)
        self.pn = PreNorm(dim, FeedForward(dim=dim))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        x = self.wa(x) + x
        x = self.swa(x) + x
        x = self.pn(x) + x
        out = x.permute(0, 3, 1, 2)
        return out
    



class CascadedVDSPA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            attention_type = 'full'
    ):
        super().__init__()
        self.sa = HSA(dim)

        self.speca = SpecHSA(dim)

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x_mid = x_in.permute(0,3,1,2)
        x=self.sa(x_mid)

        x = self.speca(x).permute(0,2,3,1)
        return x


class CascadedVDSPAStage(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=1,
            attention_type = 'full'
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                CascadedVDSPA(dim=dim, dim_head=dim_head, heads=heads,attention_type = attention_type),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out



class VDSPT(nn.Module):
    def __init__(self, dim=64, stage=2, in_channel = 31,
        out_channel = 31,  num_blocks=[2,1],attention_type = 'full'):
        super(VDSPT, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.embedding = nn.Conv2d(in_channel, self.dim, 3, 1, 1, bias=False)
        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                CascadedVDSPAStage(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim,attention_type = attention_type),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False)
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = CascadedVDSPAStage(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1],attention_type = attention_type)
        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                CascadedVDSPAStage(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim,attention_type = attention_type),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_channel, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
       
        # Embedding
        fea = self.embedding(x)
    
        # Encoder
        fea_encoder = []
        for (CDW, FeaDownSample) in self.encoder_layers:
            fea = CDW(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)
        # Decoder
        for i, (FeaUpSample, Fution, CDW) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            fea = CDW(fea)

        # Mapping
        out = self.mapping(fea) + x
        
        return out




if __name__ == '__main__':
    # model
    model = VDSPT(dim=64, stage=2, num_blocks=[2, 1]).cuda()
    print("Model created.")

    # Generate input tensor with shape [b, c, h, w]
    b, c, h, w = 4, 31, 256, 256  # Example shape
    x = torch.randn(b, c, h, w).cuda()  # Randomly generate tensor and move to GPU
    print("Input tensor shape:", x.shape)

    # Forward pass
    output = model(x)
    print("Output tensor shape:", output.shape)