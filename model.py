
import torch.nn as nn
from torchvision.models import vit_b_32, ViT_B_32_Weights
from torchvision.models.vision_transformer import EncoderBlock 

#++++++++++++++++++++++++++++++++++++++++ Basic ViT model wrapper +++++++++++++++++++++++++++++++++++++++#
class ModifiedViTB32(nn.Module):
    def __init__(self, num_classes=100):
        super(ModifiedViTB32, self).__init__()
        self.vit = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)

#++++++++++++++++++++++++++++++++++++++++ Factorized ViT components +++++++++++++++++++++++++++++++++++++++#
class FactorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=16, bias=True):
        super().__init__()
        # Low-rank factorization: Linear(in_features -> rank) then Linear(rank -> out_features)
        # Keep only the second bias to mirror a single Linear as closely as possible
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        self.linear2 = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x):
        return self.linear2(self.linear1(x))


# An Encoder layer in a ViT looks like
'''
  (encoder): Encoder(
    (dropout): Dropout(p=0.0, inplace=False)
    (layers): Sequential(
      (encoder_layer_0): EncoderBlock(
        (ln_1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (self_attention): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
        )
        (dropout): Dropout(p=0.0, inplace=False)
        (ln_2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        (mlp): MLPBlock(
          (0): Linear(in_features=768, out_features=3072, bias=True)
          (1): GELU(approximate='none')
          (2): Dropout(p=0.0, inplace=False)
          (3): Linear(in_features=3072, out_features=768, bias=True)
          (4): Dropout(p=0.0, inplace=False)
        )
      )We will create the factorized version
'''

class FactorizedMLPBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.0, rank=16):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = FactorizedLinear(in_features, hidden_features, rank=rank)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(p=dropout)
        self.fc2 = FactorizedLinear(hidden_features, out_features, rank=rank)
        self.drop2 = nn.Dropout(p=dropout)

        # Matches the standard MLP structure: Linear -> GELU -> Dropout -> Linear -> Dropout

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class FactorizedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, rank=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # Factorized projections for query, key, value
        self.q_proj = FactorizedLinear(embed_dim, embed_dim, rank=rank)
        self.k_proj = FactorizedLinear(embed_dim, embed_dim, rank=rank)
        self.v_proj = FactorizedLinear(embed_dim, embed_dim, rank=rank)

        # Output projection
        self.out_proj = FactorizedLinear(embed_dim, embed_dim, rank=rank)

    def forward(self, tokens, attn_mask=None):
        query, key, value = tokens, tokens, tokens  # Self-attention
        B, N, C = query.size()

        # Project inputs to multi-head QKV
        q = self.q_proj(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            attn_scores += attn_mask

        attn_weights = attn_scores.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(B, N, C)
        
        # Final output projection
        attn_output = self.out_proj(attn_output)

        return attn_output
    
class FactorizedEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0, rank=16):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.self_attention = FactorizedMultiheadAttention(embed_dim, num_heads, dropout=dropout, rank=rank)
        self.dropout1 = nn.Dropout(dropout)
        self.ln_2 = nn.LayerNorm(embed_dim, eps=1e-6)
        hidden_features = int(embed_dim * mlp_ratio)
        self.mlp = FactorizedMLPBlock(in_features=embed_dim, hidden_features=hidden_features, dropout=dropout, rank=rank)

    def forward(self, x):
        # Self-attention block
        attn_output = self.self_attention(self.ln_1(x))
        x = x + self.dropout1(attn_output)

        # MLP block
        mlp_output = self.mlp(self.ln_2(x))
        x = x + mlp_output

        return x





def replace_encoder_with_factorized(module: nn.Module, rank: int = 16, default_dropout: float = 0.0):
    for name, child in module.named_children():
        if isinstance(child, EncoderBlock):
            # Replace with FactorizedEncoderBlock
            factorized_block = FactorizedEncoderBlock(
                embed_dim=child.ln_1.normalized_shape[0],
                num_heads=child.self_attention.num_heads,
                mlp_ratio=child.mlp[0].out_features / child.mlp[0].in_features,
                dropout=default_dropout,
                rank=rank
            )
            setattr(module, name, factorized_block)
        else:
            replace_encoder_with_factorized(child, rank, default_dropout)
    return module

class FactorizedViTB32(nn.Module):
    def __init__(self, num_classes=100, rank=16, default_dropout=0.0):
        super(FactorizedViTB32, self).__init__()
        self.vit = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
        replace_encoder_with_factorized(self.vit, rank=rank, default_dropout=default_dropout)
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)