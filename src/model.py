"""SLIViT model for glaucoma classification on OCT volumes."""

import torch
import torch.nn as nn
from transformers import ConvNextModel


class TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block with optional input/output projection."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head

        self.norm1 = nn.LayerNorm(dim)
        # When dim != inner_dim, project in/out so MHA gets divisible embed_dim
        self._need_proj = (dim != inner_dim)
        if self._need_proj:
            self.proj_in = nn.Linear(dim, inner_dim)
            self.proj_out = nn.Linear(inner_dim, dim)
            self.attn = nn.MultiheadAttention(inner_dim, heads, dropout=dropout, batch_first=True)
        else:
            self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        normed = self.norm1(x)
        if self._need_proj:
            normed = self.proj_in(normed)
            attn_out, _ = self.attn(normed, normed, normed)
            attn_out = self.proj_out(attn_out)
        else:
            attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, dim=256, depth=5, heads=20, dim_head=64, mlp_dim=512, dropout=0.15):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SLIViT(nn.Module):
    """
    SLIViT: Slice-Level integrating Vision Transformer.

    Input:  (B, 3, num_slices*256, 256) vertically tiled OCT slices
    Output: (B, 1) logit for binary glaucoma classification
    """

    def __init__(
        self,
        num_slices=32,
        vit_dim=256,
        vit_depth=5,
        vit_heads=20,
        vit_dim_head=64,
        vit_mlp_dim=512,
        dropout=0.15,
        convnext_name="facebook/convnext-tiny-224",
        fe_checkpoint=None,
        freeze_fe=True,
    ):
        super().__init__()
        self.num_slices = num_slices
        self.freeze_fe = freeze_fe

        # ConvNeXt-Tiny: keep embeddings + encoder, discard final LN and classifier
        convnext_full = ConvNextModel.from_pretrained(convnext_name)
        children = list(convnext_full.children())
        self.convnext = nn.Sequential(*children[:2])

        # Load SLIViT OCT-pretrained weights if provided
        if fe_checkpoint is not None:
            ckpt = torch.load(fe_checkpoint, map_location="cpu")
            state = {}
            for k, v in ckpt.items():
                if k.startswith("model.convnext."):
                    new_key = k.replace("model.convnext.embeddings.", "0.")
                    new_key = new_key.replace("model.convnext.encoder.", "1.")
                    state[new_key] = v
            missing, unexpected = self.convnext.load_state_dict(state, strict=False)
            if missing:
                print("[SLIViT] Missing keys: %s" % missing)
            if unexpected:
                print("[SLIViT] Unexpected keys: %s" % unexpected)
            print("[SLIViT] Loaded FE from %s" % fe_checkpoint)

        if self.freeze_fe:
            for param in self.convnext.parameters():
                param.requires_grad = False
            self.convnext.eval()

        # Token projection: 768 * 8 * 8 = 49152 -> vit_dim
        self.token_proj = nn.Linear(768 * 8 * 8, vit_dim)

        # Positional embeddings + CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, vit_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_slices + 1, vit_dim))
        self._init_pos_embed()

        # ViT encoder
        self.vit = ViTEncoder(vit_dim, vit_depth, vit_heads, vit_dim_head, vit_mlp_dim, dropout)

        # Classification head
        self.head = nn.Sequential(nn.LayerNorm(vit_dim), nn.Linear(vit_dim, 1))

    def _init_pos_embed(self):
        """Initialize positional embeddings as scaled ordinal positions."""
        with torch.no_grad():
            n = self.num_slices + 1
            ordinals = torch.arange(n, dtype=torch.float32).unsqueeze(1)
            ordinals = ordinals.expand(n, self.pos_embed.shape[-1])
            self.pos_embed.copy_(ordinals.unsqueeze(0) / n)

    def _extract_features(self, x):
        """Run ConvNeXt and reshape output into per-slice tokens."""
        if self.freeze_fe:
            self.convnext.eval()

        embeddings = self.convnext[0](x)
        encoder_out = self.convnext[1](embeddings)
        if hasattr(encoder_out, "last_hidden_state"):
            feat = encoder_out.last_hidden_state
        elif isinstance(encoder_out, tuple):
            feat = encoder_out[0]
        else:
            feat = encoder_out

        # (B, 768, num_slices*8, 8) -> (B, num_slices, 49152)
        B, C, H, W = feat.shape
        feat = feat.view(B, C, self.num_slices, H // self.num_slices, W)
        feat = feat.permute(0, 2, 1, 3, 4)
        return feat.reshape(B, self.num_slices, -1)

    def forward(self, x):
        B = x.shape[0]
        tokens = self._extract_features(x)
        tokens = self.token_proj(tokens)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.vit(tokens)
        return self.head(tokens[:, 0])

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_fe:
            self.convnext.eval()
        return self
