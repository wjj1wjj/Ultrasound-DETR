import torch
import torch.nn as nn
from .dcnv4 import DCNv4_Block  



class DeformMixFFN_v2(nn.Module):
    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 ffn_drop=0.,
                 dropout_layer=None):
        super().__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.activate = nn.GELU()

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

        fc1 = nn.Conv2d(embed_dims, feedforward_channels, kernel_size=1, stride=1)
        pe_conv = nn.Conv2d(feedforward_channels, feedforward_channels, kernel_size=3,
                            stride=1, padding=1, bias=True, groups=feedforward_channels)
        fc2 = nn.Conv2d(feedforward_channels, embed_dims, kernel_size=1, stride=1)

        self.dcnv4_block = DCNv4_Block(
            c1=embed_dims,
            c2=embed_dims,
            k=3,
            s=1,
            depth=1,
            groups=2,
            downsample=False
        )

        drop = nn.Dropout(ffn_drop)
        self.layers = nn.Sequential(fc1, pe_conv, self.activate, drop, fc2, drop)

        self.dropout_layer = DropPath(
            dropout_layer['drop_prob']) if dropout_layer else nn.Identity()

    def forward(self, x):
        # input x: [B, C, H, W]
        identity = x
        x = x + self.dropout_layer(self._apply_norm(self.norm1, self.dcnv4_block(x)))
        out = self.layers(x)
        out = x + self.dropout_layer(self._apply_norm(self.norm2, out))
        return out

    def _apply_norm(self, norm, x):
        # (B, C, H, W) → (B, H, W, C) → LayerNorm → (B, C, H, W)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x