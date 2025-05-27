import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)


class Attention(nn.Module):
    def __init__(self, in_cdim, heads=8, head_dim=64, dropout=0.1):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = in_cdim**-0.5

        self.to_qkv = nn.Linear(in_cdim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, in_cdim), nn.Dropout(dropout))

    def forward(self, x):
        H = self.heads
        SCALE = self.scale

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=H), qkv)
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * SCALE
        attn = dots.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, in_cdim, heads, head_dim, dropout, mlp_dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.LayerNorm(in_cdim),
                            Attention(in_cdim, heads=heads, head_dim=head_dim, dropout=dropout),
                        )
                    ),
                    Residual(
                        nn.Sequential(
                            nn.LayerNorm(in_cdim),
                            FeedForward(in_cdim, mlp_dim, dropout=dropout),
                        )
                    ),
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Patch_Embedding(nn.Module):

    def __init__(self, in_cdim=3, out_cdim=768):
        super().__init__()
        self.proj = nn.Linear(in_cdim, out_cdim)

    def forward(self, x):
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.proj(x)
        return x


def Sequence_2_Image(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class Feature_Pyramid_Network(nn.Module):
    def __init__(self, in_cdim_list, out_cdim=512):
        super(Feature_Pyramid_Network, self).__init__()
        # self.max_pool1 = nn.AdaptiveMaxPool2d((16, 8))
        cdim_1, cdim_2, cdim_3 = in_cdim_list

        self.patch_embedding_1 = Patch_Embedding(in_cdim=cdim_1, out_cdim=cdim_1)
        self.attention_1 = Transformer(cdim_1, heads=8, head_dim=64, dropout=0.1, mlp_dim=2048, depth=1)

        self.patch_embedding_2 = Patch_Embedding(in_cdim=cdim_1 + cdim_2, out_cdim=cdim_1 + cdim_2)
        self.attention_2 = Transformer(cdim_1 + cdim_2, heads=8, head_dim=64, dropout=0.1, mlp_dim=2048, depth=1)

        self.patch_embedding_3 = Patch_Embedding(in_cdim=cdim_1 + cdim_2 + cdim_3, out_cdim=cdim_1 + cdim_2 + cdim_3)
        self.attention_3 = Transformer(cdim_1 + cdim_2 + cdim_3, heads=8, head_dim=64, dropout=0.1, mlp_dim=2048, depth=1)

    def forward(self, input_list):
        # feature_maps_1: (B, C, H, W) -> (B, 256, 64, 32)
        # feature_maps_2: (B, 2C, H/2, W/2) -> (B, 512, 32, 16)
        # feature_maps_3: (B, 4C, H/4, W/4) -> (B, 1024, 16, 8)
        # feature_maps_4: (B, 8C, H/4, W/4) -> (B, 2048, 16, 8)
        feature_maps_1, feature_maps_2, feature_maps_3 = input_list

        feature_maps_1 = F.interpolate(feature_maps_1, size=(16, 8), mode="bilinear")
        token_1 = self.patch_embedding_1(feature_maps_1)
        token_1 = self.attention_1(token_1)
        feature_maps_1 = Sequence_2_Image(token_1, h=16, w=8)

        feature_maps_2 = F.interpolate(feature_maps_2, size=(16, 8), mode="bilinear")
        token_2 = self.patch_embedding_2(torch.cat([feature_maps_1, feature_maps_2], dim=1))
        token_2 = self.attention_2(token_2)
        feature_maps_2 = Sequence_2_Image(token_2, h=16, w=8)

        token_3 = self.patch_embedding_3(torch.cat([feature_maps_2, feature_maps_3], dim=1))
        token_3 = self.attention_3(token_3)
        feature_maps_3 = Sequence_2_Image(token_3, h=16, w=8)

        return feature_maps_3


if __name__ == "__main__":
    B = 64

    feature_maps_1 = torch.randn(B, 256, 64, 32)
    feature_maps_2 = torch.randn(B, 512, 32, 16)
    feature_maps_3 = torch.randn(B, 1024, 16, 8)
    inter_feature_maps = [feature_maps_1, feature_maps_2, feature_maps_3]

    # feature_maps_4 = torch.randn(B, 2048, 16, 8)
    # attension_feature_maps = inter_feature_maps + [feature_maps_4]

    model = Feature_Pyramid_Network(in_cdim_list=[256, 512, 1024])
    print(model)

    outs = model(inter_feature_maps)
    print(outs.shape)
