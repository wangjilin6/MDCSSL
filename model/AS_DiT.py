import torch
from torch import nn
from timm.models.vision_transformer import PatchEmbed, Mlp
import math
import numpy as np
import torch.nn.functional as F
from .ASSA_ASCA import AdaptiveSpareCrossAttention, AdaptiveSpareSelfAttention


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# T嵌入
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class ECGEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, ecg_length, hidden_size, ecg_channels=2):
        super().__init__()
        self.norm = nn.LayerNorm(ecg_length, elementwise_affine=False, eps=1e-6)
        self.conv = nn.Conv1d(in_channels=ecg_channels, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.embedding_table = nn.Linear(ecg_length, hidden_size)
        self.ecg_length = ecg_length

    def forward(self, ecg):
        ecg = self.norm(ecg)
        ecg = F.relu(self.conv(ecg)).squeeze(1)
        embeddings = F.relu(self.embedding_table(ecg))
        return embeddings


class ECGEmbedder2(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, ecg_length, hidden_size, ecg_channels=4, patch_num=64):
        super().__init__()
        self.norm = nn.LayerNorm(ecg_length, elementwise_affine=False, eps=1e-6)
        self.conv = nn.Conv1d(in_channels=ecg_channels, out_channels=patch_num, kernel_size=3, stride=1, padding=1)
        self.embedding_table = nn.Linear(ecg_length, hidden_size)
        self.ecg_length = ecg_length

    def forward(self, ecg):
        ecg = self.norm(ecg)
        ecg = F.relu(self.conv(ecg))
        embeddings = F.relu(self.embedding_table(ecg))
        return embeddings


# 定义DiT块
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = AdaptiveSpareSelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.cross_attn = AdaptiveSpareCrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, c1, c64):
        shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c1).chunk(9, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mca.unsqueeze(1) * self.cross_attn(modulate(self.norm2(x), shift_mca, scale_mca), c64)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of model.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            input_size=64,
            patch_size=8,
            in_channels=3,
            hidden_size=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0,
            condition_length=256,
            learn_sigma=True,
            ecg_channels=4,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.condition_embedder1 = ECGEmbedder(condition_length, hidden_size, ecg_channels=4)
        self.condition_embedder64 = ECGEmbedder2(condition_length, hidden_size, patch_num=num_patches,
                                                 ecg_channels=ecg_channels)

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.condition_embedder1.embedding_table.weight, std=0.02)
        nn.init.normal_(self.condition_embedder64.embedding_table.weight, std=0.02)
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in model blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward(self, x, t, condition):
        # print(x.shape,t.shape)
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)
        condition1 = self.condition_embedder1(condition)
        condition64 = self.condition_embedder64(condition)
        c1 = t + condition1
        c64 = t.unsqueeze(1) + condition64

        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c1, c64)  # (N, T, D)
        x = self.final_layer(x, c1)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   model Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2, 'DiT-XL/4': DiT_XL_4, 'DiT-XL/8': DiT_XL_8,
    'DiT-L/2': DiT_L_2, 'DiT-L/4': DiT_L_4, 'DiT-L/8': DiT_L_8,
    'DiT-B/2': DiT_B_2, 'DiT-B/4': DiT_B_4, 'DiT-B/8': DiT_B_8,
    'DiT-S/2': DiT_S_2, 'DiT-S/4': DiT_S_4, 'DiT-S/8': DiT_S_8,
}


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


class DiT_Classifier(torch.nn.Module):
    def __init__(self, model: DiT, num_classes, encoder_layer=6, flag=False):
        super().__init__()
        self.pos_embed = model.pos_embed
        self.x_embedder = model.x_embedder
        self.condition_embedder1 = model.condition_embedder1
        self.condition_embedder64 = model.condition_embedder64
        self.t_embedder = model.t_embedder
        # depth = model.blocks.__len__() // 2
        self.encoder_layer = encoder_layer
        self.encoder = model.blocks[:self.encoder_layer]
        self.pos_embed.requires_grad = flag

        requires_grad(self.x_embedder, flag)
        requires_grad(self.condition_embedder1, flag)
        requires_grad(self.condition_embedder64, flag)
        requires_grad(self.t_embedder, flag)
        requires_grad(self.encoder, flag)

        self.ln = nn.LayerNorm(model.hidden_size, elementwise_affine=False, eps=1e-6)
        # self.gp = nn.AdaptiveAvgPool1d(16)
        self.flatten = nn.Flatten()
        self.head = nn.Linear(self.x_embedder.num_patches * model.hidden_size, num_classes)
        # self.bn = nn.BatchNorm2d()

    def set_encoder(self, model: DiT, flag=False):
        self.pos_embed = model.pos_embed
        self.x_embedder = model.x_embedder
        self.condition_embedder1 = model.condition_embedder1
        self.condition_embedder64 = model.condition_embedder64
        self.t_embedder = model.t_embedder
        # depth = model.blocks.__len__() // 2
        self.encoder = model.blocks[:self.encoder_layer]

        self.pos_embed.requires_grad = flag

        requires_grad(self.x_embedder, flag)
        requires_grad(self.condition_embedder1, flag)
        requires_grad(self.condition_embedder64, flag)
        requires_grad(self.t_embedder, flag)
        requires_grad(self.encoder, flag)

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward(self, x, t, conditon):
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)
        conditon1 = self.condition_embedder1(conditon)
        conditon64 = self.condition_embedder64(conditon)
        c1 = t + conditon1
        c64 = t.unsqueeze(1) + conditon64
        for block in self.encoder:
            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c1, c64)  # (N, T, D)

        # print(x.shape)
        x = self.ln(x)
        # x = self.gp(x)
        x = self.flatten(x)
        x = self.head(x)
        # print(x.shape)

        return x


if __name__ == '__main__':
    device = 'cuda'
    model = DiT_models["DiT-S/8"](input_size=64, in_channels=1)
    # model = model.to(device)

    # from torchinfo import summary
    # x = torch.randn(1, 3, 128, 128)
    # t = torch.from_numpy(np.array([100]))
    # y = model(x,t)
    # print(y.shape)
    # summary(model,[(128,1,64,64),(128,),(128,256)])

    classifier = DiT_Classifier(model, 5)
    # summary(classifier, [(128,1,64,64),(128,),(128,256)])
    x = torch.rand((128, 1, 64, 64))
    t = torch.rand(128)
    ecg = torch.rand((128, 4, 256))
    y = classifier(x, t, ecg)
    print(y.shape)
