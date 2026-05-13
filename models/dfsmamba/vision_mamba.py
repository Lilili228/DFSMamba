import math
from functools import partial
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from .utils import PatchMerging2D

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"




def generate_mair_ids(H, W, device='cuda'):
    """
    生成4个方向的扫描索引

    返回:
        scan_ids: (4, H*W) 正向扫描索引
        inverse_ids: (4, H*W) 逆向恢复索引
    """
    L = H * W

    # 方向1: 从左到右，从上到下 (原始顺序)
    ids_1 = torch.arange(L, device=device)

    # 方向2: 从右到左（水平翻转）
    ids_2 = torch.arange(L, device=device).view(H, W).flip(1).flatten()

    # 方向3: 从上到下，从左到右 (转置)
    ids_3 = torch.arange(L, device=device).view(H, W).t().contiguous().flatten()

    # 方向4: 转置后水平翻转
    ids_4 = torch.arange(L, device=device).view(H, W).t().flip(1).contiguous().flatten()

    scan_ids = torch.stack([ids_1, ids_2, ids_3, ids_4])

    # 生成逆索引
    inverse_ids = torch.zeros_like(scan_ids)
    for k in range(4):
        inverse_ids[k].scatter_(0, scan_ids[k], torch.arange(L, device=device))

    return scan_ids, inverse_ids


def mair_ids_scan(x, scan_ids):
    """
    根据 scan_ids 重新排列特征图

    参数:
        x: (B, C, H, W) 输入特征
        scan_ids: (K, L) 扫描索引

    返回:
        xs: (B, K, C, L) 重排后的特征
    """
    B, C, H, W = x.shape
    L = H * W
    K = scan_ids.shape[0]

    # 展平空间维度
    x_flat = x.view(B, C, L)  # (B, C, H*W)

    # 根据索引重排
    xs = []
    for k in range(K):
        # 使用 gather 进行索引
        x_scanned = torch.gather(
            x_flat,
            dim=2,
            index=scan_ids[k].unsqueeze(0).unsqueeze(0).expand(B, C, -1)
        )
        xs.append(x_scanned)

    xs = torch.stack(xs, dim=1)  # (B, K, C, L)
    return xs


def mair_ids_inverse(y, inverse_ids, shape):
    """
    将扫描后的特征恢复到原始空间布局

    参数:
        y: (B, K, C, L) 扫描后的特征
        inverse_ids: (K, L) 逆索引
        shape: (B, C, H, W) 目标形状

    返回:
        out: (B, K*C, H, W) 恢复后的特征
    """
    B, K, C, L = y.shape
    _, target_C, H, W = shape

    # 对每个方向做逆变换
    y_inverse = []
    for k in range(K):
        y_k = torch.gather(
            y[:, k],  # (B, C, L)
            dim=2,
            index=inverse_ids[k].unsqueeze(0).unsqueeze(0).expand(B, C, -1)
        )
        y_inverse.append(y_k)

    # 拼接所有方向
    y_merged = torch.stack(y_inverse, dim=1)  # (B, K, C, L)

    return y_merged.view(B, K * C, H, W)  # (B, K*C, H, W)

class ShuffleAttn(nn.Module):
    def __init__(self, in_features, out_features, group=4):
        super().__init__()
        self.group = group
        self.in_features = in_features
        self.out_features = out_features

        self.gating = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, out_features, groups=self.group, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batchsize, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

    def channel_rearrange(self ,x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batchsize, self.group, group_channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

    def forward(self, x):
        x = self.channel_shuffle(x)
        x = self.gating(x)
        x = self.channel_rearrange(x)

        return x

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            use_vmm_strategy=True,  # 使用VMM策略
            use_shuffle_attn=True,  # 使用ShuffleAttn
            input_resolution=None,  # 输入分辨率(用于预生成scan_ids)
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.use_vmm_strategy = use_vmm_strategy
        self.use_shuffle_attn = use_shuffle_attn
        self.input_resolution = input_resolution

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        if self.use_shuffle_attn:
            self.gating = ShuffleAttn(
                in_features=self.d_inner * 4,
                out_features=self.d_inner * 4,
                group=self.d_inner
            )
        if self.use_vmm_strategy and input_resolution is not None:
            H, W = input_resolution
            scan_ids, inverse_ids = generate_mair_ids(H, W, device=device if device else 'cpu')
            self.register_buffer('mair_scan_ids', scan_ids)
            self.register_buffer('mair_inverse_ids', inverse_ids)
        else:
            self.mair_scan_ids = None
            self.mair_inverse_ids = None

        # 选择扫描函数
        self.selective_scan = selective_scan_fn


        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D


    def forward_core_vmm(self, x: torch.Tensor, mair_ids=None):
        """VMM风格实现（使用F.conv1d + mair_ids）"""
        B, C, H, W = x.shape
        L = H * W
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        K = 4

        # 动态生成或使用预存的扫描索引
        if mair_ids is None:
            if self.mair_scan_ids is not None:
                # 使用预生成的索引
                scan_ids = self.mair_scan_ids
                inverse_ids = self.mair_inverse_ids
            else:
                # 动态生成
                scan_ids, inverse_ids = generate_mair_ids(H, W, device=x.device)
        else:
            scan_ids, inverse_ids = mair_ids

        # MaIR扫描
        xs = mair_ids_scan(x, scan_ids)

        # 使用F.conv1d进行投影
        x_dbl = F.conv1d(
            xs.reshape(B, -1, L),
            self.x_proj_weight.reshape(-1, D, 1),
            bias=None,
            groups=K
        )
        dts, Bs, Cs = torch.split(x_dbl.reshape(B, K, -1, L), [R, N, N], dim=2)

        dts = F.conv1d(
            dts.reshape(B, -1, L),
            self.dt_projs_weight.reshape(K * D, -1, 1),
            groups=K
        )

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)

        out_y = self.selective_scan(
            xs, dts,
            -torch.exp(self.A_logs.float()).view(-1, self.d_state),
            Bs, Cs,
            self.Ds.float().view(-1),
            z=None,
            delta_bias=self.dt_projs_bias.float().view(-1),
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return mair_ids_inverse(out_y, inverse_ids, shape=(B, -1, H, W))

    def forward(self, x: torch.Tensor, mair_ids=None, **kwargs):
        """
        统一前向传播接口

        参数:
            x: (B, H, W, C) 输入特征
            mair_ids: 可选的自定义扫描索引
        """
        B, H, W, C = x.shape

        # 输入投影
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # 卷积
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        # 选择策略
        if self.use_vmm_strategy:
            # VMM策略
            y = self.forward_core_vmm(x, mair_ids)

            if self.use_shuffle_attn:
                y = y * self.gating(y)
            # 拆分并融合4个方向
            y1, y2, y3, y4 = torch.chunk(y, 4, dim=1)
            y = y1 + y2 + y3 + y4
            y = y.permute(0, 2, 3, 1).contiguous()
        else:
            # 原始SS2D策略
            y1, y2, y3, y4 = self.forward_corev0(x)
            y = y1 + y2 + y3 + y4
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        # 输出处理
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)

        return out


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        # self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.self_attention = SS2D(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            use_vmm_strategy=True,  # 启用 VMM 策略
            use_shuffle_attn=True,  # 启用 ShuffleAttn 门控
            input_resolution=None,  # 动态分辨率（或设置固定值）
            **kwargs
        )

        self.drop_path = DropPath(drop_path)

    def forward(self, input1: torch.Tensor, **kwargs):
        x = input1 + self.drop_path(self.self_attention(self.ln_1(input1)))
        return x



class VSSLayer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            upsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


class VMDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.mamba.output_chans
        self.num_layers = len(config.mamba.decoder_depths)
        self.dims = config.mamba.embed_dims
        self.embed_dim = config.mamba.embed_dims[0]
        self.num_features = config.mamba.embed_dims[-1]
        self.dims_decoder = config.mamba.embed_dims[::-1]
        self.pos_drop = nn.Dropout(p=config.mamba.drop_rate)
        dpr_decoder = [x.item() for x in
                       torch.linspace(0, config.mamba.drop_path_rate, sum(config.mamba.decoder_depths))][::-1]

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim=self.dims_decoder[i_layer],
                depth=config.mamba.decoder_depths[i_layer],
                d_state=math.ceil(self.dims[0] / 6) if config.mamba.d_state is None else config.mamba.d_state,
                drop=config.mamba.drop_rate,
                attn_drop=config.mamba.attn_drop_rate,
                drop_path=dpr_decoder[
                          sum(config.mamba.decoder_depths[:i_layer]):sum(config.mamba.decoder_depths[:i_layer + 1])],
                norm_layer=config.mamba.norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=config.mamba.use_checkpoint,
            )
            self.layers_up.append(layer)

        self.final_up = Final_PatchExpand2D(dim=self.dims_decoder[-1], dim_scale=4, norm_layer=config.mamba.norm_layer)
        self.final_conv = nn.Conv2d(self.dims_decoder[-1] // 4, self.num_classes, 1)

    def forward_features_up(self, x, skip_list):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skip_list[-inx])
        return x

    def forward_final(self, x):
        x = self.final_up(x)
        x = x.permute(0, 3, 1, 2)
        x = self.final_conv(x)
        return x

    def forward(self, x, skip_list):
        x = self.forward_features_up(x, skip_list)
        x = self.forward_final(x)
        return x




class VMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config.mamba.output_chans
        self.num_layers = len(config.mamba.encoder_depths)
        self.dims = config.mamba.embed_dims
        self.embed_dim = config.mamba.embed_dims[0]
        self.num_features = config.mamba.embed_dims[-1]

        self.pos_drop = nn.Dropout(p=config.mamba.drop_rate)
        dpr = [x.item() for x in torch.linspace(0, config.mamba.drop_path_rate,
                                                sum(config.mamba.encoder_depths))]  # stochastic depth decay rule
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=config.mamba.embed_dims[i_layer],
                depth=config.mamba.encoder_depths[i_layer],
                d_state=math.ceil(
                    config.mamba.embed_dims[0] / 6) if config.mamba.d_state is None else config.mamba.d_state,
                # default 16
                drop=config.mamba.drop_rate,
                attn_drop=config.mamba.attn_drop_rate,
                drop_path=dpr[
                          sum(config.mamba.encoder_depths[:i_layer]):sum(config.mamba.encoder_depths[:i_layer + 1])
                          ],
                norm_layer=config.mamba.norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=config.mamba.use_checkpoint,
            )
            self.layers.append(layer)

    def forward_features(self, x):
        # print(f"Input shape: {x.shape}")
        skip_list = []

        x = self.pos_drop(x)
        # print(f"0. After pos_drop: {x.shape}")

        for i, layer in enumerate(self.layers):
            skip_list.append(x)
            # print(f"{i + 1}. Before VSSLayer {i}: {x.shape}")
            x = layer(x)
            # print(f"   After VSSLayer {i}: {x.shape}")

        return x, skip_list

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x, skip_list = self.forward_features(x)
        return x, skip_list

