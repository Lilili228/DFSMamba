import copy
from einops import rearrange
import torch.nn as nn
import torch


# class ChannelAttentionModule(nn.Module):
#     def __init__(self, inc, reduction=16):
#         super(ChannelAttentionModule, self).__init__()
#         mid_channel = inc // reduction
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.shared_MLP = nn.Sequential(
#             nn.Linear(in_features=inc, out_features=mid_channel),
#             nn.ReLU(),
#             nn.Linear(in_features=mid_channel, out_features=inc)
#         )
#         self.sigmoid = nn.Sigmoid()
#         # self.act=SiLU()
#
#     def forward(self, x):
#         # print(f'ChannelAttentionModule x.shape: {x.shape}')
#         # print(f'ChannelAttentionModule x.shape: {self.avg_pool(x).view(x.size(0), -1).shape}')
#         avg_out = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
#         max_out = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
#         out = self.sigmoid(avg_out + max_out)
#         out = out.expand_as(x)
#         return out
#
#
# class SpatialAttentionModule(nn.Module):
#     def __init__(self, inc):
#         super(SpatialAttentionModule, self).__init__()
#         self.shared_conv2d = nn.Sequential(nn.Conv2d(2, inc, 7, stride=1, padding=9, dilation=3),
#                                            nn.Sigmoid())
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         att = torch.cat([avg_out, max_out], dim=1)
#         att = self.shared_conv2d(att)
#         return att
#
#
# class HSCAMBlock(nn.Module):
#     def __init__(self, hidden_size):
#         super(HSCAMBlock, self).__init__()
#         self.channel_attention = ChannelAttentionModule(hidden_size)
#         self.spatial_attention = SpatialAttentionModule(hidden_size)
#         self.conv = nn.Conv2d(hidden_size * 2, hidden_size, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = x.permute(0, 3, 1, 2)
#         # print(f'HSCAMBlock x.shape: {x.shape}')
#         residual = x
#         outc = self.channel_attention(x)
#         outs = self.spatial_attention(x)
#         assert outc.shape == outs.shape, f'{outc, outs} tensors must have the same shape'
#         out = torch.cat([outc, outs], dim=1)
#         out = self.sigmoid(self.conv(out))
#         out = out * outc
#         out = out * outs
#         out = out + residual
#         return out.permute(0, 2, 3, 1)
#
#
# class HSCAMLayer(nn.Module):
#     def __init__(self, config):
#         super(HSCAMLayer, self).__init__()
#         self.layers = nn.ModuleList()
#         for idx_layer in range(len(config.mamba.encoder_depths)):
#             layer = HSCAMBlock(config.mamba.embed_dims[idx_layer])
#             self.layers.append(copy.deepcopy(layer))
#
#     def forward(self, feature_list):
#
#         assert len(feature_list) == len(self.layers), "feature_list must not be None"
#
#         result = []
#         for idx, layer in enumerate(self.layers):
#             # print(f'current layer: {idx}')
#             # print(f'feature_list[{idx}]: {feature_list[idx].shape}')
#             t = layer(feature_list[idx])
#             result.append(t)
#
#         return result


class EnhancedChannelAttentionModule(nn.Module):
    def __init__(self, dim, head_num, window_size=7, down_sample_mode='avg_pool',
                 qkv_bias=False, attn_drop_ratio=0., gate_layer='sigmoid'):
        super(EnhancedChannelAttentionModule, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.window_size = window_size

        # 归一化层
        self.norm = nn.GroupNorm(1, dim)

        # QKV投影
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()

        # 下采样策略
        self.conv_d = nn.Identity()
        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans
                self.conv_d = nn.Conv2d(dim * window_size ** 2, dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=window_size, stride=window_size)

    def space_to_chans(self, x):
        """空间到通道的重组合"""
        b, c, h, w = x.size()
        assert h % self.window_size == 0 and w % self.window_size == 0
        x = rearrange(x, 'b c (h hs) (w ws) -> b (c hs ws) h w',
                      hs=self.window_size, ws=self.window_size)
        return x

    def forward(self, x):
        # 下采样减少计算量
        y = self.down_func(x)
        y = self.conv_d(y)
        _, _, h, w = y.size()

        # 归一化
        y = self.norm(y)

        # 计算QKV
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)

        # 重塑为多头形式
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)',
                      head_num=self.head_num, head_dim=self.head_dim)
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)',
                      head_num=self.head_num, head_dim=self.head_dim)
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)',
                      head_num=self.head_num, head_dim=self.head_dim)

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        attn = attn @ v

        # 重塑回原始形状
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=h, w=w)

        # 全局平均生成通道权重
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)

        return attn


class EnhancedSpatialAttentionModule(nn.Module):
    def __init__(self, dim, group_kernel_sizes=[3, 5, 7, 9], gate_layer='sigmoid'):
        super(EnhancedSpatialAttentionModule, self).__init__()
        self.dim = dim
        self.group_kernel_sizes = group_kernel_sizes
        self.group_chans = dim // 4

        assert dim % 4 == 0, f"Dimension {dim} must be divisible by 4"

        # 多尺度深度卷积
        self.local_dwc = nn.Conv1d(self.group_chans, self.group_chans,
                                   kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2,
                                   groups=self.group_chans)
        self.global_dwc_s = nn.Conv1d(self.group_chans, self.group_chans,
                                      kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2,
                                      groups=self.group_chans)
        self.global_dwc_m = nn.Conv1d(self.group_chans, self.group_chans,
                                      kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2,
                                      groups=self.group_chans)
        self.global_dwc_l = nn.Conv1d(self.group_chans, self.group_chans,
                                      kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2,
                                      groups=self.group_chans)

        # 归一化层
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)

        # 注意力门控
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # 水平方向注意力
        x_h = x.mean(dim=3)  # (B, C, H)
        # 确保正确拆分
        split_size = self.group_chans
        splits_h = torch.split(x_h, split_size, dim=1)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = splits_h

        # 应用多尺度卷积并拼接
        h_features = torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)

        x_h_attn = self.sa_gate(self.norm_h(h_features))
        x_h_attn = x_h_attn.view(b, c, h, 1)

        # 垂直方向注意力
        x_w = x.mean(dim=2)  # (B, C, W)
        splits_w = torch.split(x_w, split_size, dim=1)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = splits_w

        w_features = torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l),
        ), dim=1)

        x_w_attn = self.sa_gate(self.norm_w(w_features))
        x_w_attn = x_w_attn.view(b, c, 1, w)

        return x_h_attn, x_w_attn


class HSCAMBlock(nn.Module):
    def __init__(self, hidden_size, head_num=8, window_size=7,
                 group_kernel_sizes=[3, 5, 7, 9], down_sample_mode='avg_pool',
                 qkv_bias=False, gate_layer='sigmoid'):
        super(HSCAMBlock, self).__init__()

        # 检查hidden_size是否可以被4整除
        if hidden_size % 4 != 0:
            # 调整到最近的可以被4整除的值
            adjusted_size = ((hidden_size + 3) // 4) * 4
            print(f"Warning: hidden_size {hidden_size} is not divisible by 4. Adjusting to {adjusted_size}")
            hidden_size = adjusted_size

        # 检查hidden_size是否可以被head_num整除
        if hidden_size % head_num != 0:
            # 调整head_num到可以整除hidden_size的值
            head_num = self._find_divisor(hidden_size)
            print(f"Warning: hidden_size {hidden_size} is not divisible by head_num. Adjusting head_num to {head_num}")

        self.hidden_size = hidden_size
        self.head_num = head_num

        # 使用SCSA的先进注意力机制
        self.spatial_attention = EnhancedSpatialAttentionModule(
            dim=hidden_size,
            group_kernel_sizes=group_kernel_sizes,
            gate_layer=gate_layer
        )

        self.channel_attention = EnhancedChannelAttentionModule(
            dim=hidden_size,
            head_num=head_num,
            window_size=window_size,
            down_sample_mode=down_sample_mode,
            qkv_bias=qkv_bias,
            gate_layer=gate_layer
        )

        # 可选的残差连接
        self.use_residual = True

    def _find_divisor(self, n):
        """找到可以整除n的最大数（不超过8）"""
        for i in range(8, 0, -1):
            if n % i == 0:
                return i
        return 1  # 如果找不到，返回1

    def forward(self, x):
        # 处理输入格式
        if len(x.shape) == 4:
            if x.shape[1] == self.hidden_size:  # (B, C, H, W)
                x_conv = x
            else:  # (B, H, W, C)
                x_conv = x.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        residual = x_conv

        # SCSA风格的顺序注意力：空间→通道
        # 1. 多尺度双向空间注意力
        x_h_attn, x_w_attn = self.spatial_attention(x_conv)
        spatial_enhanced = x_conv * x_h_attn * x_w_attn

        # 2. 多头自注意力通道注意力
        channel_weights = self.channel_attention(spatial_enhanced)

        # 3. 应用通道注意力
        out = spatial_enhanced * channel_weights

        # 残差连接
        if self.use_residual:
            out = out + residual

        # 转换回原始格式
        if x.shape[1] == self.hidden_size:  # 原本就是 (B, C, H, W)
            return out
        else:
            return out.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)


class HSCAMLayer(nn.Module):
    def __init__(self, config, head_nums=None, window_sizes=None):
        super(HSCAMLayer, self).__init__()
        self.layers = nn.ModuleList()

        # 设置默认值
        if head_nums is None:
            head_nums = [8] * len(config.mamba.encoder_depths)
        if window_sizes is None:
            window_sizes = [7] * len(config.mamba.encoder_depths)

        for idx_layer in range(len(config.mamba.encoder_depths)):
            # 调整embed_dims确保可以被4整除
            embed_dim = config.mamba.embed_dims[idx_layer]
            if embed_dim % 4 != 0:
                embed_dim = ((embed_dim + 3) // 4) * 4
                print(f"Adjusted embed_dims[{idx_layer}] to {embed_dim} for divisibility by 4")

            layer = HSCAMBlock(
                hidden_size=embed_dim,
                head_num=head_nums[idx_layer],
                window_size=window_sizes[idx_layer]
            )
            self.layers.append(layer)

    def forward(self, feature_list):
        assert len(feature_list) == len(self.layers), "feature_list must match layers length"

        result = []
        for idx, layer in enumerate(self.layers):
            output = layer(feature_list[idx])
            result.append(output)

        return result


class FeatureDiffAndProd(nn.Module):
    def __init__(self, config):
        super(FeatureDiffAndProd, self).__init__()

        self.config = config

        self.use_conv = True

        if config.use_hscam:
            self.tf_layers = nn.ModuleList()
            for idx_layer in range(len(config.mamba.encoder_depths)):
                # print(idx_layer)
                # print(config.mamba.embed_dims[idx_layer])
                layer = HSCAMBlock(config.mamba.embed_dims[idx_layer])
                self.tf_layers.append(copy.deepcopy(layer))

            self.vm_layers = nn.ModuleList()
            for idx_layer in range(len(config.mamba.encoder_depths)):
                layer = HSCAMBlock(config.mamba.embed_dims[idx_layer])
                self.vm_layers.append(copy.deepcopy(layer))

        if self.use_conv:
            self.conv_layers = nn.ModuleList()
            for idx_layer in range(len(config.mamba.encoder_depths)):
                layer = nn.Conv2d(config.mamba.embed_dims[idx_layer], config.mamba.embed_dims[idx_layer],
                                  kernel_size=3, padding=1)
                self.conv_layers.append(copy.deepcopy(layer))

        self.skip_conv_layers = nn.ModuleList()
        for idx_layer in range(len(config.mamba.encoder_depths)):
            layer = nn.Conv2d(config.mamba.embed_dims[idx_layer] * 2, config.mamba.embed_dims[idx_layer],
                              kernel_size=3, padding=1)
            self.skip_conv_layers.append(copy.deepcopy(layer))

    def forward(self, tf_feature_list, vm_feature_list):
        assert len(tf_feature_list) == len(vm_feature_list), \
            "The dimensions of the two input parameters must be consistent."

        if self.config.use_hscam:
            df = []
            for idx, layer in enumerate(self.tf_layers):
                t = layer(tf_feature_list[idx])
                df.append(t)

            pf = []
            for idx, layer in enumerate(self.vm_layers):
                t = layer(vm_feature_list[idx])
                pf.append(t)

            tf_feature_list = df
            vm_feature_list = pf

        diffs = []
        prods = []
        for idx, (t, v) in enumerate(zip(tf_feature_list, vm_feature_list)):
            diff = torch.abs(v - t)
            prod = torch.mul(v, t)
            if self.use_conv:
                conv = self.conv_layers[idx]
                diff = diff.permute(0, 3, 1, 2).contiguous()
                prod = prod.permute(0, 3, 1, 2).contiguous()
                diff = conv(diff)
                prod = conv(prod)
                diff = diff.permute(0, 2, 3, 1).contiguous()
                prod = prod.permute(0, 2, 3, 1).contiguous()
            diffs.append(diff)
            prods.append(prod)

        dp_convs = []
        for idx, (d, p) in enumerate(zip(diffs, prods)):
            c = torch.cat((d, p), 3)
            c = c.permute(0, 3, 1, 2).contiguous()
            skip_conv = self.skip_conv_layers[idx]
            c = skip_conv(c)
            c = c.permute(0, 2, 3, 1).contiguous()
            dp_convs.append(c)

        return dp_convs, diffs, prods

# Test...
# batch_size, in_channels, H, W = 4, 256, 512, 512
# x = torch.randn(batch_size, in_channels, H, W)
# model = HSCAM(layer_chans=[64, 128, 256, 512], layer=2)
# # print(model)
# total = sum([param.nelement() for param in model.parameters()])
# # 精确地计算：1MB=1024KB=1048576字节
# print('Number of parameter: % .4fM' % (total / 1e6))
# out = model(x)
# print("Output shape:", out.shape)
