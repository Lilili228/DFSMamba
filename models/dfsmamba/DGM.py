import torch
import torch.nn as nn
import torch.nn.functional as F


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    min_height = min(x1.size(2), x2.size(2), x3.size(2), x4.size(2))
    min_width = min(x1.size(3), x2.size(3), x3.size(3), x4.size(3))

    x1 = x1[:, :, :min_height, :min_width]
    x2 = x2[:, :, :min_height, :min_width]
    x3 = x3[:, :, :min_height, :min_width]
    x4 = x4[:, :, :min_height, :min_width]

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class ChannelAttentionModified(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModified, self).__init__()
        # Correctly initialize AdaptiveMaxPool2d to squeeze spatial dimensions to 1x1
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Output size is (1, 1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Use max pooling to capture detail information
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(max_out)


class WaveletAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(WaveletAttention, self).__init__()

        self.dwt = DWT()
        self.channel_attention = ChannelAttentionModified(in_channels, reduction_ratio)

    def forward(self, x):
        dwt_result = self.dwt(x)

        cA, cH, cV, cD = dwt_result

        cA = F.interpolate(cA, scale_factor=2, mode='bicubic', align_corners=False)
        cH = F.interpolate(cH, scale_factor=2, mode='bicubic', align_corners=False)
        cV = F.interpolate(cV, scale_factor=2, mode='bicubic', align_corners=False)
        cD = F.interpolate(cD, scale_factor=2, mode='bicubic', align_corners=False)

        return cA, cH, cV, cD


class FeatureMapping(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureMapping, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.silu(self.conv1(x))
        x = self.conv2(x)
        return x


class FeatureModulation(nn.Module):
    def __init__(self, in_channels, wavelet_channels, scale_factor=1):
        super(FeatureModulation, self).__init__()
        self.mapping = FeatureMapping(in_channels * 4, wavelet_channels * scale_factor)
        self.scale_factor = scale_factor

    def forward(self, large_feature_map, detail_feature_map):
        # Generate modulation parameters from detail features
        modulation_params = self.mapping(detail_feature_map)

        if self.scale_factor > 1:
            modulation_params = F.interpolate(modulation_params, scale_factor=self.scale_factor, mode='bilinear')

        # Apply modulation to the large scale feature map
        desired_size = (large_feature_map.size(2), large_feature_map.size(3))
        modulation_params = F.interpolate(modulation_params, size=desired_size, mode='bilinear', align_corners=False)
        modulated_feature_map = large_feature_map * modulation_params
        return modulated_feature_map


class SmallScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SmallScaleFeatureExtractor, self).__init__()

        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv3x3(x)


class LargeScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LargeScaleFeatureExtractor, self).__init__()

        self.dwconv7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.dwconv7x7(x)
        return self.pointwise(x)


class DGM(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, wavelet_channels):
        super(DGM, self).__init__()

        # mid_channels = in_channels // 2
        mid_channels = in_channels
        wavelet_channels = in_channels
        self.small_scale_extractor = SmallScaleFeatureExtractor(in_channels, mid_channels)
        self.large_scale_extractor = LargeScaleFeatureExtractor(in_channels, in_channels)

        self.wavelet_attention = WaveletAttention(mid_channels)

        self.feature_modulation = FeatureModulation(mid_channels, wavelet_channels, scale_factor=1)

        self.fusion_conv = nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1)
        self.downsample = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )

        # Fix the parameters in this module
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        small_scale_features = self.small_scale_extractor(x)
        large_scale_features = self.large_scale_extractor(x)

        cA, attn_cH, attn_cV, attn_cD = self.wavelet_attention(small_scale_features)

        recombined_features = torch.cat((cA, attn_cH, attn_cV, attn_cD), dim=1)

        modulated_large_scale_features = self.feature_modulation(large_scale_features, recombined_features)

        combined_features = torch.cat([modulated_large_scale_features, small_scale_features], dim=1)
        # print("combined_features shape:",combined_features.shape)
        # combined_features = torch.cat([large_scale_features, small_scale_features], dim=1)
        fused_features = self.fusion_conv(combined_features)
        fused_features = self.downsample(fused_features)
        fused_features = fused_features.permute(0, 2, 3, 1)
        return fused_features



