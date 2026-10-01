"""Detail-Guided Module (DGM)."""

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

    return x_LL, x_LH, x_HL, x_HH


class DWT(nn.Module):
    def forward(self, x):
        return dwt_init(x)


class FeatureMapping(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureMapping, self).__init__()
        hidden_channels = max(in_channels // 2, 1)
        self.conv1 = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        x = self.silu(self.conv1(x))
        x = self.conv2(x)
        return x


class FeatureModulation(nn.Module):
    def __init__(self, detail_channels, modulation_channels, scale_factor):
        super(FeatureModulation, self).__init__()
        self.mapping = FeatureMapping(
            detail_channels * 4,
            modulation_channels,
        )
        self.scale_factor = scale_factor

    def forward(self, large_feature_map, detail_feature_map):
        modulation_params = self.mapping(detail_feature_map)

        if self.scale_factor != 1:
            modulation_params = F.interpolate(
                modulation_params,
                scale_factor=self.scale_factor,
                mode='bilinear',
                align_corners=False,
            )

        desired_size = (large_feature_map.size(2), large_feature_map.size(3))
        if modulation_params.shape[-2:] != desired_size:
            modulation_params = F.interpolate(
                modulation_params,
                size=desired_size,
                mode='bilinear',
                align_corners=False,
            )
        return large_feature_map * modulation_params


class SmallScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SmallScaleFeatureExtractor, self).__init__()

        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv3x3(x)


class LargeScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LargeScaleFeatureExtractor, self).__init__()

        self.conv7x7 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            padding=3,
        )

    def forward(self, x):
        return self.conv7x7(x)


class DGM(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor,
        wavelet_channels,
        use_dwt=True,
    ):
        super(DGM, self).__init__()

        if scale_factor <= 0:
            raise ValueError('DGM scale_factor must be positive.')
        if wavelet_channels <= 0:
            raise ValueError('DGM wavelet_channels must be positive.')

        self.use_dwt = use_dwt
        self.small_scale_extractor = SmallScaleFeatureExtractor(
            in_channels,
            wavelet_channels,
        )
        self.large_scale_extractor = LargeScaleFeatureExtractor(
            in_channels,
            wavelet_channels,
        )
        self.dwt = DWT()
        self.feature_modulation = FeatureModulation(
            detail_channels=wavelet_channels,
            modulation_channels=wavelet_channels,
            scale_factor=scale_factor,
        )
        self.fusion_conv = nn.Conv2d(
            wavelet_channels * 2,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        small_scale_features = self.small_scale_extractor(x)
        large_scale_features = self.large_scale_extractor(x)

        if self.use_dwt:
            wavelet_features = torch.cat(
                self.dwt(small_scale_features),
                dim=1,
            )
            modulated_large_scale_features = self.feature_modulation(
                large_scale_features,
                wavelet_features,
            )
        else:
            modulated_large_scale_features = large_scale_features

        combined_features = torch.cat(
            [small_scale_features, modulated_large_scale_features],
            dim=1,
        )
        fused_features = self.fusion_conv(combined_features)
        fused_features = self.downsample(fused_features)
        fused_features = fused_features.permute(0, 2, 3, 1)
        return fused_features
