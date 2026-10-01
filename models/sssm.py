"""Paper SSSM loss backed by a frozen, pretrained infrared encoder."""

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class SSSMFeatureEncoder(nn.Module):
    """Return the four encoder scales and the final bottleneck feature."""

    def __init__(self, config):
        super().__init__()
        from .dfsmamba.utils import GenPatchEmbed2D
        from .dfsmamba.vision_mamba import VMEncoder

        self.patch_embed = GenPatchEmbed2D(config)
        self.encoder = VMEncoder(config)

    def forward(self, images):
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        bottleneck, encoder_scales = self.encoder(self.patch_embed(images))
        return tuple(encoder_scales) + (bottleneck,)


class SSSMLoss(nn.Module):
    """Five-scale SSSM using externally pretrained, frozen encoder weights.

    ``encoder_weights`` may be a checkpoint path or a state-dict-like mapping.
    Checkpoints may contain the state dict directly or under one of
    ``feature_encoder``, ``sssm_encoder``, ``encoder_state_dict``, or
    ``state_dict``. Generator-style ``vm_encoder.*`` keys are remapped to the
    feature encoder's ``encoder.*`` keys.
    """

    def __init__(
        self,
        encoder_weights,
        lambda_weight=1.0,
        scale_weights=(1.0, 0.5, 0.25, 0.125, 0.125),
        input_size=256,
        device="cpu",
    ):
        super().__init__()
        if encoder_weights is None:
            raise ValueError(
                "SSSM requires pretrained infrared encoder weights."
            )
        if len(scale_weights) != 5:
            raise ValueError("SSSM requires exactly five scale weights.")
        if any(weight < 0 for weight in scale_weights):
            raise ValueError("SSSM scale weights must be non-negative.")
        if input_size <= 0:
            raise ValueError("SSSM input_size must be positive.")

        from .dfsmamba.configs import get_train_gen_config

        self.lambda_weight = float(lambda_weight)
        self.input_size = int(input_size)
        self.feature_encoder = SSSMFeatureEncoder(
            get_train_gen_config()
        ).to(device)
        self.register_buffer(
            "scale_weights",
            torch.tensor(scale_weights, dtype=torch.float32),
        )
        self.load_encoder_weights(encoder_weights, device)
        self._freeze_encoder()

    @staticmethod
    def _unwrap_state_dict(payload):
        if not isinstance(payload, Mapping):
            raise TypeError("SSSM weights must be a mapping or checkpoint path.")

        if (
            isinstance(payload.get("patch_embed"), Mapping)
            and isinstance(payload.get("encoder"), Mapping)
        ):
            state_dict = {
                "patch_embed." + key: value
                for key, value in payload["patch_embed"].items()
            }
            state_dict.update({
                "encoder." + key: value
                for key, value in payload["encoder"].items()
            })
            return state_dict

        for key in (
            "feature_encoder",
            "sssm_encoder",
            "encoder_state_dict",
            "encoder",
            "state_dict",
            "model",
        ):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                return candidate
        return payload

    @staticmethod
    def _normalize_key(key):
        wrapper_prefixes = (
            "module.",
            "model.",
            "autoencoder.",
            "feature_encoder.",
            "sssm_encoder.",
        )
        changed = True
        while changed:
            changed = False
            for prefix in wrapper_prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
                    break
        if key.startswith("vm_encoder."):
            key = "encoder." + key[len("vm_encoder."):]
        return key

    def load_encoder_weights(self, encoder_weights, device="cpu"):
        if isinstance(encoder_weights, (str, Path)):
            checkpoint_path = Path(encoder_weights)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    "SSSM encoder checkpoint not found: %s" % checkpoint_path
                )
            payload = torch.load(checkpoint_path, map_location=device)
        else:
            payload = encoder_weights

        state_dict = self._unwrap_state_dict(payload)
        normalized = {
            self._normalize_key(key): value
            for key, value in state_dict.items()
        }
        expected_keys = set(self.feature_encoder.state_dict())
        encoder_state = {
            key: value
            for key, value in normalized.items()
            if key in expected_keys
        }
        missing_keys = sorted(expected_keys - set(encoder_state))
        if missing_keys:
            preview = ", ".join(missing_keys[:5])
            raise RuntimeError(
                "SSSM checkpoint is missing %d encoder keys (first: %s)"
                % (len(missing_keys), preview)
            )
        self.feature_encoder.load_state_dict(encoder_state, strict=True)

    def _freeze_encoder(self):
        self.feature_encoder.eval()
        for parameter in self.feature_encoder.parameters():
            parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(False)
        self.feature_encoder.eval()
        return self

    def _resize(self, images):
        if images.shape[-2:] == (self.input_size, self.input_size):
            return images
        return F.interpolate(
            images,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, real_infrared, generated_infrared):
        generated_features = self.feature_encoder(
            self._resize(generated_infrared)
        )
        with torch.no_grad():
            real_features = self.feature_encoder(
                self._resize(real_infrared)
            )

        losses = [
            weight * F.l1_loss(generated, real, reduction="mean")
            for weight, generated, real in zip(
                self.scale_weights,
                generated_features,
                real_features,
            )
        ]
        return self.lambda_weight * torch.stack(losses).sum()
