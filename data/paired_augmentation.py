"""Shared geometry and stable identity helpers for registered image pairs."""

import hashlib
import os
import random

from PIL import Image, ImageEnhance
import torch


_MODALITY_SUFFIXES = (
    '_visible', '_thermal', '_vis', '_rgb', '_lwir', '_infrared', '_ir',
)


def make_sample_id(path, root=None, prefix='pair'):
    relative = os.path.relpath(path, root) if root else os.path.basename(path)
    directory, filename = os.path.split(relative)
    stem = os.path.splitext(filename)[0]
    lower = stem.lower()
    for suffix in _MODALITY_SUFFIXES:
        if lower.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    normalized = os.path.join(directory, stem).replace(os.sep, '/')
    return '%s:%s' % (prefix, normalized)


def sample_id_hash(sample_id):
    digest = hashlib.blake2b(
        sample_id.encode('utf-8'), digest_size=8
    ).digest()
    # Stay inside signed int64 so PyTorch's default collator is portable.
    return int.from_bytes(digest, 'big') & ((1 << 63) - 1)


def indexed_sample_id(prefix, index):
    return '%s:%08d' % (prefix, index)


def augment_pil_pair(visible, infrared, opt):
    """Apply one sampled geometry to both modalities, then separate light jitter."""
    size = (opt.loadSize, opt.loadSize)
    visible = visible.resize(size, Image.BICUBIC)
    infrared = infrared.resize(size, Image.BICUBIC)

    rotation = float(opt.paired_rotation_degrees)
    if opt.isTrain and rotation > 0:
        angle = random.uniform(-rotation, rotation)
        visible = visible.rotate(angle, resample=Image.BICUBIC)
        infrared = infrared.rotate(angle, resample=Image.BICUBIC)

    if opt.isTrain and opt.visible_brightness_jitter > 0:
        amount = float(opt.visible_brightness_jitter)
        visible = ImageEnhance.Brightness(visible).enhance(
            random.uniform(max(0.0, 1.0 - amount), 1.0 + amount)
        )
    if opt.isTrain and opt.infrared_brightness_jitter > 0:
        amount = float(opt.infrared_brightness_jitter)
        infrared = ImageEnhance.Brightness(infrared).enhance(
            random.uniform(max(0.0, 1.0 - amount), 1.0 + amount)
        )
    return visible, infrared


def augment_tensor_pair(visible, infrared, opt, paired_extras=()):
    """Tensor equivalent used by array-backed datasets."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as transform_functional

    extras = list(paired_extras)
    rotation = float(opt.paired_rotation_degrees)
    if opt.isTrain and rotation > 0:
        angle = random.uniform(-rotation, rotation)
        visible = transform_functional.rotate(
            visible, angle, interpolation=InterpolationMode.BICUBIC
        )
        infrared = transform_functional.rotate(
            infrared, angle, interpolation=InterpolationMode.BICUBIC
        )
        extras = [
            transform_functional.rotate(
                item, angle, interpolation=InterpolationMode.BICUBIC
            )
            for item in extras
        ]
    if opt.isTrain and opt.visible_brightness_jitter > 0:
        amount = float(opt.visible_brightness_jitter)
        visible = transform_functional.adjust_brightness(
            visible, random.uniform(max(0.0, 1.0 - amount), 1.0 + amount)
        )
    if opt.isTrain and opt.infrared_brightness_jitter > 0:
        amount = float(opt.infrared_brightness_jitter)
        infrared = transform_functional.adjust_brightness(
            infrared, random.uniform(max(0.0, 1.0 - amount), 1.0 + amount)
        )
    return visible, infrared, extras


def paired_crop_and_flip(
    visible,
    infrared,
    opt,
    legacy_flip=False,
    legacy_random_crop=False,
):
    """Crop/flip two equal-size CHW tensors with exactly shared parameters."""
    if visible.shape[-2:] != infrared.shape[-2:]:
        raise ValueError('paired tensors must have identical spatial dimensions')
    height, width = visible.shape[-2:]
    max_x = max(0, width - opt.fineSize - 1)
    max_y = max(0, height - opt.fineSize - 1)
    if opt.isTrain and (opt.paired_random_crop or legacy_random_crop):
        x_offset = random.randint(0, max_x)
        y_offset = random.randint(0, max_y)
    else:
        x_offset = max_x // 2
        y_offset = max_y // 2

    visible = visible[
        :, y_offset:y_offset + opt.fineSize,
        x_offset:x_offset + opt.fineSize,
    ]
    infrared = infrared[
        :, y_offset:y_offset + opt.fineSize,
        x_offset:x_offset + opt.fineSize,
    ]
    probability = float(opt.paired_flip_probability)
    if legacy_flip and probability == 0.0 and not opt.no_flip:
        probability = 0.5
    if opt.isTrain and not opt.no_flip and random.random() < probability:
        visible = torch.flip(visible, dims=(-1,))
        infrared = torch.flip(infrared, dims=(-1,))
    return visible, infrared
