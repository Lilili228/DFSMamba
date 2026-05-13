# DFSMamba

DFSMamba is a visible-to-infrared image translation framework for paired remote-sensing scenes. It builds a Mamba-based generator with detail-guided texture compensation, directional state-space modeling, and latent structural consistency supervision.

This repository contains the implementation used for training, testing, and evaluating DFSMamba on visible-infrared datasets such as VEDAI, FMB, and AVIID-1.

## Highlights

- **DFSMamba generator**: an encoder-decoder image translation network based on visual state-space blocks.
- **DGM**: a Detail-Guided Module using DWT-based frequency decomposition, multi-scale feature extraction, and feature modulation.
- **Directional SS2D**: a multi-directional state-space scanning strategy for improving spatial continuity.
- **SSSM loss**: a State-Space Structural Similarity Matching loss for latent structural alignment.
- **Dual-encoder discriminator**: a Transformer-Mamba discriminator with difference-and-product feature interaction.

## Project Structure

```text
DFSMamba/
  data/                         Dataset loaders
  models/
    dfsmamba_model.py           Training wrapper for DFSMamba
    dfsmamba/
      dfsmamba.py               Generator and discriminator definitions
      DGM.py                    Detail-Guided Module
      vision_mamba.py           Mamba encoder/decoder and SS2D blocks
      vision_transformer.py     Transformer branch for discriminator
      hscam.py                  Skip-feature attention and DPM modules
      configs.py                Model configuration
      utils.py                  Patch embedding and helper layers
  options/                      Training and testing options
  util/                         Logging, visualization, and helper utilities
  train.py                      Training entry
  test.py                       Testing entry
  evaluate.py                   Metric evaluation entry
  train.sh                      Default training script
  test.sh                       Default testing script
  evaluate.sh                   Default evaluation script
```

## Environment

The code was developed with:

- Python 3.x
- PyTorch 2.1.1
- torchvision 0.16.1
- CUDA 11.8
- `mamba-ssm`
- `causal-conv1d`
- `pytorch-msssim`
- `lpips`
- `visdom`

Install dependencies:

```bash
pip install -r requirements.txt
```

If Mamba-related packages fail to install, install versions compatible with your CUDA and PyTorch environment.

## Dataset Preparation

Prepare paired visible and infrared images using the dataset structure expected by the selected dataset loader. For the default VEDAI setting, place data under:

```text
DFSMamba/
  datasets/
    VEDAI_512/
      trainA/
      trainB/
      testA/
      testB/
```

The default scripts use:

```bash
dataroot="./datasets/VEDAI_512"
dataset_mode="VEDAI"
```

Update `train.sh`, `test.sh`, and `evaluate.sh` if your dataset path or loader name is different.

## Training

Run:

```bash
sh train.sh
```

The default training script uses:

```bash
model="dfsmamba"
name="DFSMamba_VEDAI_512"
which_model_netG="DFSMambaGenerator"
which_model_netD="DFSMambaDiscriminator"
```

Checkpoints and logs are saved to:

```text
checkpoints/DFSMamba_VEDAI_512/
```

## Testing

Run:

```bash
sh test.sh
```

Set `which_epoch` in `test.sh` to the checkpoint you want to load, for example:

```bash
which_epoch="best_196"
```

## Evaluation

Run:

```bash
sh evaluate.sh
```

The evaluation script reports metrics used in the paper, including SSIM, MS-SSIM, PSNR, L1 distance, and LPIPS.

## Implementation Notes

- The main model entry is `models/dfsmamba_model.py`.
- The generator and discriminator are defined in `models/dfsmamba/dfsmamba.py`.
- DGM is implemented in `models/dfsmamba/DGM.py`.
- The directional SS2D implementation is in `models/dfsmamba/vision_mamba.py`.
- SSSM loss is implemented in `models/dfsmamba_model.py`.

## Acknowledgement

This codebase was developed by modifying an existing visible-to-infrared GAN/Mamba implementation and adapting it into DFSMamba. We thank the authors of the related open-source implementations for their contributions to the community.

## Citation

If you use this code, please cite the corresponding DFSMamba paper after publication.
