# DFSMamba

> Detail-Guided and Direction-Adaptive State-Space Modeling for Structure-Preserving Visible-to-Infrared Image Translation

DFSMamba is a PyTorch implementation for paired visible-to-infrared image translation.

## Repository Structure

```text
DFSmamba/
├── configs/
│   └── sssm.yaml                  # SSSM and paired-augmentation settings
├── data/                           # Dataset loaders and paired augmentation
├── models/
│   ├── dfsmamba/
│   │   ├── architecture.py        # DFSMamba generator and discriminator
│   │   ├── configs.py             # Network configurations
│   │   ├── dgm.py                 # Detail-Guided Module
│   │   ├── ms_scsa.py             # MS-SCSA and DPM
│   │   ├── vision_mamba.py        # Direction-Adaptive SS2D
│   │   ├── vision_transformer.py  # Transformer discriminator branch
│   │   └── utils.py               # Shared layers and tensor utilities
│   ├── dfsmamba_model.py          # Training logic and optimization objectives
│   ├── models.py                  # Model factory
│   └── sssm.py                    # Frozen encoder and SSSM loss
├── options/                        # Training and runtime options
├── util/                           # Logging and visualization utilities
├── train.py                        # Training entry point
├── train.sh                        # Example VEDAI training script
├── requirements.txt
└── RS.pdf                          # Paper manuscript
```

## Requirements

The recommended environment is:

- Linux
- NVIDIA GPU
- CUDA 11.8
- PyTorch 2.1.1
- torchvision 0.16.1
- mamba-ssm 1.2.0.post1
- causal-conv1d 1.1.1

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`mamba-ssm` and `causal-conv1d` must be compatible with the installed CUDA and PyTorch versions. The current network implementation uses `cuda:0` in several layers, so training on the first GPU is recommended.

## Dataset Preparation

The paper uses the FMB, VEDAI, and AVIID-1 paired visible-infrared datasets. Download them from their official sources and prepare the splits as follows:

| Dataset | Training pairs | Test pairs | Input size |
| --- | ---: | ---: | ---: |
| FMB | 1,220 | 280 | 512 × 512 |
| VEDAI | 996 | 124 | 512 × 512 |
| AVIID-1 | 794 | 199 | 512 × 512 |

Example directory layout and filenames:

```text
datasets/
├── FMB/
│   ├── train/
│   │   ├── 000001_vis.jpg
│   │   └── 000001_ir.jpg
│   └── test/
│       ├── 000002_vis.jpg
│       └── 000002_ir.jpg
├── VEDAI_512/
│   ├── train/
│   │   ├── 000001_vis.png
│   │   └── 000001_ir.png
│   └── test/
│       ├── 000002_vis.png
│       └── 000002_ir.png
└── AVIID-1/
    ├── train/
    │   ├── 000001_vis.png
    │   └── 000001_ir.png
    └── test/
        ├── 000002_vis.png
        └── 000002_ir.png
```

The current training loader reads data from `<dataroot>/train`. Registered visible and infrared images receive the same geometric transformations. By default, random cropping, flipping, rotation, and brightness jitter are disabled in `configs/sssm.yaml`.

## SSSM Encoder Checkpoint

Full DFSMamba training requires an infrared feature encoder pretrained through real-infrared reconstruction:

```text
checkpoints/sssm_encoder.pth
```

The checkpoint must contain the patch embedding and four-stage VMEncoder. The loader accepts a direct state dictionary or a checkpoint containing one of these entries:

- `feature_encoder`
- `sssm_encoder`
- `encoder_state_dict`
- `state_dict`
- `model`
- separate `patch_embed` and `encoder` mappings

The paper pretrains the infrared autoencoder for 50 epochs using Adam with a learning rate of `1e-4`, `beta1=0.5`, and `beta2=0.999`. The L1 and SSIM reconstruction-loss weights are both set to 100.

The current repository loads and freezes a pretrained SSSM encoder but does not include the autoencoder pretraining entry point or pretrained weights. Provide a compatible checkpoint before enabling SSSM.

## Training

### VEDAI Example

`train.sh` is configured for VEDAI. Set the SSSM checkpoint path before running it:

```bash
SSSM_CHECKPOINT=/absolute/path/to/sssm_encoder.pth bash train.sh
```

### AVIID-1 Example

```bash
python train.py \
  --model dfsmamba \
  --dataset_mode AVIID_1 \
  --dataroot ./datasets/AVIID-1 \
  --name DFSMamba_AVIID1 \
  --which_model_netG DFSMambaGenerator \
  --which_model_netD DFSMambaDiscriminator \
  --which_direction AtoB \
  --input_nc 3 \
  --output_nc 1 \
  --loadSize 512 \
  --fineSize 512 \
  --batchSize 1 \
  --gpu_ids 0 \
  --lr 0.0002 \
  --lambda_A 100 \
  --no_lsgan \
  --sssm_weight 1 \
  --sssm_checkpoint ./checkpoints/sssm_encoder.pth
```

### Training Without SSSM

```bash
python train.py \
  --model dfsmamba \
  --dataset_mode AVIID_1 \
  --dataroot ./datasets/AVIID-1 \
  --name DFSMamba_AVIID1_no_SSSM \
  --gpu_ids 0 \
  --no_lsgan \
  --sssm_weight 0
```

Training runs for 200 epochs by default. The learning rate remains constant for the first 100 epochs and linearly decays to zero over the final 100 epochs. The initial generator and discriminator learning rates are `2e-4` and `2e-6`, respectively.

Checkpoints, resolved options, training history, and visualization outputs are saved under:

```text
checkpoints/<experiment_name>/
```
