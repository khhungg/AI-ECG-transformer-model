# ECG Training

ECG classification training using a pretrained Transformer model.

## Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install dependencies and run training
uv sync
uv pip install git+https://github.com/Jwoo5/fairseq-signals.git
uv run python train.py
```

## Data Preparation

Prepare the following files before running:

| File | Description |
|------|-------------|
| `ecg_tensor_modified_excluded.pt` | ECG signal tensor |
| `labels_tensor_new_corrected_excluded.pt` | Labels tensor |
| `ECG_FM/mimic_iv_ecg_physionet_pretrained.pt` | Pretrained backbone weights |

Download the pretrained backbone checkpoint from:
- [`transformer_excluded` weights folder](https://drive.google.com/drive/folders/1bCoGUfdZXNuPVNZYzzkf4My52-jDb52-?usp=sharing)

Example layout:

```text
AI-ECG-transformer-model/
├── ECG_FM/
│   └── mimic_iv_ecg_physionet_pretrained.pt
├── ecg_tensor_modified_excluded.pt
├── labels_tensor_new_corrected_excluded.pt
└── train.py
```

Place `mimic_iv_ecg_physionet_pretrained.pt` under `ECG_FM/` before running training.

## Released Weights

We provide two sets of trained checkpoints:

- [`transformer_excluded`](https://drive.google.com/drive/folders/1bCoGUfdZXNuPVNZYzzkf4My52-jDb52-?usp=sharing)  
  Includes the pretrained backbone checkpoint (`mimic_iv_ecg_physionet_pretrained.pt`) and 5-fold trained model weights (`best_model_fold_1.pth` to `best_model_fold_5.pth`).

- [`10S_FINETUNED`](https://drive.google.com/drive/folders/1J9-lkWgQ73IrB4pww-flhpL5BAuovCQB?usp=sharing)  
  Includes 5-fold checkpoints fine-tuned on 10-second ECG segments (`best_model_fold_1.pth` to `best_model_fold_5.pth`).

## Output

The trained model will be saved in the `saved_models_tavi_transformer_excluded/` directory as
`best_model_fold_1.pth` to `best_model_fold_5.pth`.
