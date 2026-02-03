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
| `ECG_FM/mimic_iv_ecg_physionet_pretrained.pt` | Pretrained model weights |

## Output

The trained model will be saved in the `saved_models_tavi_transformer_excluded/` directory.
