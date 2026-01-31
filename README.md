# ECG Training

使用預訓練 Transformer 模型進行 ECG 分類訓練。

## 安裝

### 使用 uv（推薦）

```bash
# 安裝 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 安裝依賴
uv sync

# 安裝 fairseq-signals（需要 git）
uv pip install git+https://github.com/Jwoo5/fairseq-signals.git
```

### 使用 pip

```bash
pip install numpy torch scikit-learn fairseq
pip install git+https://github.com/Jwoo5/fairseq-signals.git
```

## 資料準備

需要以下檔案：
- `ecg_tensor_modified_excluded.pt` - ECG 訊號張量
- `labels_tensor_new_corrected_excluded.pt` - 標籤張量
- `ECG_FM/mimic_iv_ecg_physionet_pretrained.pt` - 預訓練模型權重

## 執行訓練

```bash
# 使用 uv
uv run python train.py

# 或直接使用 python
python train.py
```

## 輸出

訓練好的模型會儲存在 `saved_models_tavi_transformer_excluded/` 資料夾中。
