# ECG Training

使用預訓練 Transformer 模型進行 ECG 分類訓練。

## 快速開始

```bash
# 安裝 uv（如果還沒安裝）
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 安裝依賴並執行訓練
uv sync
uv pip install git+https://github.com/Jwoo5/fairseq-signals.git
uv run python train.py
```

## 資料準備

執行前需準備以下檔案：

| 檔案 | 說明 |
|------|------|
| `ecg_tensor_modified_excluded.pt` | ECG 訊號張量 |
| `labels_tensor_new_corrected_excluded.pt` | 標籤張量 |
| `ECG_FM/mimic_iv_ecg_physionet_pretrained.pt` | 預訓練模型權重 |

## 輸出

訓練好的模型會儲存在 `saved_models_tavi_transformer_excluded/` 資料夾中。
