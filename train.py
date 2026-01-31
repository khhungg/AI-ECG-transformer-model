import os, random, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from fairseq_signals.models import build_model_from_checkpoint

ECG_PT   = "ecg_tensor_modified_excluded.pt"
LABEL_PT = "labels_tensor_new_corrected_excluded.pt"
CKPT     = "ECG_FM/mimic_iv_ecg_physionet_pretrained.pt"
OUT_DIR  = "saved_models_tavi_transformer_excluded"
BATCH    = 32
EPOCHS   = 10
FOLDS    = 5
LR       = 1e-5
PATIENCE = 7
SEED     = 42

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def to_nct(x):
    return x if x.ndim==3 and x.shape[1]==12 else x.permute(0,2,1)

def grab_feats(d):
    for k in ("features","encoder_out","x"):
        if k in d: return d[k]
    raise KeyError("Backbone output missing features/encoder_out/x")

class ECGHead(nn.Module):
    def __init__(self, backbone):
        super().__init__(); self.backbone = backbone; self.cls = nn.LazyLinear(1)
    def forward(self, x):
        z = grab_feats(self.backbone(source=x))
        if isinstance(z,(list,tuple)): z = z[0]
        if z.dim()==3 and z.shape[0]!=x.shape[0]: z = z.transpose(0,1)
        z = z.mean(1)
        return self.cls(z).squeeze(-1)

def train_fold(model, tr_ld, va_ld, device):
    crit = nn.BCEWithLogitsLoss()
    opt  = optim.Adam(model.parameters(), lr=LR)
    sch  = ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=3, verbose=False)
    best_auc, no_imp, best = 0.0, 0, None
    for ep in range(1, EPOCHS+1):
        model.train(); tl, ytr, ptr = 0.0, [], []
        for x,y in tr_ld:
            x,y = x.to(device), y.to(device)
            opt.zero_grad(); logit = model(x); loss = crit(logit, y); loss.backward(); opt.step()
            tl += loss.item()*x.size(0); ytr += [y.detach().cpu()]; ptr += [torch.sigmoid(logit).detach().cpu()]
        tr_auc = roc_auc_score(torch.cat(ytr).numpy(), torch.cat(ptr).numpy())

        model.eval(); vl, yva, pva = 0.0, [], []
        with torch.no_grad():
            for x,y in va_ld:
                x,y = x.to(device), y.to(device)
                logit = model(x); loss = crit(logit, y); vl += loss.item()*x.size(0)
                yva += [y.cpu()]; pva += [torch.sigmoid(logit).cpu()]
        va_auc = roc_auc_score(torch.cat(yva).numpy(), torch.cat(pva).numpy())
        sch.step(va_auc)
        print(f"Epoch {ep:02d}  AUC {tr_auc:.4f} | val_AUC {va_auc:.4f}")

        if va_auc > best_auc: best_auc, no_imp, best = va_auc, 0, {k:v.cpu() for k,v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= PATIENCE: break
    if best is not None: model.load_state_dict(best)
    return best_auc, model

def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ecg, y = torch.load(ECG_PT).float(), torch.load(LABEL_PT).float()
    ecg = to_nct(ecg); N = ecg.shape[0]
    os.makedirs(OUT_DIR, exist_ok=True)

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    aucs = []
    for i,(tr,va) in enumerate(skf.split(np.zeros(N), y.numpy().astype(int)), 1):
        tr_ld = DataLoader(TensorDataset(ecg[tr], y[tr]), batch_size=BATCH, shuffle=True)
        va_ld = DataLoader(TensorDataset(ecg[va], y[va]), batch_size=BATCH, shuffle=False)
        backbone = build_model_from_checkpoint(checkpoint_path=CKPT).to(device)
        model = ECGHead(backbone).to(device)
        best_auc, model = train_fold(model, tr_ld, va_ld, device)
        torch.save(model.state_dict(), os.path.join(OUT_DIR, f"best_model_fold_{i}.pth"))
        print(f"Fold {i}: best val AUC {best_auc:.4f}"); aucs.append(best_auc)
    print(f"\nMean val AUC: {np.mean(aucs):.4f}")

if __name__ == "__main__":
    main()
