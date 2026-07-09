import os
import ast
import h5py
import torch
import numpy as np
import pandas as pd
import logging
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================
# 0. Logging
# ==========================================
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_filepath = LOG_DIR / datetime.now().strftime("train_abmil_mcdropout_fixed_%Y%m%d_%H%M%S.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. Config
# ==========================================
CSV_PATH     = PROJECT_ROOT / "data" / "interim" / "matched_clinical_slides.csv"
FEATURES_DIR = str(PROJECT_ROOT / "data" / "processed" / "uni_v2" / "TCGA_LUAD")
SAVE_DIR     = PROJECT_ROOT / "results"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH    = str(SAVE_DIR / "abmil_mcdropout_kfold_fixed.pt")

SEED          = 42
N_FOLDS       = 5
BATCH_SIZE    = 16
LR            = 5e-5
WEIGHT_DECAY  = 1e-4
EPOCHS        = 8
DROPOUT       = 0.25     # 학습 + MC 추론 모두 동일하게 사용

RNC_ALPHA     = 0.5
RNC_TEMP      = 0.1

# MC Dropout 추론 설정
N_MC_SAMPLES  = 30       # 추론을 30번 반복해서 불확실성 측정

# ==========================================
# 2. Seed
# ==========================================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ==========================================
# 3. Dataset
# ==========================================
class WSISurvivalDataset(Dataset):
    def __init__(self, df, features_dir):
        self.df = df.reset_index(drop=True)
        self.features_dir = features_dir

        def extract_slide_ids(x):
            try:
                file_list = ast.literal_eval(x)
                if isinstance(file_list, list) and len(file_list) > 0:
                    return [f.replace('.svs', '') for f in file_list]
            except Exception:
                pass
            return [str(x).strip("['").strip("']").split("', '")[0].replace('.svs', '')]

        self.df['slide_ids'] = self.df['file_names'].apply(extract_slide_ids)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        all_features = []
        for slide_id in row['slide_ids']:
            h5_path = os.path.join(self.features_dir, f'{slide_id}.h5')
            try:
                with h5py.File(h5_path, 'r') as f:
                    all_features.append(f['features'][:])
            except Exception as e:
                logger.warning(f'Skipping missing feature: {h5_path} ({e})')

        features_concat = np.concatenate(all_features, axis=0) if all_features else np.zeros((10, 1536))
        return (
            torch.tensor(features_concat, dtype=torch.float32),
            torch.tensor(row['time'],  dtype=torch.float32),
            torch.tensor(row['event'], dtype=torch.float32),
            row['case_id'],
        )

def list_collate_fn(batch):
    features, times, events, case_ids = zip(*batch)
    return list(features), torch.stack(times), torch.stack(events), list(case_ids)

# ==========================================
# 4. Model — dropout=0.25 (학습 + MC 추론 공용)
# ==========================================
class ABMILCoxRNC(nn.Module):
    def __init__(self, input_dim=1536, proj_dim=512, hidden_dim=256, dropout=DROPOUT):
        super().__init__()
        self.feature_norm = nn.LayerNorm(input_dim)
        self.projector = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attn_V    = nn.Sequential(nn.Linear(proj_dim, hidden_dim), nn.Tanh())
        self.attn_U    = nn.Sequential(nn.Linear(proj_dim, hidden_dim), nn.Sigmoid())
        self.attn_w    = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),   # ← MC Dropout의 핵심: 추론 시에도 이 레이어가 켜져 있어야 함
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h):
        h = self.feature_norm(h)
        h = self.projector(h)
        V = self.attn_V(h)
        U = self.attn_U(h)
        A = torch.softmax(self.attn_w(V * U), dim=0)
        z = (A * h).sum(dim=0)             # [512] bag repr (SurvRNC용)
        risk = self.risk_head(z).squeeze() # scalar (Cox용)
        return risk, z

# ==========================================
# 5. Cox Loss
# ==========================================
def cox_loss(risk, times, events):
    idx  = torch.argsort(times, descending=True)
    r, e = risk[idx], events[idx]
    return -((r - torch.logcumsumexp(r, dim=0)) * e).sum() / e.sum().clamp(min=1)

# ==========================================
# 6. SurvRNC Loss (수정됨: 올바른 가중치 및 분모 자기자신 제거)
# ==========================================
class SurvRNCLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temp = temperature

    def forward(self, features, survival_times, events):
        B = features.size(0)
        if B < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        sim_matrix = F.cosine_similarity(
            features.unsqueeze(1), features.unsqueeze(0), dim=-1
        ) / self.temp

        loss = torch.tensor(0.0, device=features.device, requires_grad=True)
        valid_pairs = 0

        for i in range(B):
            if events[i] == 0: continue
            
            # 분모에서 자기 자신(i==i)을 제외하기 위한 마스크 생성
            mask = torch.ones(B, dtype=torch.bool, device=features.device)
            mask[i] = False
            
            denominator = torch.sum(torch.exp(sim_matrix[i, mask]))
            if denominator == 0:
                continue

            for j in range(B):
                if i == j: continue
                if events[j] == 0: continue
                
                # 가중치 수정: 생존 시간 차이가 적을수록(비슷할수록) weight가 1에 가까워지고, 차이가 클수록 0에 가까워짐
                # 배치 의존적인 t_max 대신 절대적인 상수 365.0(1년)을 기준으로 감쇠 스케일링
                time_diff = torch.abs(survival_times[i] - survival_times[j])
                weight = torch.exp(-time_diff / 365.0)  
                
                loss = loss + (
                    -torch.log(
                        torch.exp(sim_matrix[i, j]) / denominator
                    ) * weight
                )
                valid_pairs += 1

        if valid_pairs > 0:
            return loss / valid_pairs
        return torch.tensor(0.0, device=features.device, requires_grad=True)

# ==========================================
# 7. C-index
# ==========================================
def compute_cindex(times, events, risks):
    t, e, r = map(np.asarray, [times, events, risks])
    concordant = tied = permissible = 0
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            if t[i] == t[j] or (e[i] == 0 and e[j] == 0):
                continue
            if t[i] < t[j]:
                if e[i] == 0: continue
                ei, ej = i, j
            else:
                if e[j] == 0: continue
                ei, ej = j, i
            permissible += 1
            if   r[ei] > r[ej]: concordant += 1
            elif r[ei] == r[ej]: tied += 1
    return (concordant + 0.5 * tied) / max(permissible, 1)

# ==========================================
# 8. 일반 추론 (학습 중 val 평가용 — dropout OFF)
# ==========================================
def run_epoch(model, loader, device, survrnc_criterion, alpha, optimizer=None, scaler=None, train=True):
    model.train(train)
    total, all_r, all_t, all_e = 0.0, [], [], []
    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for feat_list, times, events, _ in loader:
            times  = times.to(device)
            events = events.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                risks, bag_reprs = zip(*[model(f.to(device)) for f in feat_list])
                risks     = torch.stack(risks)
                bag_reprs = torch.stack(bag_reprs)
                l_cox = cox_loss(risks, times, events)
                l_rnc = survrnc_criterion(bag_reprs, times, events)
                loss  = l_cox + alpha * l_rnc

            if train and optimizer and scaler:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            total  += loss.item()
            all_r  += risks.detach().float().cpu().tolist()
            all_t  += times.cpu().tolist()
            all_e  += events.cpu().tolist()

    ci = compute_cindex(all_t, all_e, all_r)
    return total / len(loader), ci, all_r, all_t, all_e

# ==========================================
# 9. MC Dropout 추론 (테스트 평가용 — dropout ON, N회 반복)
# ==========================================
def mc_dropout_predict(model, loader, device, n_samples=N_MC_SAMPLES):
    """
    model.train() 상태에서 n_samples번 forward pass 반복.
    환자마다 risk score의 평균(예측)과 표준편차(불확실성)를 반환.
    """
    model.train()   # dropout을 켜기 위해 train 모드 유지

    all_samples = []  # [n_samples, N_patients]
    all_t, all_e, all_ids = [], [], []

    # 첫 패스에서 times/events/ids 수집
    with torch.no_grad():
        for _ in range(n_samples):
            batch_risks = []
            batch_t, batch_e, batch_ids = [], [], []
            for feat_list, times, events, case_ids in loader:
                times  = times.to(device)
                events = events.to(device)
                risks, _ = zip(*[model(f.to(device)) for f in feat_list])
                risks = torch.stack(risks)
                batch_risks += risks.float().cpu().tolist()
                batch_t     += times.cpu().tolist()
                batch_e     += events.cpu().tolist()
                batch_ids   += list(case_ids)
            all_samples.append(batch_risks)
            if not all_t:   # 첫 번째 패스에서만 저장
                all_t   = batch_t
                all_e   = batch_e
                all_ids = batch_ids

    # [n_samples, N_patients] → [N_patients, n_samples]
    samples_arr  = np.array(all_samples).T
    mean_risk    = samples_arr.mean(axis=1)   # 예측값
    std_risk     = samples_arr.std(axis=1)    # 불확실성

    mc_ci = compute_cindex(all_t, all_e, mean_risk.tolist())

    return {
        'mean_risk':  mean_risk.tolist(),
        'std_risk':   std_risk.tolist(),
        'all_samples': samples_arr.tolist(),
        'times':      all_t,
        'events':     all_e,
        'case_ids':   all_ids,
        'mc_ci':      mc_ci,
    }

# ==========================================
# 10. Train one fold
# ==========================================
def train_fold(fold_seed, train_loader, val_loader, device):
    set_seed(fold_seed)
    model     = ABMILCoxRNC().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    survrnc   = SurvRNCLoss(temperature=RNC_TEMP)

    hist = []

    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_ci, *_ = run_epoch(model, train_loader, device, survrnc, RNC_ALPHA, optimizer, scaler, train=True)
        _,       va_ci, *_ = run_epoch(model, val_loader,   device, survrnc, RNC_ALPHA, train=False)
        scheduler.step()
        hist.append(va_ci)
        logger.info(f'  Ep {ep:3d} | loss={tr_loss:.4f} | tr_ci={tr_ci:.4f} | val_ci={va_ci:.4f}')

    return model, va_ci, hist

# ==========================================
# 11. Main
# ==========================================
def train_model():
    logger.info('ABMIL Cox+SurvRNC + MC Dropout — 5-Fold CV')
    logger.info(f'  RNC_ALPHA={RNC_ALPHA} | DROPOUT={DROPOUT} | N_MC_SAMPLES={N_MC_SAMPLES}')
    logger.info(f'  EPOCHS={EPOCHS} | BATCH_SIZE={BATCH_SIZE}')
    set_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    df = pd.read_csv(CSV_PATH)
    df = df[df['time'] > 0].reset_index(drop=True)

    def first_slide_id(x):
        try:
            return ast.literal_eval(x)[0].replace('.svs', '')
        except Exception:
            return str(x).replace('.svs', '')

    avail = {os.path.splitext(f)[0] for f in os.listdir(FEATURES_DIR) if f.endswith('.h5')}
    df['_slide_id'] = df['file_names'].apply(first_slide_id)
    df = df[df['_slide_id'].isin(avail)].reset_index(drop=True)
    logger.info(f'Patients: {len(df)} | Events: {int(df["event"].sum())} | Censored: {int((df["event"]==0).sum())}')

    df['time_bin'] = pd.qcut(df['time'], q=4, labels=False)
    df['stratify_group'] = df['event'].astype(str) + "_" + df['time_bin'].astype(str)

    all_ids = df['case_id'].tolist()
    all_strat = df['stratify_group'].tolist()

    skf          = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (tv_idx, te_idx) in enumerate(skf.split(all_ids, all_strat)):
        logger.info(f'\n══ Fold {fold+1}/{N_FOLDS} ══')

        tv_ids = [all_ids[i] for i in tv_idx]
        tv_strat = [all_strat[i] for i in tv_idx]
        te_ids = [all_ids[i] for i in te_idx]

        tr_ids, va_ids = train_test_split(
            tv_ids, test_size=0.125, random_state=SEED, stratify=tv_strat
        )
        logger.info(f'  Train {len(tr_ids)} | Val {len(va_ids)} | Test {len(te_ids)}')

        tr_df = df[df['case_id'].isin(tr_ids)]
        va_df = df[df['case_id'].isin(va_ids)]
        te_df = df[df['case_id'].isin(te_ids)]

        train_loader = DataLoader(
            WSISurvivalDataset(tr_df, FEATURES_DIR),
            batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=list_collate_fn
        )
        val_loader = DataLoader(
            WSISurvivalDataset(va_df, FEATURES_DIR),
            batch_size=BATCH_SIZE, shuffle=False, collate_fn=list_collate_fn
        )
        test_loader = DataLoader(
            WSISurvivalDataset(te_df, FEATURES_DIR),
            batch_size=BATCH_SIZE, shuffle=False, collate_fn=list_collate_fn
        )

        # 학습
        model, best_val_ci, hist = train_fold(SEED + fold, train_loader, val_loader, device)

        # 일반 추론 (dropout OFF) — 베이스라인 C-index
        survrnc = SurvRNCLoss(temperature=RNC_TEMP)
        _, det_ci, det_r, det_t, det_e = run_epoch(model, test_loader, device, survrnc, RNC_ALPHA, train=False)

        # MC Dropout 추론 (dropout ON, 30회)
        logger.info(f'  MC Dropout 추론 중 (N={N_MC_SAMPLES})...')
        mc_result = mc_dropout_predict(model, test_loader, device, n_samples=N_MC_SAMPLES)
        mc_ci = mc_result['mc_ci']

        # 불확실성 요약
        std_arr = np.array(mc_result['std_risk'])
        logger.info(f'  Fold {fold+1} → val_best={best_val_ci:.4f} | det_ci={det_ci:.4f} | mc_ci={mc_ci:.4f}')
        logger.info(f'  Uncertainty (std) — mean={std_arr.mean():.4f} | max={std_arr.max():.4f} | min={std_arr.min():.4f}')

        fold_results.append({
            'fold':         fold + 1,
            'det_ci':       det_ci,      # 일반 추론 C-index
            'mc_ci':        mc_ci,       # MC Dropout C-index
            'mc_result':    mc_result,   # 환자별 mean/std/samples
            'det_risks':    det_r,
            'det_times':    det_t,
            'det_events':   det_e,
            'val_hist':     hist,
            'model_state':  model.state_dict(),
        })

    det_cis = [r['det_ci'] for r in fold_results]
    mc_cis  = [r['mc_ci']  for r in fold_results]

    logger.info(f'\nDeterministic C-index:  {np.mean(det_cis):.4f} ± {np.std(det_cis):.4f}')
    logger.info(f'MC Dropout C-index:     {np.mean(mc_cis):.4f} ± {np.std(mc_cis):.4f}')

    torch.save({
        'model':        'ABMIL-Cox-RNC-MCDropout',
        'config': {
            'rnc_alpha':       RNC_ALPHA,
            'rnc_temp':        RNC_TEMP,
            'dropout':         DROPOUT,
            'n_mc_samples':    N_MC_SAMPLES,
            'weight_normalized': True,
        },
        'n_folds':      N_FOLDS,
        'det_mean_ci':  float(np.mean(det_cis)),
        'det_std_ci':   float(np.std(det_cis)),
        'mc_mean_ci':   float(np.mean(mc_cis)),
        'mc_std_ci':    float(np.std(mc_cis)),
        'fold_results': fold_results,
    }, SAVE_PATH)
    logger.info(f'Saved → {SAVE_PATH}')

    print('\n' + '=' * 56)
    print(f"  ABMIL + Cox + SurvRNC + MC Dropout (N={N_MC_SAMPLES})")
    print('=' * 56)
    print(f"  {'Fold':<6}  {'Det C-index':>11}  {'MC C-index':>10}")
    print('-' * 56)
    for r in fold_results:
        print(f"  {r['fold']:<6}  {r['det_ci']:>11.4f}  {r['mc_ci']:>10.4f}")
    print('-' * 56)
    print(f"  {'Mean':<6}  {np.mean(det_cis):>11.4f}  {np.mean(mc_cis):>10.4f}")
    print(f"  {'Std':<6}  {np.std(det_cis):>11.4f}  {np.std(mc_cis):>10.4f}")
    print('=' * 56)
    print(f"\n  Det vs MC 차이: {np.mean(mc_cis) - np.mean(det_cis):+.4f}")
    print('=' * 56)


if __name__ == '__main__':
    train_model()
