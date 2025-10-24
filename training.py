#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from joblib import dump

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.utils import set_random_seed

from env_anomali import AnomalyEnv

# ============================================================
# 17D FEATURE ORDER — harus konsisten dengan runtime/main.py
# ============================================================
FEATURE_ORDER: List[str] = [
    "flow_duration",
    "fwd_pkts_tot", "bwd_pkts_tot",
    "fwd_data_pkts_tot", "bwd_data_pkts_tot",
    "fwd_pkts_per_sec", "bwd_pkts_per_sec", "flow_pkts_per_sec",
    "down_up_ratio",
    "proto_tcp", "proto_udp", "proto_icmp",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "dst_port_norm",
]

# -----------------------------
# Utilities
# -----------------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    set_random_seed(seed)


def robust_read_csv(path: str) -> pd.DataFrame:
    """
    Baca CSV 'jorok' (delimiter campur, baris rusak).
    - infer delimiter
    - skip baris rusak
    - cegah 'NA' otomatis jadi NaN
    """
    return pd.read_csv(
        path,
        engine="python",
        sep=None,                 # auto-detect delimiter
        on_bad_lines="skip",      # lewati baris rusak
        quotechar='"',
        escapechar="\\",
        keep_default_na=False,
    )


def ensure_train_val_paths() -> Tuple[str, str]:
    """
    Auto-split 80/20 dari satu dataset lokal: datasets/dummy_dataset.csv (wajib ada).
    """
    src_path = "datasets/dummy_dataset.csv"
    if not os.path.exists(src_path):
        raise FileNotFoundError("File 'datasets/dummy_dataset.csv' tidak ditemukan di direktori kerja saat ini.")

    print(f"[info] Auto-split dari: {src_path}")
    df = robust_read_csv(src_path)

    # Pastikan kolom minimal tersedia (isi default jika hilang)
    req = [
        "timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "proto",
        "fwd_pkts_tot", "bwd_pkts_tot", "fwd_data_pkts_tot", "bwd_data_pkts_tot",
        "signature", "action"
    ]
    for c in req:
        if c not in df.columns:
            df[c] = 0 if c in {
                "src_port","dst_port","fwd_pkts_tot","bwd_pkts_tot","fwd_data_pkts_tot","bwd_data_pkts_tot","action"
            } else ""

    # Parse timestamp bila ada
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    # Shuffle + split 80/20
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = int(len(df) * 0.8)
    out_train, out_val = "dataset_train.csv", "dataset_val.csv"
    df.iloc[:n].to_csv(out_train, index=False)
    df.iloc[n:].to_csv(out_val, index=False)

    print(f"[info] Split: {len(df)} rows -> {len(df.iloc[:n])} train / {len(df.iloc[n:])} val")
    return out_train, out_val


def build_reward_fn(tp=2.0, tn=0.5, fp=-3.5, fn=-5.0):
    """Skema reward yang mengontrol trade-off FP/FN."""
    def _rf(action: int, true_label: int) -> float:
        if action == 1 and true_label == 1:  # TP
            return float(tp)
        if action == 0 and true_label == 0:  # TN
            return float(tn)
        if action == 1 and true_label == 0:  # FP
            return float(fp)
        if action == 0 and true_label == 1:  # FN
            return float(fn)
        return -1.0
    return _rf


def make_env(csv_path: str, verbose: bool = False, reward_fn=None,
             scaler_path: Optional[str] = None, load_scaler: bool = True):
    """
    Thunk untuk DummyVecEnv.
    Semua worker memuat skaler yang SAMA (load_scaler=True).
    """
    return lambda: AnomalyEnv(
        dataset_path=csv_path,
        online_mode=False,
        verbose=verbose,
        reward_fn=reward_fn,
        scaler_path=scaler_path,
        load_scaler=load_scaler,
        save_scaler=False,  # jangan fit+simpan di tiap worker
    )

# ============================================================
#                  PPO TRAINING (binary)
# ============================================================

def train_ppo(
    train_csv: str,
    eval_csv: Optional[str] = None,
    total_timesteps: int = 600_000,
    out_dir: str = "models",
    model_name_latest: str = "ppo_anomali_latest.zip",
    seed: int = 42,
    policy: str = "MlpPolicy",
    lr: float = 3e-4,
    n_envs: int = 4,
    n_steps: int = 4096,
    batch_size: int = 256,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.03,
    eval_freq: int = 10_000,
    eval_episodes: int = 5,
    save_freq: int = 20_000,
    tp: float = 2.0,
    tn: float = 0.5,
    fp: float = -3.5,
    fn: float = -5.0,
    device: str = "auto",
    scaler_path: str = "models/minmax.pkl",
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    set_all_seeds(seed)

    # Tentukan device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reward function
    reward_fn = build_reward_fn(tp=tp, tn=tn, fp=fp, fn=fn)

    # ---------------------------------------------------------
    # 0) Fit & simpan MinMaxScaler SEKALI (konsisten lintas worker)
    # ---------------------------------------------------------
    warm_env = AnomalyEnv(
        dataset_path=train_csv,
        verbose=True,
        scaler_path=scaler_path,
        load_scaler=False,   # FIT baru dari train_csv
        save_scaler=True     # simpan skaler
    )
    del warm_env  # skaler sudah tersimpan ke scaler_path

    # ---------------------------------------------------------
    # 1) Build train envs (semua memuat skaler yang sama)
    # ---------------------------------------------------------
    train_env = DummyVecEnv([
        make_env(train_csv, verbose=False, reward_fn=reward_fn,
                 scaler_path=scaler_path, load_scaler=True)
        for _ in range(n_envs)
    ])

    # 2) eval env
    if eval_csv is None:
        eval_env = DummyVecEnv([make_env(train_csv, verbose=False, reward_fn=reward_fn,
                                         scaler_path=scaler_path, load_scaler=True)])
    else:
        eval_env = DummyVecEnv([make_env(eval_csv, verbose=False, reward_fn=reward_fn,
                                         scaler_path=scaler_path, load_scaler=True)])

    # 3) Callbacks
    ckpt_cb = CheckpointCallback(save_freq=save_freq, save_path=out_dir, name_prefix="ppo_ckpt")
    earlystop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=5, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=out_dir,
        log_path="logs",
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        callback_after_eval=earlystop_cb,
    )

    # 4) PPO
    model = PPO(
        policy=policy,
        env=train_env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        target_kl=target_kl,
        verbose=1,
        seed=seed,
        device=device,
    )

    # 5) Train
    model.learn(total_timesteps=total_timesteps, callback=[ckpt_cb, eval_cb])

    # 6) Save latest model
    latest_path = os.path.join(out_dir, model_name_latest)
    model.save(latest_path)

    print("✅ Training selesai.")
    print(f"   Best/ckpts dir : {out_dir}")
    print(f"   Latest model   : {latest_path}")
    print(f"   Train CSV      : {train_csv}")
    if eval_csv:
        print(f"   Eval  CSV      : {eval_csv}")
    print(f"   Scaler (MinMax): {scaler_path}")

    return latest_path

# ============================================================
#           MULTI-CLASS TRAINING (scan/brute/dos)
# ============================================================

def _normalize_proto_column(df: pd.DataFrame) -> pd.DataFrame:
    # Perbaikan: jangan gunakan `or` langsung pada Series; ambil kolom proto jika ada
    ser = df["proto"] if "proto" in df.columns else pd.Series(["unknown"] * len(df))
    proto = ser.astype(str).str.lower()
    df = df.copy()
    df["proto_tcp"] = (proto == "tcp").astype(np.float32)
    df["proto_udp"] = (proto == "udp").astype(np.float32)
    df["proto_icmp"] = proto.isin(["icmp", "ipv4-icmp", "ipv6-icmp", "icmpv6"]).astype(np.float32)
    return df

def _encode_time_and_dstport(df: pd.DataFrame) -> pd.DataFrame:
    # timestamp → cyclical
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    ts = ts.fillna(method="ffill").fillna(method="bfill")
    hour = ts.dt.hour % 24
    wday = ts.dt.weekday % 7
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    df["weekday_sin"] = np.sin(2 * np.pi * wday / 7.0).astype(np.float32)
    df["weekday_cos"] = np.cos(2 * np.pi * wday / 7.0).astype(np.float32)

    # dst_port_norm
    dport = pd.to_numeric(df.get("dst_port", 0), errors="coerce").fillna(0.0)
    df["dst_port_norm"] = (dport / 65535.0).astype(np.float32)
    return df

def _standardize_attack_reason(reason: str) -> Optional[str]:
    if not isinstance(reason, str) or len(reason) == 0:
        return None
    r = reason.lower().replace(".", "_").replace("-", "_")
    if "scan" in r:
        return "scan"
    if "brute" in r:
        return "bruteforce"
    if "dos" in r or "flood" in r:
        return "dos"
    return None  # kelas lain/unknown → dibuang

def _build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # Pastikan kolom numerik utama ada, isi default 0 bila hilang
    base_numeric = [
        "flow_duration",
        "fwd_pkts_tot", "bwd_pkts_tot",
        "fwd_data_pkts_tot", "bwd_data_pkts_tot",
        "fwd_pkts_per_sec", "bwd_pkts_per_sec", "flow_pkts_per_sec",
        "down_up_ratio",
    ]
    for c in base_numeric:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)

    df = _normalize_proto_column(df)
    df = _encode_time_and_dstport(df)

    # Pastikan semua fitur ada
    for f in FEATURE_ORDER:
        if f not in df.columns:
            df[f] = 0.0

    return df[FEATURE_ORDER].astype(np.float32)

def train_attack_classifier(
    train_csv: str,
    out_path: str = "models/attack_classifier.pkl",
    report_path: str = "logs/attack_classifier_report.txt",
    model_type: str = "rf",
    seed: int = 42,
) -> str:
    """
    Latih classifier supervised untuk tipe serangan (scan/bruteforce/dos).
    - Menggunakan hanya baris y_attack==1.
    - Label diambil dari y_reason → dinormalisasi ke {scan, bruteforce, dos}.
    - Model default RandomForest; alternatif MLP.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    if model_type == "rf":
        from sklearn.ensemble import RandomForestClassifier
    elif model_type == "mlp":
        from sklearn.neural_network import MLPClassifier
    else:
        raise ValueError("model_type harus salah satu dari {'rf','mlp'}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    set_all_seeds(seed)

    print(f"[multi] load: {train_csv}")
    df = robust_read_csv(train_csv)

    # Filter hanya attack
    if "y_attack" not in df.columns:
        raise ValueError("Dataset tidak punya kolom 'y_attack' untuk filter attack==1.")
    df = df[df["y_attack"].astype(str).isin(["1", 1])].copy()
    if df.empty:
        raise ValueError("Tidak ada baris attack (y_attack==1).")

    # Normalisasi label tipe serangan dari y_reason
    df["attack_type"] = df["y_reason"].map(_standardize_attack_reason)
    df = df.dropna(subset=["attack_type"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("Tidak ada baris attack dengan y_reason yang bisa dipetakan ke {scan, bruteforce, dos}.")

    # Build fitur 17D (konsisten runtime)
    X = _build_feature_matrix(df)
    y = df["attack_type"].astype(str)

    # Split train/val (robust utk dataset kecil / single-class)
    class_counts = y.value_counts()
    use_stratify = (len(class_counts) >= 2 and class_counts.min() >= 2)

    # Atur test_size minimal masuk akal untuk dataset kecil
    test_size = 0.2
    if len(y) < 5:
        test_size = 0.4  # biar tetap ada val sample
    stratify_arg = y if use_stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify_arg
    )

    # Model
    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced",
            random_state=seed,
        )
    else:  # mlp
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=100,
            random_state=seed,
        )

    print(f"[multi] training model={model.__class__.__name__} on {len(X_train)} rows ...")
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_val)
    rep = classification_report(y_val, y_pred, digits=3)
    print("\n[multi] Evaluation:\n", rep)
    with open(report_path, "w") as f:
        f.write(rep)

    # Simpan artifact
    artifact: Dict[str, object] = {
        "model": model,
        "feature_order": FEATURE_ORDER,
        "classes_": sorted(y.unique().tolist()),
        "model_type": model_type,
        "seed": seed,
    }
    dump(artifact, out_path)
    print(f"[multi] saved: {out_path}")
    print(f"[multi] report: {report_path}")
    return out_path

# -----------------------------
# Main training entry
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Train PPO (binary) and optional Multi-class attack classifier")
    ap.add_argument("--outdir", type=str, default="models", help="Output dir for PPO models/artifacts")
    ap.add_argument("--timesteps", type=int, default=1_000_000, help="Total timesteps to train (PPO)")
    ap.add_argument("--n-envs", type=int, default=4, help="Number of parallel envs")
    ap.add_argument("--policy", type=str, default="MlpPolicy", help="Policy class name")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    ap.add_argument("--n-steps", type=int, default=4096, help="Rollout steps per env")
    ap.add_argument("--batch-size", type=int, default=256, help="Batch size")
    ap.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    ap.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    ap.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    ap.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coef")
    ap.add_argument("--vf-coef", type=float, default=0.5, help="Value loss coef")
    ap.add_argument("--max-grad-norm", type=float, default=0.5, help="Max grad norm")
    ap.add_argument("--target-kl", type=float, default=0.03, help="Target KL for early stopping in PPO")
    ap.add_argument("--eval-freq", type=int, default=10_000, help="Eval frequency (steps)")
    ap.add_argument("--eval-episodes", type=int, default=5, help="Episodes per evaluation")
    ap.add_argument("--save-freq", type=int, default=20_000, help="Checkpoint save frequency (steps)")
    ap.add_argument("--fp", type=float, default=-3.5, help="Penalty for False Positive")
    ap.add_argument("--fn", type=float, default=-5.0, help="Penalty for False Negative")
    ap.add_argument("--tp", type=float, default=2.0, help="Reward for True Positive")
    ap.add_argument("--tn", type=float, default=0.5, help="Reward for True Negative")
    ap.add_argument("--device", type=str, default="auto", help="cpu | cuda | auto")
    ap.add_argument("--scaler-path", type=str, default="models/minmax.pkl", help="Path MinMaxScaler .pkl")

    # ---- opsi multiclass ----
    ap.add_argument("--train-multiclass", action="store_true",
                    help="Jalankan juga training multi-class attack classifier (scan/bruteforce/dos)")
    ap.add_argument("--multi-model", type=str, default="rf", choices=["rf", "mlp"],
                    help="Tipe model supervised untuk klasifikasi jenis serangan")
    ap.add_argument("--multi-out", type=str, default="models/attack_classifier.pkl",
                    help="Path output .pkl untuk classifier multi-class")
    ap.add_argument("--multi-report", type=str, default="logs/attack_classifier_report.txt",
                    help="Path laporan evaluasi multi-class")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    set_all_seeds(args.seed)

    # ---------- Auto-split dari datasets/dummy_dataset.csv ----------
    train_csv, val_csv = ensure_train_val_paths()

    # ---------- Reward function ----------
    reward_fn = build_reward_fn(tp=args.tp, tn=args.tn, fp=args.fp, fn=args.fn)

    # ---------- Fit & simpan scaler sekali ----------
    warm_env = AnomalyEnv(
        dataset_path=train_csv,
        verbose=True,
        scaler_path=args.scaler_path,
        load_scaler=False,
        save_scaler=True
    )
    del warm_env

    # ---------- Build VecEnvs (tanpa VecNormalize) ----------
    train_env = DummyVecEnv([
        make_env(train_csv, verbose=False, reward_fn=reward_fn,
                 scaler_path=args.scaler_path, load_scaler=True)
        for _ in range(args.n_envs)
    ])

    eval_env = DummyVecEnv([
        make_env(val_csv, verbose=False, reward_fn=reward_fn,
                 scaler_path=args.scaler_path, load_scaler=True)
    ])

    # ---------- Callbacks ----------
    ckpt_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.outdir,
        name_prefix="ppo_ckpt"
    )
    earlystop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=5,
        verbose=1
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.outdir,
        log_path="logs",
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        callback_after_eval=earlystop_cb
    )

    # ---------- PPO ----------
    model = PPO(
        policy=args.policy,
        env=train_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        verbose=1,
        seed=args.seed,
        device=(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")),
    )

    # ---------- Train ----------
    model.learn(total_timesteps=args.timesteps, callback=[ckpt_cb, eval_cb])

    # ---------- Save latest model ----------
    latest_path = os.path.join(args.outdir, "ppo_anomali_latest.zip")
    model.save(latest_path)

    print("✅ Training PPO (binary) selesai.")
    print(f"   Best/ckpts dir : {args.outdir}")
    print(f"   Latest model   : {latest_path}")
    print(f"   Train CSV      : {train_csv}")
    print(f"   Val CSV        : {val_csv}")
    print(f"   Scaler (MinMax): {args.scaler_path}")

    # ---------- (Opsional) Multi-class ----------
    if args.train_multiclass:
        print("\n================ MULTI-CLASS TRAINING ================\n")
        try:
            # Pakai dummy dataset jika ada; fallback ke train_csv
            multi_csv = "datasets/dummy_dataset.csv" if os.path.exists("datasets/dummy_dataset.csv") else train_csv
            print(f"[multi] using dataset: {multi_csv}")
            train_attack_classifier(
                train_csv=multi_csv,
                out_path=args.multi_out,
                report_path=args.multi_report,
                model_type=args.multi_model,
                seed=args.seed,
            )
        except Exception as e:
            print(f"[multi] ❌ Gagal training multi-class: {e}")

if __name__ == "__main__":
    main()
