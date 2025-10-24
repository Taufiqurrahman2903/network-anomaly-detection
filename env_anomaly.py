# env_anomali.py
# -*- coding: utf-8 -*-

import os
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
from typing import Optional, Tuple, Callable, List, Dict

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

# =========================
# Skema fitur (HARUS sama dengan runtime/main.py)
# =========================
FEATURE_ORDER = [
    "src_port", "dest_port",
    "proto_tcp", "proto_udp",
    "flow_duration",
    "fwd_pkts_tot", "bwd_pkts_tot",
    "fwd_data_pkts_tot", "bwd_data_pkts_tot",
    "fwd_pkts_per_sec", "bwd_pkts_per_sec", "flow_pkts_per_sec",
    "down_up_ratio",
    "hour", "weekday",
    "sig_len", "sig_hash",
]


def calculate_reward(action: int, true_label: int) -> float:
    """
    Skema reward konservatif untuk menekan False Positive (FP).
    0 = allow/normal, 1 = block/anomaly
    """
    if action == 1 and true_label == 1:
        return +2.0   # TP
    if action == 0 and true_label == 0:
        return +0.5   # TN
    if action == 1 and true_label == 0:
        return -6.0   # FP (blokir benign)
    if action == 0 and true_label == 1:
        return -2.5   # FN
    return -1.0


class AnomalyEnv(gym.Env):
    """
    Environment serbaguna untuk DRL Anomaly Detection.

    Dua mode sumber data:
      1) dataset_path (CSV)  -> mode BINER (0=normal, 1=anomali)
      2) log_path (eve.json) -> mode KATEGORI (0=allow, 1..N=block-by-category)

    Antarmuka terjamin (mode biner):
      - self.features: np.ndarray (num_samples, 17) skala [0,1]
      - self.labels  : np.ndarray (num_samples,) int64 {0,1}
      - self.current_step: int
      - observation_space: Box(17,)
      - action_space: Discrete(2)

    Catatan penting:
      - Agar inference konsisten, gunakan opsi save/load scaler_path.
        Saat training, fit scaler di env lalu simpan scaler (save_scaler=True).
        Saat runtime/inference, load scaler yang sama (load_scaler=True).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        log_path: Optional[str] = None,
        online_mode: bool = False,
        verbose: bool = False,
        reward_fn: Optional[Callable[[int, int], float]] = None,

        # ==== tambahan konfigurasi ====
        label_priority: Optional[List[str]] = None,
        shuffle_on_reset: bool = True,

        # ==== konsistensi scaler (training vs runtime) ====
        scaler_path: Optional[str] = None,   # path file .pkl untuk MinMaxScaler
        load_scaler: bool = False,           # True: load scaler dari scaler_path (runtime)
        save_scaler: bool = False,           # True: simpan scaler yang di-fit (training)
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.log_path = log_path
        self.online_mode = online_mode
        self.verbose = verbose
        self.reward_fn = reward_fn or calculate_reward
        self.shuffle_on_reset = shuffle_on_reset

        self.scaler_path = scaler_path
        self._want_load_scaler = load_scaler
        self._want_save_scaler = save_scaler

        # prioritas default pelabelan
        self.label_priority = label_priority or [
            "label", "y_attack", "auto_label", "status", "action", "threat_score"
        ]

        # Mode biner default (untuk evaluasi/training)
        if self.dataset_path:
            self._load_dataset_binary()
            self.mode = "binary"
            n_features = self.features.shape[1]
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(n_features,), dtype=np.float32)
            self.action_space = spaces.Discrete(2)  # 0 normal, 1 anomaly/block
        else:
            # Fallback ke mode kategori dari log Suricata
            self._load_from_log_category()
            self.mode = "category"
            n_features = self.features.shape[1]
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(n_features,), dtype=np.float32)
            self.action_space = spaces.Discrete(len(self.category_list) + 1)  # 0=allow, 1..N block category

        self.current_step = 0

    # =========================
    # Util umum
    # =========================
    @staticmethod
    def feature_order() -> List[str]:
        """Kembalikan urutan fitur yang dipakai env."""
        return list(FEATURE_ORDER)

    def _sanitize(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        X[X < 0.0] = 0.0
        X[X > 1.0] = 1.0
        return X

    # === Tambahan untuk online training ===
    @property
    def n_features(self) -> int:
        return int(self.features.shape[1]) if hasattr(self, "features") else 0

    def reload_now(self) -> bool:
        """Reload dataset biner saat training berjalan."""
        if getattr(self, "mode", None) != "binary":
            return False
        old_n = self.n_features
        self._load_dataset_binary()
        new_n = self.n_features
        self.current_step = 0
        return (old_n == 0) or (new_n == old_n)

    def dataset_info(self) -> Dict[str, int | str]:
        return {
            "mode": getattr(self, "mode", "unknown"),
            "n_samples": int(len(self.features)) if hasattr(self, "features") else 0,
            "n_features": self.n_features,
        }

    # ===== Scaler I/O (konsistensi training ↔ runtime) =====
    def _fit_or_load_scaler(self, X: np.ndarray) -> MinMaxScaler:
        """
        - Jika load_scaler=True dan scaler_path ada → load scaler dan transform X.
        - Selain itu, fit scaler baru dari X.
          Jika save_scaler=True dan scaler_path diset, simpan scaler.
        """
        if self._want_load_scaler and self.scaler_path and os.path.exists(self.scaler_path):
            try:
                with open(self.scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                if not isinstance(scaler, MinMaxScaler):
                    raise TypeError("Objek scaler di file bukan MinMaxScaler.")
                return scaler
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Gagal load scaler: {e}. Re-fit scaler baru dari data.")
                # continue to fit a new scaler

        scaler = MinMaxScaler()
        scaler.fit(X)

        if self._want_save_scaler and self.scaler_path:
            try:
                os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
            except Exception:
                pass
            try:
                with open(self.scaler_path, "wb") as f:
                    pickle.dump(scaler, f)
                if self.verbose:
                    print(f"[INFO] Scaler disimpan ke: {self.scaler_path}")
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Gagal menyimpan scaler: {e}")

        return scaler

    # =========================
    # Mode 1: Dataset BINER (17 fitur fixed)
    # =========================
    def _load_dataset_binary(self):
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset {self.dataset_path} tidak ditemukan!")

        df = pd.read_csv(self.dataset_path, engine="python", on_bad_lines="skip")

        # ---------- 1) Tentukan LABEL dengan prioritas ----------
        def has_col(c): return c in df.columns

        label = None
        for key in self.label_priority:
            if not has_col(key):
                continue
            if key == "label":
                label = pd.to_numeric(df["label"], errors="coerce")
            elif key == "y_attack":
                label = (pd.to_numeric(df["y_attack"], errors="coerce").fillna(0) > 0).astype(int)
            elif key == "auto_label":
                s = df["auto_label"].astype(str).str.strip().str.lower()
                label = (s != "benign").astype(int)
            elif key == "status":
                s = df["status"].astype(str).str.strip().str.lower()
                label = s.isin(["blocked", "detected", "alert"]).astype(int)
            elif key == "action":
                label = (pd.to_numeric(df["action"], errors="coerce").fillna(0) > 0).astype(int)
            elif key == "threat_score":
                label = (pd.to_numeric(df["threat_score"], errors="coerce").fillna(0) > 70).astype(int)
            if label is not None:
                break

        if label is None:
            raise ValueError(
                "Tidak menemukan kolom yang bisa dipakai untuk label. "
                "Butuh salah satu dari: label, y_attack, auto_label, status, action, threat_score."
            )

        df["label"] = pd.to_numeric(label, errors="coerce").fillna(0).astype(int)

        # ---------- 2) Bangun 17 fitur sesuai FEATURE_ORDER ----------
        # Pastikan kolom-kolom inti ada (kalau tidak, isi default)
        must_have_defaults = {
            "timestamp": "",
            "src_port": 0, "dest_port": 0, "dst_port": 0,
            "proto": "", "signature": "",
            "flow_duration": 0.0,
            "fwd_pkts_tot": 0, "bwd_pkts_tot": 0,
            "fwd_data_pkts_tot": 0, "bwd_data_pkts_tot": 0,
            "fwd_pkts_per_sec": 0.0, "bwd_pkts_per_sec": 0.0, "flow_pkts_per_sec": 0.0,
            "down_up_ratio": 0.0,
        }
        for c, default in must_have_defaults.items():
            if c not in df.columns:
                df[c] = default

        # === Normalisasi nama kolom dest_port
        # Jika 'dest_port' kosong tapi ada 'dst_port' → gunakan dst_port
        dest_port_series = df["dest_port"]
        if (dest_port_series == 0).all() and "dst_port" in df.columns:
            dest_port_series = df["dst_port"]

        # Helper numeric
        def to_float(s):
            return pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)

        # timestamp -> hour/weekday
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        hour = ts.dt.hour.fillna(0).astype(float)
        weekday = ts.dt.weekday.fillna(0).astype(float)

        # proto flags
        proto = df["proto"].astype(str).str.lower().fillna("")
        proto_tcp = (proto == "tcp").astype(float)
        proto_udp = (proto == "udp").astype(float)

        # signature feats
        sig = df["signature"].astype(str).fillna("")
        sig_len = sig.str.len().astype(float)
        sig_hash = sig.apply(lambda s: float(int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 1000)).astype(float)

        feat_df = pd.DataFrame({
            "src_port": to_float(df["src_port"]),
            "dest_port": to_float(dest_port_series),
            "proto_tcp": proto_tcp,
            "proto_udp": proto_udp,
            "flow_duration": to_float(df["flow_duration"]),
            "fwd_pkts_tot": to_float(df["fwd_pkts_tot"]),
            "bwd_pkts_tot": to_float(df["bwd_pkts_tot"]),
            "fwd_data_pkts_tot": to_float(df["fwd_data_pkts_tot"]),
            "bwd_data_pkts_tot": to_float(df["bwd_data_pkts_tot"]),
            "fwd_pkts_per_sec": to_float(df["fwd_pkts_per_sec"]),
            "bwd_pkts_per_sec": to_float(df["bwd_pkts_per_sec"]),
            "flow_pkts_per_sec": to_float(df["flow_pkts_per_sec"]),
            "down_up_ratio": to_float(df["down_up_ratio"]),
            "hour": hour,
            "weekday": weekday,
            "sig_len": sig_len,
            "sig_hash": sig_hash,
        })

        # urut sesuai FEATURE_ORDER
        raw_features = feat_df[FEATURE_ORDER].to_numpy(dtype=np.float32)

        # ==== Fit/Load scaler agar konsisten ====
        scaler = self._fit_or_load_scaler(raw_features)
        X = self._sanitize(scaler.transform(raw_features))

        self.features = X
        self.labels = df["label"].to_numpy(dtype=np.int64)
        self._binary_scaler = scaler  # simpan untuk referensi opsional

        if self.verbose:
            pos = int(self.labels.sum())
            neg = int((self.labels == 0).sum())
            print("Dataset berhasil dimuat!")
            print(f"Fitur: {len(FEATURE_ORDER)} kolom (skema runtime)")
            print(f"Sampel: {len(df)} | Normal(0): {neg} | Anomali(1): {pos}")

    # =========================
    # Mode 2: Log KATEGORI (opsional, tidak dipakai di training biner)
    # =========================
    def _load_from_log_category(self, max_steps: int = 2000):
        # Tentukan file log
        if self.log_path:
            path = self.log_path
        else:
            today = datetime.now().strftime("%Y-%m-%d")
            path = f"/var/log/suricata/eve-{today}.json"

        raws = []
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("event_type") == "alert":
                        raws.append(rec)
                        if len(raws) >= max_steps:
                            break
        except FileNotFoundError:
            raise RuntimeError(f"Log file not found: {path}")

        if not raws:
            raise RuntimeError("Tidak ada alert Suricata di log hari ini.")

        # Flatten JSON ke DataFrame
        df = pd.json_normalize(raws, sep="_")

        # Tambahkan kolom default bila hilang
        defaults = {
            "src_ip": "",
            "dest_ip": "",
            "src_port": 0,
            "dest_port": 0,
            "flow_pkts_toserver": 0,
            "flow_pkts_toclient": 0,
            "flow_bytes_toserver": 0,
            "flow_bytes_toclient": 0,
            "proto": "",
            "alert_signature": "",
            "alert_category": "",
            "tcp_tcp_flags": 0,
            "timestamp": "",
        }
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default

        # Fitur temporal
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour_of_day"] = df["timestamp"].dt.hour.fillna(0).astype(int)
        df["day_of_week"] = df["timestamp"].dt.weekday.fillna(0).astype(int)

        # Kategori
        cats = df["alert_category"].fillna("Unknown").unique().tolist()
        self.category_list = cats
        df["category_enc"] = pd.Categorical(
            df["alert_category"].fillna("Unknown"), categories=cats
        ).codes  # 0..N-1

        # Hash/encode IP & categorical
        def hash_ip(ip: str) -> int:
            s = (ip or "").encode()
            return int(hashlib.sha256(s).hexdigest()[:8], 16) % 10000

        df["src_ip_enc"]  = df["src_ip"].fillna("").astype(str).apply(hash_ip)
        df["dest_ip_enc"] = df["dest_ip"].fillna("").astype(str).apply(hash_ip)
        df["proto_enc"]   = df["proto"].fillna("na").astype("category").cat.codes
        df["sig_enc"]     = df["alert_signature"].fillna("none").astype("category").cat.codes

        for c in [
            "src_port", "dest_port",
            "flow_pkts_toserver", "flow_pkts_toclient",
            "flow_bytes_toserver", "flow_bytes_toclient",
            "tcp_tcp_flags",
        ]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

        # Candidate features
        feats = [
            "src_ip_enc", "dest_ip_enc",
            "src_port", "dest_port", "proto_enc",
            "flow_pkts_toserver", "flow_pkts_toclient",
            "flow_bytes_toserver", "flow_bytes_toclient",
            "hour_of_day", "day_of_week",
        ]
        df[feats] = df[feats].fillna(0)

        # Hapus fitur konstan
        selector = VarianceThreshold(threshold=0.0)
        selector.fit(df[feats])
        feats = pd.Index(feats)[selector.get_support()].tolist()

        # Simpan
        self._cat_df = df.reset_index(drop=True)
        self._cat_feats = feats

        scaler = MinMaxScaler().fit(self._cat_df[self._cat_feats].values.astype(np.float32))
        X = self._sanitize(scaler.transform(self._cat_df[self._cat_feats].values.astype(np.float32)))

        self.features = X
        self._cat_scaler = scaler

        # labels “dummy” untuk kompatibilitas (semua 1 = alert)
        self.labels = np.ones(len(self.features), dtype=np.int64)

        # Indexing episode
        self._indices = np.arange(len(self._cat_df))
        np.random.shuffle(self._indices)
        
    # =========================
    # Runtime helper: single-event → 17D normalized
    # =========================
    def _build_feature_row_from_event(self, ev: dict) -> dict:
        """
        Bangun 17 fitur (sesuai FEATURE_ORDER) dari 1 event Suricata-like dict.
        Robust terhadap variasi struktur eve.json (top-level vs flow/alert).
        """
        import hashlib
        import math
        from datetime import datetime

        def _to_float(x, default=0.0):
            try:
                v = float(x)
                if math.isnan(v) or math.isinf(v):
                    return float(default)
                return v
            except Exception:
                return float(default)

        # --- sumber data ---
        flow = ev.get("flow", {}) or {}
        alert = ev.get("alert", {}) or {}

        # ports
        src_port = int(ev.get("src_port") or 0)
        dst_port = int(ev.get("dest_port") or ev.get("dst_port") or 0)

        # proto flags
        proto = (ev.get("proto") or "").strip().lower()
        proto_tcp = 1.0 if proto == "tcp" else 0.0
        proto_udp = 1.0 if proto == "udp" else 0.0

        # timestamps → hour/weekday
        ts_raw = ev.get("timestamp")
        try:
            ts = datetime.fromisoformat(ts_raw) if ts_raw else None
        except Exception:
            from dateutil import parser as _dp
            ts = _dp.parse(ts_raw) if ts_raw else None
        hour = float(ts.hour) if ts else 0.0
        weekday = float(ts.weekday()) if ts else 0.0

        # flow core
        # dur
        def _parse_dt(s):
            if not s:
                return None
            try:
                return datetime.fromisoformat(s)
            except Exception:
                try:
                    from dateutil import parser as _dp
                    return _dp.parse(s)
                except Exception:
                    return None

        start = flow.get("start")
        end = flow.get("end")
        ts_start = _parse_dt(start) or ts
        ts_end = _parse_dt(end) or ts
        if ts_start and ts_end:
            flow_duration = max((ts_end - ts_start).total_seconds(), 0.0)
        else:
            # fallback
            flow_duration = _to_float(flow.get("flow_duration", 0.0), 0.0)

        fwd_pkts = int(flow.get("fwd_pkts_tot", flow.get("pkts_toserver", 0)) or 0)
        bwd_pkts = int(flow.get("bwd_pkts_tot", flow.get("pkts_toclient", 0)) or 0)
        fwd_bytes = int(flow.get("fwd_data_pkts_tot", flow.get("bytes_toserver", 0)) or 0)
        bwd_bytes = int(flow.get("bwd_data_pkts_tot", flow.get("bytes_toclient", 0)) or 0)

        denom = flow_duration if flow_duration > 0 else 1.0
        fwd_pps = float(fwd_pkts) / denom
        bwd_pps = float(bwd_pkts) / denom
        flow_pps = float(fwd_pkts + bwd_pkts) / denom
        down_up_ratio = (float(bwd_pkts) / float(fwd_pkts)) if fwd_pkts > 0 else (999.0 if bwd_pkts > 0 else 0.0)

        # signature-derived
        signature = (alert.get("signature") or ev.get("signature") or "").strip()
        sig_len = float(len(signature))
        sig_hash = float(int(hashlib.sha256(signature.encode()).hexdigest()[:8], 16) % 1000) if signature else 0.0

        # rakit sesuai FEATURE_ORDER
        row = {
            "src_port": float(src_port),
            "dest_port": float(dst_port),
            "proto_tcp": float(proto_tcp),
            "proto_udp": float(proto_udp),
            "flow_duration": _to_float(flow_duration),
            "fwd_pkts_tot": float(fwd_pkts),
            "bwd_pkts_tot": float(bwd_pkts),
            "fwd_data_pkts_tot": float(fwd_bytes),
            "bwd_data_pkts_tot": float(bwd_bytes),
            "fwd_pkts_per_sec": _to_float(fwd_pps),
            "bwd_pkts_per_sec": _to_float(bwd_pps),
            "flow_pkts_per_sec": _to_float(flow_pps),
            "down_up_ratio": _to_float(down_up_ratio, 0.0),
            "hour": _to_float(hour),
            "weekday": _to_float(weekday),
            "sig_len": _to_float(sig_len),
            "sig_hash": _to_float(sig_hash),
        }

        # pastikan semua field ada
        for k in FEATURE_ORDER:
            if k not in row:
                row[k] = 0.0
        return row

    def transform_single_event(self, ev: dict) -> np.ndarray:
        """
        Ubah 1 event menjadi vektor 17D ter-normalisasi (float32).
        Gunakan scaler yang sama dengan training:
          - jika env sudah pernah _load_dataset_binary() → pakai self._binary_scaler
          - else jika self.scaler_path tersedia → load dari file
          - else: raise agar tidak ada “scaler mismatch”.
        """
        import pickle
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler

        # pastikan punya scaler
        scaler = getattr(self, "_binary_scaler", None)
        if scaler is None:
            if self.scaler_path and os.path.exists(self.scaler_path):
                with open(self.scaler_path, "rb") as f:
                    scaler = pickle.load(f)
            else:
                raise RuntimeError(
                    "Scaler belum tersedia. Muat dataset biner terlebih dahulu atau set scaler_path yang valid."
                )

        row = self._build_feature_row_from_event(ev)
        df = pd.DataFrame([row], columns=FEATURE_ORDER)
        X = df.to_numpy(dtype=np.float32)

        Xn = scaler.transform(X)
        # sanitasi seperti di env
        Xn = np.nan_to_num(Xn, nan=0.0, posinf=1.0, neginf=0.0)
        Xn[Xn < 0.0] = 0.0
        Xn[Xn > 1.0] = 1.0
        return Xn[0].astype(np.float32)


    # =========================
    # Gym API
    # =========================
    def reset(self, *, seed: Optional[int] = None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if self.online_mode and self.mode == "binary" and self.dataset_path:
            self._load_dataset_binary()

        self.current_step = 0
        # category mode index shuffle
        if self.mode == "category" and self.shuffle_on_reset:
            np.random.shuffle(self._indices)

        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        if self.mode == "binary":
            # batas aman index
            idx = min(self.current_step, len(self.labels) - 1)
            true_label = int(self.labels[idx])
            reward = float(self.reward_fn(int(action), true_label))

            self.current_step += 1
            terminated = self.current_step >= len(self.features)
            truncated = False
            obs = np.zeros(self.observation_space.shape, dtype=np.float32) if terminated else self._get_obs()
            return obs, reward, terminated, truncated, {}

        # mode kategori (opsional)
        idx = self._indices[self.current_step]
        row = self._cat_df.iloc[idx]
        true_cat = int(row["category_enc"]) + 1  # 1..N; 0=allow

        if int(action) == true_cat:
            r = +10.0
        elif int(action) == 0:
            r = -5.0
        else:
            r = -10.0

        self.current_step += 1
        terminated = self.current_step >= len(self._cat_df)
        truncated = False
        obs = np.zeros(self.features.shape[1], dtype=np.float32) if terminated else self._get_obs()

        info = {
            "true_category": self.category_list[true_cat - 1],
            "predicted_category": None if action == 0 else self.category_list[int(action) - 1],
        }
        return obs, float(r), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        if self.mode == "binary":
            # batas aman index
            idx = min(self.current_step, len(self.features) - 1)
            return np.array(self.features[idx], dtype=np.float32)
        # kategori
        idx = self._indices[self.current_step]
        vec = self._cat_df.loc[idx, self._cat_feats].astype(np.float32).values
        return self._cat_scaler.transform(vec.reshape(1, -1))[0].astype(np.float32)

    # =========================
    # Optional helper (debug)
    # =========================
    def render(self, mode="human"):
        if self.mode == "binary":
            msg = f"Step {self.current_step}/{len(self.features)}"
            if mode == "human":
                print(msg)
            return msg
        idx = self._indices[self.current_step] if self.current_step < len(self._indices) else self._indices[-1]
        cat_idx = int(self._cat_df.loc[idx, "category_enc"])
        cat = self.category_list[cat_idx]
        msg = f"Step {self.current_step}/{len(self._cat_df)} | True={cat}"
        if mode == "human":
            print(msg)
        return msg
