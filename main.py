#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from stable_baselines3 import PPO
import numpy as np
import json
import sqlite3
import time
import os
import socket
import logging
import asyncio
from dateutil import parser as dateparser
from threading import Thread, Lock, RLock
from datetime import datetime, timedelta
import subprocess
import signal
import ipaddress
import random
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict, deque, OrderedDict
import pandas as pd
import math
import psutil
from contextlib import contextmanager
from dataclasses import dataclass

# ==== Gymnasium/Gym fallback ====
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    _GYMNASIUM = False

# ======= project-local =======
from env_anomali import AnomalyEnv  # untuk pipeline training/transformer scaler
from training import train_ppo
from dataset_extension import init_dataset, append_dataset

# ========== LOGGING CONFIG ==========
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ==== Telegram (opsional untuk notifikasi) ====
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)
from telegram.error import BadRequest, NetworkError
from telegram.request import HTTPXRequest  # client yang lebih tahan jaringan flakey

# ========== GLOBAL STATE ==========
shutdown_flag = False
config = None
application = None
bot = None
processed_events = {}
current_log_file = None
LOCAL_IP = None
drl_analyzer = None
db_lock = Lock()
ip_lock = RLock()

# Gating start/stop monitoring
monitor_event: asyncio.Event | None = None  # diinisialisasi di run_application()

# ========== IMPROVED FLOW CACHE ==========
class FlowCache:
    """LRU Cache dengan TTL untuk flow features"""
    def __init__(self, max_size: int = 5000, ttl: int = 900):
        self.max_size = max_size
        self.ttl = ttl
        self._cache = OrderedDict()
        self._lock = RLock()
    
    def put(self, key: Any, value: Dict) -> None:
        with self._lock:
            # Remove oldest if at capacity
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = {
                'value': value, 
                'timestamp': time.time()
            }
            # Move to end (most recently used)
            self._cache.move_to_end(key)
    
    def get(self, key: Any) -> Optional[Dict]:
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            # Check TTL
            if time.time() - entry['timestamp'] > self.ttl:
                del self._cache[key]
                return None
            
            # Move to end (recently used)
            self._cache.move_to_end(key)
            return entry['value']
    
    def cleanup(self) -> int:
        """Bersihkan expired entries, return jumlah yang dihapus"""
        with self._lock:
            now = time.time()
            to_delete = []
            for key, entry in self._cache.items():
                if now - entry['timestamp'] > self.ttl:
                    to_delete.append(key)
            
            for key in to_delete:
                del self._cache[key]
            
            return len(to_delete)

# Initialize improved cache
FLOW_CACHE = FlowCache(max_size=5000, ttl=900)

# ========== SYSTEM METRICS ==========
@dataclass
class SystemMetrics:
    """Metrics collection untuk monitoring performance"""
    queue_size: int = 0
    avg_processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    events_processed: int = 0
    cache_hit_rate: float = 0.0
    cache_size: int = 0
    db_operations: int = 0

class MetricsCollector:
    def __init__(self):
        self.metrics = SystemMetrics()
        self.start_time = time.time()
        self.processing_times = deque(maxlen=100)
        self.cache_hits = 0
        self.cache_misses = 0
    
    def update_queue_size(self, size: int):
        self.metrics.queue_size = size
    
    def record_processing_time(self, duration: float):
        self.processing_times.append(duration)
        self.metrics.avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        self.metrics.events_processed += 1
    
    def record_cache_hit(self, hit: bool):
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        total = self.cache_hits + self.cache_misses
        self.metrics.cache_hit_rate = self.cache_hits / total if total > 0 else 0.0
    
    def record_db_operation(self):
        self.metrics.db_operations += 1
    
    def update_system_metrics(self):
        process = psutil.Process()
        self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        self.metrics.cache_size = len(FLOW_CACHE._cache)
    
    def get_metrics(self) -> SystemMetrics:
        self.update_system_metrics()
        return self.metrics
    
    def get_uptime(self) -> float:
        return time.time() - self.start_time

METRICS = MetricsCollector()

# ========== CONFIGURATION ==========
CONFIG_DEFAULTS = {
    "enable_telegram": True,
    "token": "7639764590:AAHjCwXxfrEOqFc9l2Da-J2dWNAkhPZXikg",
    "chat_id": "6696560598",
    "suricata_log_path": "/var/log/suricata/eve-%Y-%m-%d.json",
    "drl_model_path": "models/ppo_anomali_latest.zip",
    "scaler_path": "models/minmax.pkl",
    "attack_clf_path": "models/attack_classifier.pkl",
    "expected_obs_dim": 17,

    # I/O & timeout
    "pool_size": 8,
    "timeout": 30,
    "read_timeout": 30,
    "write_timeout": 30,
    "connect_timeout": 30,
    "cooldown": 1,
    "event_cooldown": 60,
    "log_rotation_check_interval": 60,

    # Tail tuning
    "tail_idle_sleep_ms": 150,
    "tail_error_backoff_ms": 500,

    # Net policy
    "whitelist_ips": ["127.0.0.1", "192.168.1.1"],
    "min_confidence": 0.70,
    "monitor_direction": "inbound",
    "home_net_cidr": "192.168.38.0/24",

    # Benign capture & flow filtering
    "capture_benign": True,
    "benign_sample_rate": 0.15,
    "flow_min_pkts": 4,
    "flow_min_bytes": 600,
    "flow_cache_ttl_s": 900,

    # Flow heuristic thresholds
    "dos_pps_thr": 3555.0,
    "dos_fwd_thr": 40,
    "scan_dur_thr": 0.20,
    "scan_bwd_thr": 2,
    "brute_fwd_min": 4,
    "brute_bwd_max": 3,

    # Auto-unblock
    "auto_unblock_check_interval_s": 60,

    # NEW: default ignored signatures (override-able via config file)
    "ignored_signatures": [
        "SURICATA Applayer Detector failed",
        "SURICATA STREAM ESTABLISHED",
        "SURICATA TLS invalid record",
        "SURICATA STREAM TIMEWAIT",
        "SURICATA UDPv4 invalid checksum",
        "SURICATA ICMPv6 unknown type"
    ]
}

# ==== DB PATH terpusat ====
DB_PATH = "anomali.db"

# ==== Hanya 3 kelas serangan: SCAN / BRUTEFORCE / DOS ====
SIGNATURE_KLASIFIKASI = {
    "scan": [
        "scan", "portscan", "masscan", "nmap",
        "xmas scan", "syn scan", "fin scan", "null scan"
    ],
    "bruteforce": [
        "brute", "bruteforce", "dictionary", "credential stuffing",
        "password guess", "hydra", "medusa", "patator"
    ],
    "dos": [
        "dos", "denial of service", "ddos", "flood",
        "syn flood", "icmp flood", "udp flood", "slowloris", "slow http"
    ],
}

def klasifikasi_jenis_serangan(signature: str) -> str:
    s = (signature or "").lower()
    for kategori, keywords in SIGNATURE_KLASIFIKASI.items():
        if any(k in s for k in keywords):
            return kategori.upper()  # SCAN | BRUTEFORCE | DOS
    return "UNKNOWN"

# Pemetaan ringan dari category/signature alert → 3 kelas
ALERT_MAP = {
    "SSH Brute Force": "bruteforce",
    "Bruteforce": "bruteforce",
    "DoS": "dos",
    "DDoS": "dos",
    "Scan": "scan",
    "Port Scan": "scan",
}

# Prioritas resolusi label
HIERARCHY = ["dos", "bruteforce", "scan", "anomaly", "benign"]
ATTACK_LABELS = {"dos", "bruteforce", "scan", "anomaly"}

# ====== NUMERIC SAFETY HELPERS ======
def _safe_num(x, lo=-1e12, hi=1e12, default=0.0):
    try:
        xf = float(x)
    except Exception:
        return float(default)
    if math.isnan(xf) or math.isinf(xf):
        return float(default)
    if xf < lo:
        return float(lo)
    if xf > hi:
        return float(hi)
    return xf

def _safe_ratio(num, den, default=0.0):
    try:
        denf = float(den)
        numf = float(num)
        if denf == 0.0:
            return float(default)
        return _safe_num(numf / denf, default=default)
    except Exception:
        return float(default)

def _sanitize_flow_numbers(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float)):
            out = {**out, k: _safe_num(v)}
        else:
            out = {**out, k: v}
    return out

ALLOW_SIG_KEYWORDS = ["STUN", "Session Traversal Utilities for NAT", "Binding Response"]
ALLOW_CATEGORIES = {"INFO"}
ALLOW_UDP_PORTS = {3478}

# ==== FAST-TRACK (alert+flow kompak, auto-label tinggi) ====
AUTO_BLOCK_CONF_THR = 0.90
AUTO_BLOCK_REQUIRE_SOURCES = {"alert", "flow"}

# ===== Hit counter + agregat per-src =====
RECENT_HITS = defaultdict(lambda: deque(maxlen=40))      # timestamps deteksi per-src
WINDOW_S = 30
HITS_N = 3

AGG_LOCK = RLock()
# src_ip -> dict(deque times)
PER_SRC = defaultdict(lambda: {
    "flows": deque(maxlen=200),      # timestamps flow
    "alerts": deque(maxlen=200),     # timestamps alert
    "syn": deque(maxlen=200)         # timestamps alert yang mengandung 'SYN'
})
LAST_SENT = {}  # (src, signature) -> ts (dedup notifikasi)

# NEW: ignored signatures (default + bisa override via config)
IGNORED_SIGNATURES_DEFAULT = tuple(CONFIG_DEFAULTS["ignored_signatures"])
IGNORED_SIGNATURES: tuple[str, ...] = IGNORED_SIGNATURES_DEFAULT

def _is_benign_event(event: dict) -> bool:
    a = event.get("alert", {}) or {}
    sig = (a.get("signature") or "").upper()
    cat = (a.get("category") or "").upper()
    proto = (event.get("proto") or "").upper()
    dport = int(event.get("dest_port") or event.get("dst_port") or 0)
    if cat in ALLOW_CATEGORIES:
        return True
    if any(k in sig for k in (s.upper() for s in ALLOW_SIG_KEYWORDS)):
        return True
    if proto == "UDP" and dport in ALLOW_UDP_PORTS:
        return True
    return False

def _seen_enough_hits(ip: str, window_s: int = WINDOW_S, min_hits: int = HITS_N) -> bool:
    now = time.time()
    dq = RECENT_HITS[ip]
    dq.append(now)
    while dq and now - dq[0] > window_s:
        dq.popleft()
    return len(dq) >= min_hits

def _update_aggregates_from_flow(ev: dict):
    src = ev.get("src_ip")
    if not src:
        return
    with AGG_LOCK:
        PER_SRC[src]["flows"].append(time.time())

def _update_aggregates_from_alert(ev: dict):
    src = ev.get("src_ip")
    if not src:
        return
    sig_l = (ev.get("signature") or "").lower()
    now = time.time()
    with AGG_LOCK:
        PER_SRC[src]["alerts"].append(now)
        if "syn" in sig_l:
            PER_SRC[src]["syn"].append(now)

def _compute_src_aggregates(src: str) -> tuple[float, float, int]:
    """Return (flows_per_sec, syn_only_ratio, alerts_count_5s)."""
    now = time.time()
    with AGG_LOCK:
        d = PER_SRC.get(src)
        if not d:
            return 0.0, 0.0, 0
        # purging
        for k in ("flows", "alerts", "syn"):
            dq = d[k]
            while dq and now - dq[0] > 10.0:
                dq.popleft()
        flows_per_sec = _safe_num(len(d["flows"]) / 10.0)
        alerts_5s = [t for t in d["alerts"] if now - t <= 5.0]
        syn_5s = [t for t in d["syn"] if now - t <= 5.0]
        syn_only_ratio = _safe_ratio(len(syn_5s), max(1, len(alerts_5s)))
        return float(flows_per_sec), float(syn_only_ratio), int(len(alerts_5s))

# ========== IMPROVED DATABASE HANDLING ==========
@contextmanager
def db_connection():
    """Context manager untuk koneksi database yang thread-safe"""
    with db_lock:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Untuk akses kolom by name
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
            METRICS.record_db_operation()

def init_db():
    """
    - Membuat file DB jika belum ada.
    - Membuat/menambah tabel & kolom yang dibutuhkan.
    - Membuat indeks.
    - Menjalankan normalisasi/migrasi nilai agar konsisten dengan runtime saat ini.
    """
    with db_connection() as conn:
        try:
            c = conn.cursor()
            c.execute("PRAGMA journal_mode=WAL;")
            c.execute("PRAGMA synchronous=NORMAL;")

            # Tabel log
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    src_ip TEXT,
                    dst_ip TEXT,
                    signature TEXT,
                    status TEXT,
                    action_by TEXT,
                    confidence REAL,
                    log_file TEXT,
                    jenis_serangan TEXT,
                    direction TEXT
                )
                """
            )

            # Pastikan kolom 'direction' ada (untuk DB lama)
            c.execute("PRAGMA table_info(log)")
            columns = [col[1] for col in c.fetchall()]
            if "direction" not in columns:
                c.execute("ALTER TABLE log ADD COLUMN direction TEXT")
                logger.info("Added direction column to log table")

            # Tabel blocked_ips
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS blocked_ips (
                    ip TEXT PRIMARY KEY,
                    timestamp TEXT,
                    reason TEXT,
                    confidence REAL,
                    unblock_time TEXT
                )
                """
            )

            # Index
            c.execute("CREATE INDEX IF NOT EXISTS idx_log_ts ON log(timestamp)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_blocked_unblock ON blocked_ips(unblock_time)")

            # Jalankan migrasi nilai agar konsisten
            migrate_db_values(conn)

            logger.info("Database initialized & migrated")

        except Exception as e:
            logger.error(f"DB init/migration error: {e}")
            raise

def migrate_db_values(conn: sqlite3.Connection):
    """
    Normalisasi nilai historis supaya konsisten dengan runtime saat ini:
    - BRUTE -> BRUTEFORCE
    - DDoS -> DOS
    """
    try:
        c = conn.cursor()
        updates = [
            ("BRUTEFORCE", "BRUTE"),
            ("DOS", "DDoS"),
        ]
        for new_v, old_v in updates:
            c.execute("UPDATE log SET jenis_serangan=? WHERE jenis_serangan=?", (new_v, old_v))
    except Exception as e:
        logger.warning(f"Value migration warning: {e}")

def simpan_log(timestamp, src_ip, dst_ip, signature, status, action_by="system", confidence=None, direction="inbound"):
    jenis_serangan = klasifikasi_jenis_serangan(signature)
    with db_connection() as conn:
        try:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO log
                (timestamp, src_ip, dst_ip, signature, status, action_by, confidence, log_file, jenis_serangan, direction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (timestamp, src_ip, dst_ip, signature, status, action_by, confidence, current_log_file, jenis_serangan, direction),
            )
        except Exception as e:
            logger.error(f"Error saving log: {e}")
            raise

# ========== CONFIGURATION VALIDATION ==========
def validate_config(config: Dict) -> Tuple[Dict, List[str]]:
    """Validasi konfigurasi dan return (config_valid, error_messages)"""
    errors = []
    
    # Validasi required fields untuk Telegram
    if config.get("enable_telegram", True):
        if not config.get("token"):
            errors.append("Telegram token is required when enable_telegram is True")
        if not config.get("chat_id"):
            errors.append("Telegram chat_id is required when enable_telegram is True")
    
    # Validasi numeric ranges
    try:
        config["benign_sample_rate"] = max(0.0, min(1.0, float(config.get("benign_sample_rate", 0.15))))
    except (ValueError, TypeError):
        errors.append("benign_sample_rate must be a float between 0.0 and 1.0")
    
    try:
        config["min_confidence"] = max(0.0, min(1.0, float(config.get("min_confidence", 0.70))))
    except (ValueError, TypeError):
        errors.append("min_confidence must be a float between 0.0 and 1.0")
    
    # Validasi path exists untuk model files
    model_path = config.get("drl_model_path")
    if model_path and not os.path.exists(model_path):
        errors.append(f"DRL model path does not exist: {model_path}")
    
    scaler_path = config.get("scaler_path")
    if scaler_path and not os.path.exists(scaler_path):
        errors.append(f"Scaler path does not exist: {scaler_path}")
    
    # Validasi CIDR format
    try:
        home_net = config.get("home_net_cidr", "192.168.38.0/24")
        ipaddress.ip_network(home_net, strict=False)
    except ValueError:
        errors.append(f"Invalid home_net_cidr: {home_net}")
    
    return config, errors

def load_config():
    global config, FLOW_CACHE, IGNORED_SIGNATURES
    path = "bot_config.json"
    try:
        if not os.path.exists(path):
            logger.error("Config file not found, creating default config")
            with open(path, "w") as f:
                json.dump(CONFIG_DEFAULTS, f, indent=2)
            config = dict(CONFIG_DEFAULTS)
        else:
            with open(path) as f:
                config = json.load(f)

        # Apply defaults untuk missing keys
        for k, v in CONFIG_DEFAULTS.items():
            if k not in config:
                config[k] = v
                logger.warning(f"Using default value for missing config: {k}={v}")

        # Validasi konfigurasi
        config, errors = validate_config(config)
        if errors:
            for error in errors:
                logger.error(f"Config validation error: {error}")
            if config.get("enable_telegram", True):
                logger.warning("Disabling Telegram due to config errors")
                config["enable_telegram"] = False

        # sync ignored signatures dari config
        try:
            IGNORED_SIGNATURES = tuple(config.get("ignored_signatures", list(IGNORED_SIGNATURES_DEFAULT)))
        except Exception:
            IGNORED_SIGNATURES = IGNORED_SIGNATURES_DEFAULT

        # Update cache TTL
        cache_ttl = int(config.get("flow_cache_ttl_s", 900))
        FLOW_CACHE = FlowCache(max_size=5000, ttl=cache_ttl)
        
        return config
    except Exception as e:
        logger.critical(f"Failed to load config: {e}")
        config = dict(CONFIG_DEFAULTS)
        config["enable_telegram"] = False
        IGNORED_SIGNATURES = IGNORED_SIGNATURES_DEFAULT
        return config

# ========== UTILITY FUNCTIONS ==========
def get_local_ip():
    global LOCAL_IP
    if not LOCAL_IP:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 1))
            LOCAL_IP = s.getsockname()[0]
        except Exception:
            LOCAL_IP = "127.0.0.1"
        finally:
            s.close()
    return LOCAL_IP

def get_latest_suricata_log():
    pattern = config.get("suricata_log_path", "/var/log/suricata/eve-%Y-%m-%d.json")
    try:
        log_path = datetime.now().strftime(pattern)
    except Exception:
        log_path = pattern
    if os.path.exists(log_path):
        return log_path
    fallback = "/var/log/suricata/eve.json"
    if os.path.exists(fallback):
        return fallback
    logger.warning(f"Suricata log not found at {log_path} (and no fallback).")
    return None

# ========== FLOW FEATURE HELPERS ==========
def _dt_from_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        try:
            return dateparser.parse(ts)
        except Exception:
            return None

def _compute_flow_features(ev: dict) -> dict:
    flow = ev.get("flow", {}) or {}
    start = flow.get("start")
    end = flow.get("end")
    ts_start = _dt_from_iso(start) or _dt_from_iso(ev.get("timestamp"))
    ts_end = _dt_from_iso(end) or _dt_from_iso(ev.get("timestamp"))
    duration = 0.0
    if ts_start and ts_end:
        duration = max((ts_end - ts_start).total_seconds(), 0.0)

    fwd_pkts = int(flow.get("pkts_toserver", 0) or 0)
    bwd_pkts = int(flow.get("pkts_toclient", 0) or 0)
    fwd_bytes = int(flow.get("bytes_toserver", 0) or 0)
    bwd_bytes = int(flow.get("bytes_toclient", 0) or 0)

    denom = duration if duration > 0 else 1.0
    fwd_pps = _safe_num(fwd_pkts / denom)
    bwd_pps = _safe_num(bwd_pkts / denom)
    flow_pps = _safe_num((fwd_pkts + bwd_pkts) / denom)
    down_up_ratio = _safe_ratio(bwd_pkts, fwd_pkts, default=0.0)

    feats = {
        "flow_duration": _safe_num(duration),
        "fwd_pkts_tot": int(fwd_pkts),
        "bwd_pkts_tot": int(bwd_pkts),
        "fwd_data_pkts_tot": int(fwd_bytes),
        "bwd_data_pkts_tot": int(bwd_bytes),
        "fwd_pkts_per_sec": fwd_pps,
        "bwd_pkts_per_sec": bwd_pps,
        "flow_pkts_per_sec": flow_pps,
        "down_up_ratio": down_up_ratio,
        "fwd_header_size_tot": 0.0,
        "fwd_header_size_min": 0.0,
        "fwd_header_size_max": 0.0,
        "bwd_header_size_tot": 0.0,
        "bwd_header_size_min": 0.0,
        "bwd_header_size_max": 0.0,
    }
    return feats

def _tuple_key(ev: dict) -> Tuple[str, str, int, int, str]:
    src_ip = ev.get("src_ip")
    dst_ip = ev.get("dest_ip") or ev.get("dst_ip")
    src_p = int(ev.get("src_port") or 0)
    dst_p = int(ev.get("dest_port") or ev.get("dst_port") or 0)
    proto = (ev.get("proto") or "").upper()
    return (src_ip or "", dst_ip or "", src_p, dst_p, proto)

def _cache_put(ev: dict, feats: dict):
    now = time.time()
    ci = ev.get("community_id")
    fid = ev.get("flow_id")
    tkey = _tuple_key(ev)
    
    if ci:
        FLOW_CACHE.put(ci, feats)
    if fid:
        FLOW_CACHE.put(fid, feats)
    FLOW_CACHE.put(tkey, feats)

def _cache_get(ev: dict) -> Optional[dict]:
    ci = ev.get("community_id")
    if ci:
        result = FLOW_CACHE.get(ci)
        if result:
            METRICS.record_cache_hit(True)
            return result

    fid = ev.get("flow_id")
    if fid:
        result = FLOW_CACHE.get(fid)
        if result:
            METRICS.record_cache_hit(True)
            return result

    tkey = _tuple_key(ev)
    result = FLOW_CACHE.get(tkey)
    if result:
        METRICS.record_cache_hit(True)
        return result
    
    METRICS.record_cache_hit(False)
    return None

def _merge_flow_features_into_event(event: dict):
    base_flow = event.get("flow", {}) or {}
    feats = _cache_get(event)
    if feats is None and base_flow:
        feats = _compute_flow_features(event)
    elif feats is None:
        feats = {
            "flow_duration": 0.0,
            "fwd_pkts_tot": 0,
            "bwd_pkts_tot": 0,
            "fwd_data_pkts_tot": 0,
            "bwd_data_pkts_tot": 0,
            "fwd_pkts_per_sec": 0.0,
            "bwd_pkts_per_sec": 0.0,
            "flow_pkts_per_sec": 0.0,
            "down_up_ratio": 0.0,
            "fwd_header_size_tot": 0.0,
            "fwd_header_size_min": 0.0,
            "fwd_header_size_max": 0.0,
            "bwd_header_size_tot": 0.0,
            "bwd_header_size_min": 0.0,
            "bwd_header_size_max": 0.0,
        }
    merged = dict(base_flow)
    merged.update(feats)
    event["flow"] = _sanitize_flow_numbers(merged)

# ========== NFTABLES ==========
NFT_TABLE = ("inet", "filter")
NFT_SET_V4 = "blocked_v4"
NFT_SET_V6 = "blocked_v6"

def _run_nft(args: List[str]) -> bool:
    cmds = [["sudo", "nft", *args], ["nft", *args]]
    for cmd in cmds:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.debug(f"nft call failed ({' '.join(cmd)}): {e.stderr or e}")
            continue
        except FileNotFoundError:
            continue
    return False

def ensure_nftables():
    family, table = NFT_TABLE
    if not _run_nft(["list", "table", family, table]):
        _run_nft(["add", "table", family, table])
    if not _run_nft(["list", "chain", family, table, "input"]):
        _run_nft(
            ["add", "chain", family, table, "input", "{", "type", "filter",
             "hook", "input", "priority", "0", ";", "policy", "accept", ";", "}"]
        )
    if not _run_nft(["list", "set", family, table, NFT_SET_V4]):
        _run_nft(["add", "set", family, table, NFT_SET_V4, "{", "type", "ipv4_addr", ";", "flags", "interval", ";", "}"])
    if not _run_nft(["list", "set", family, table, NFT_SET_V6]):
        _run_nft(["add", "set", family, table, NFT_SET_V6, "{", "type", "ipv6_addr", ";", "flags", "interval", ";", "}"])
    _run_nft(["add", "rule", family, table, "input", "ip", "saddr", f"@{NFT_SET_V4}", "drop"])
    _run_nft(["add", "rule", family, table, "input", "ip6", "saddr", f"@{NFT_SET_V6}", "drop"])

def _nft_add_ip(ip: str) -> bool:
    try:
        ip_obj = ipaddress.ip_address(ip)
    except ValueError:
        logger.error(f"Invalid IP for nft: {ip}")
        return False
    family, table = NFT_TABLE
    if isinstance(ip_obj, ipaddress.IPv4Address):
        return _run_nft(["add", "element", family, table, NFT_SET_V4, "{", ip, "}"])
    else:
        return _run_nft(["add", "element", family, table, NFT_SET_V6, "{", ip, "}"])

def _nft_del_ip(ip: str) -> bool:
    try:
        ip_obj = ipaddress.ip_address(ip)
    except ValueError:
        logger.error(f"Invalid IP for nft: {ip}")
        return False
    family, table = NFT_TABLE
    if isinstance(ip_obj, ipaddress.IPv4Address):
        _run_nft(["delete", "element", family, table, NFT_SET_V4, "{", ip, "}"])
    else:
        _run_nft(["delete", "element", family, table, NFT_SET_V6, "{", ip, "}"])
    return True

def blokir_ip(ip, metode="manual", confidence=None, reason=""):
    if ip in config["whitelist_ips"] or ip == LOCAL_IP:
        logger.info(f"Skipping whitelisted IP: {ip}")
        return False
    with ip_lock:
        try:
            with db_connection() as conn:
                c = conn.cursor()
                c.execute("SELECT 1 FROM blocked_ips WHERE ip=?", (ip,))
                if c.fetchone():
                    return False
                
                ensure_nftables()
                if not _nft_add_ip(ip):
                    logger.error(f"Failed to add {ip} to nft set")
                    return False
                
                unblock_time = (datetime.now() + timedelta(hours=24)).isoformat()
                c.execute(
                    """
                    INSERT INTO blocked_ips (ip, timestamp, reason, confidence, unblock_time)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (ip, datetime.now().isoformat(), reason, confidence, unblock_time),
                )
                logger.info(f"[✓] IP {ip} blocked via nftables ({metode}) - {reason}")
                return True
                
        except Exception as e:
            logger.error(f"[!] Failed to block IP {ip}: {e}")
            return False

def izinkan_ip(ip):
    with ip_lock:
        try:
            ensure_nftables()
            _nft_del_ip(ip)
            with db_connection() as conn:
                c = conn.cursor()
                c.execute("DELETE FROM blocked_ips WHERE ip=?", (ip,))
            logger.info(f"[✓] IP {ip} unblocked (nftables)")
            return True
        except Exception as e:
            logger.error(f"[!] Failed to unblock IP {ip}: {e}")
            return False

# ========== DRL ==========
class DRLAnalyzer:
    """
    Loader PPO; observasi 17D mengikuti EXACT FEATURE_ORDER milik env_anomali.
    Runtime memakai AnomalyEnv.transform_single_event() + scaler yang sama.
    Attack classifier (opsional) untuk {scan, bruteforce, dos}.
    """
    def __init__(self, model_path: str):
        self.vecnorm = None
        self.model = PPO.load(model_path)
        logger.info(f"DRL model loaded from {model_path}")

        try:
            scaler_path = (config or {}).get("scaler_path", "models/minmax.pkl")
        except Exception:
            scaler_path = "models/minmax.pkl"

        try:
            self.env_transform = AnomalyEnv(
                dataset_path=None,
                scaler_path=scaler_path,
                load_scaler=True,
                save_scaler=False,
                verbose=False
            )
        except Exception as e:
            logger.critical(
                f"Gagal inisialisasi transformer ENV/Scaler. Pastikan scaler_path benar. Error: {e}"
            )
            self.env_transform = None

        try:
            self.expected_features = int(self.model.observation_space.shape[0])  # type: ignore
        except Exception:
            self.expected_features = None

        self.attack_clf = None
        self.feature_order: Optional[List[str]] = None
        self.attack_classes: Optional[List[str]] = None
        clf_path = (config or {}).get("attack_clf_path", "models/attack_classifier.pkl")
        try:
            if os.path.exists(clf_path):
                from joblib import load
                art: Dict[str, object] = load(clf_path)
                self.attack_clf = art.get("model")
                self.feature_order = art.get("feature_order")  # type: ignore
                self.attack_classes = art.get("classes_")      # type: ignore
                if self.attack_clf is not None:
                    logger.info(f"Attack classifier loaded from {clf_path} with classes: {self.attack_classes}")
        except Exception as e:
            logger.warning(f"Attack classifier NOT loaded: {e}")

    def _ensure_obs_dim(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        if self.expected_features is not None and x.shape[1] != self.expected_features:
            n = int(self.expected_features)
            if x.shape[1] < n:
                pad = np.zeros((1, n - x.shape[1]), dtype=np.float32)
                x = np.concatenate([x, pad], axis=1)
            elif x.shape[1] > n:
                x = x[:, :n]
            logger.warning(
                f"[DRLAnalyzer] Obs dim mismatch (got {x.shape[1]} vs model {n}). "
                "Cek kembali FEATURE_ORDER & scaler yang dipakai saat training."
            )
        return x

    def preprocess_event(self, event: dict) -> Optional[np.ndarray]:
        try:
            flow = event.get("flow", {}) or {}
            if "flow_pkts_per_sec" not in flow or "down_up_ratio" not in flow:
                _merge_flow_features_into_event(event)

            if self.env_transform is None:
                logger.error("env_transform belum siap (scaler tidak termuat).")
                return None

            vec17 = self.env_transform.transform_single_event(event)  # shape (17,)
            feats = np.nan_to_num(vec17, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            feats = self._ensure_obs_dim(feats)
            return feats
        except Exception as e:
            logger.error(f"Error preprocessing event via env: {e}")
            return None

    def predict(self, event: dict) -> Tuple[int, float, Optional[str]]:
        """
        Return:
          - binary action (0/1),
          - confidence,
          - attack_type (scan/bruteforce/dos) jika classifier tersedia
        """
        try:
            obs = self.preprocess_event(event)
            if obs is None:
                return 0, 0.0, None

            action, _ = self.model.predict(obs, deterministic=True)
            act = int(action[0]) if isinstance(action, (list, np.ndarray)) else int(action)
            confidence = 0.80 if act == 1 else 0.75

            attack_type = None
            if self.attack_clf is not None:
                try:
                    vec_df = self.env_transform.transform_single_event_df(event)  # [1 x 17]
                    pred = self.attack_clf.predict(vec_df)[0]  # type: ignore
                    attack_type = str(pred)
                except Exception as e:
                    logger.debug(f"Attack classifier predict failed: {e}")

            return act, float(confidence), attack_type
        except Exception as e:
            logger.error(f"[!] DRL prediction error: {e}")
            return 0, 0.0, None

# ========== LABELING (ALERT + FLOW) ==========
def label_from_alert(alert_obj: dict) -> Optional[dict]:
    if not alert_obj:
        return None
    cat = (alert_obj.get("category") or alert_obj.get("signature") or "").lower()
    sev = alert_obj.get("severity", 2)
    for k, v in ALERT_MAP.items():
        if k.lower() in cat:
            score = {1: 0.95, 2: 0.9, 3: 0.85}.get(sev, 0.9)
            return {"label": v, "score": score, "src": "alert"}
    sig = (alert_obj.get("signature") or "")
    mapped = klasifikasi_jenis_serangan(sig).lower()
    if mapped != "unknown":
        score = {1: 0.95, 2: 0.9, 3: 0.85}.get(sev, 0.85)
        return {"label": mapped if mapped in HIERARCHY else "anomaly", "score": score, "src": "alert"}
    return None

def label_from_flow_feats(event: dict) -> dict:
    flow = event.get("flow", {}) or {}

    duration = _safe_num(flow.get("flow_duration", 0.0))
    fwd = int(flow.get("fwd_pkts_tot", 0) or 0)
    bwd = int(flow.get("bwd_pkts_tot", 0) or 0)
    pps = _safe_num(flow.get("flow_pkts_per_sec", 0.0))
    dport = int(event.get("dest_port", event.get("dst_port", 0)) or 0)

    DOS_PPS_THR = _safe_num(config.get("dos_pps_thr", 3555.0))
    DOS_FWD_THR = int(config.get("dos_fwd_thr", 40))
    SCAN_DUR_THR = _safe_num(config.get("scan_dur_thr", 0.20))
    SCAN_BWD_THR = int(config.get("scan_bwd_thr", 2))
    BF_FWD_MIN = int(config.get("brute_fwd_min", 4))
    BF_BWD_MAX = int(config.get("brute_bwd_max", 3))

    if (pps > DOS_PPS_THR) or (fwd > DOS_FWD_THR):
        return {"label": "dos", "score": 0.70, "src": "flow"}
    if (duration < SCAN_DUR_THR) and (bwd <= SCAN_BWD_THR) and (dport not in (0,)):
        return {"label": "scan", "score": 0.65, "src": "flow"}
    if (dport == 22) and (fwd >= BF_FWD_MIN) and (bwd <= BF_BWD_MAX):
        return {"label": "bruteforce", "score": 0.60, "src": "flow"}
    return {"label": "benign", "score": 0.50, "src": "flow"}

def resolve_conflict(candidates: List[dict]) -> dict:
    if not candidates:
        return {"label": "benign", "confidence": 0.40, "sources": []}
    by_label = defaultdict(list)
    for c in candidates:
        by_label[c["label"]].append(c)
    for pref in HIERARCHY:
        if pref in by_label:
            best = max(by_label[pref], key=lambda x: x["score"])
            agree = [c for c in candidates if c["label"] == pref]
            conf = min(1.0, best["score"] + 0.1 * (len(agree) - 1))
            srcs = list({c["src"] for c in agree})
            return {"label": pref, "confidence": conf, "sources": srcs}
    best = max(candidates, key=lambda x: x["score"])
    return {"label": best["label"], "confidence": best["score"], "sources": [best["src"]]}

def label_flow(event: dict) -> dict:
    cands = []
    a = label_from_alert(event.get("alert", {}) or {})
    if a:
        cands.append(a)
    cands.append(label_from_flow_feats(event))
    result = resolve_conflict(cands)
    return result

# ======== BINARY LABEL (0/1) ========
def binary_from_alert_flow_drl(auto: dict, drl_action: int, drl_conf: float, min_conf: float = None) -> tuple[int, str]:
    if min_conf is None:
        try:
            min_conf = float(config.get("min_confidence", 0.70))
        except Exception:
            min_conf = 0.70
    if int(drl_action) == 1 and float(drl_conf) >= min_conf:
        return 1, "drl"
    if auto and auto.get("label") in ATTACK_LABELS and float(auto.get("confidence", 0.0)) >= AUTO_BLOCK_CONF_THR:
        return 1, "auto"
    return 0, "none"

# ========= DATASET (untuk retraining DRL) =========
def convert_to_training_format(event, action, confidence):
    flow = event.get("flow", {}) or {}
    signature = event.get("signature") or "Unknown"
    direction = event.get("direction", "inbound")
    status = event.get("status", "detected")
    # agregat per-src untuk retraining next-gen
    flows_per_sec = event.get("flows_per_sec", 0.0)
    syn_only_ratio = event.get("syn_only_ratio", 0.0)
    alerts_count_5s = event.get("alerts_count_5s", 0)

    return {
        "timestamp": event.get("timestamp", datetime.now().isoformat()),
        "src_ip": event.get("src_ip", ""),
        "dst_ip": event.get("dst_ip", ""),
        "src_port": event.get("src_port", 0),
        "dst_port": event.get("dest_port", 0) or event.get("dst_port", 0),
        "proto": event.get("proto", "unknown"),
        "flow_duration": flow.get("flow_duration", 0.0),
        "fwd_pkts_tot": flow.get("fwd_pkts_tot", flow.get("pkts_toserver", 0)),
        "bwd_pkts_tot": flow.get("bwd_pkts_tot", flow.get("pkts_toclient", 0)),
        "fwd_data_pkts_tot": flow.get("fwd_data_pkts_tot", flow.get("bytes_toserver", 0)),
        "bwd_data_pkts_tot": flow.get("bwd_data_pkts_tot", flow.get("bytes_toclient", 0)),
        "fwd_pkts_per_sec": flow.get("fwd_pkts_per_sec", 0.0),
        "bwd_pkts_per_sec": flow.get("bwd_pkts_per_sec", 0.0),
        "flow_pkts_per_sec": flow.get("flow_pkts_per_sec", 0.0),
        "down_up_ratio": flow.get("down_up_ratio", 0.0),
        "fwd_header_size_tot": flow.get("fwd_header_size_tot", 0.0),
        "fwd_header_size_min": flow.get("fwd_header_size_min", 0.0),
        "fwd_header_size_max": flow.get("fwd_header_size_max", 0.0),
        "bwd_header_size_tot": flow.get("bwd_header_size_tot", 0.0),
        "bwd_header_size_min": flow.get("bwd_header_size_min", 0.0),
        "bwd_header_size_max": flow.get("bwd_header_size_max", 0.0),
        "signature": signature,
        "action": action,
        "confidence": confidence,
        "status": status,
        "direction": direction,
        # agregat runtime
        "flows_per_sec": flows_per_sec,
        "syn_only_ratio": syn_only_ratio,
        "alerts_count_5s": alerts_count_5s,
        # label biner
        "y_attack": int(event.get("y_attack", 0)),
        "y_reason": event.get("y_reason", ""),
    }

# ========== HOME_NET helpers ==========
def _dst_in_homenet(dst_ip: str) -> bool:
    try:
        cidr = config.get("home_net_cidr", CONFIG_DEFAULTS["home_net_cidr"])
        net = ipaddress.ip_network(cidr, strict=False)
        return ipaddress.ip_address(dst_ip) in net
    except Exception:
        logger.error("Invalid home_net_cidr; refusing to treat traffic as inbound by CIDR.")
        return False

# ========== BENIGN CAPTURE ==========
def _flow_stats(flow: dict):
    pkts_to_srv = int(flow.get("pkts_toserver", flow.get("fwd_pkts_tot", 0)))
    pkts_to_cli = int(flow.get("pkts_toclient", flow.get("bwd_pkts_tot", 0)))
    bytes_to_srv = int(flow.get("bytes_toserver", flow.get("fwd_data_pkts_tot", 0)))
    bytes_to_cli = int(flow.get("bytes_toclient", flow.get("bwd_data_pkts_tot", 0)))
    return (pkts_to_srv + pkts_to_cli, bytes_to_srv + bytes_to_cli)

def _append_benign_sample(event: dict):
    try:
        _merge_flow_features_into_event(event)
        flow = event.get("flow", {}) or {}
        row = {
            "timestamp": event.get("timestamp", datetime.now().isoformat()),
            "src_ip": event.get("src_ip", ""),
            "dst_ip": event.get("dest_ip", "") or event.get("dst_ip", ""),
            "src_port": int(event.get("src_port", 0) or 0),
            "dst_port": int(event.get("dest_port", 0) or event.get("dst_port", 0) or 0),
            "proto": event.get("proto", "unknown"),
            "signature": "BENIGN_FLOW",
            "flow_duration": flow.get("flow_duration", 0.0),
            "fwd_pkts_tot": flow.get("fwd_pkts_tot", flow.get("pkts_toserver", 0)),
            "bwd_pkts_tot": flow.get("bwd_pkts_tot", flow.get("pkts_toclient", 0)),
            "fwd_data_pkts_tot": flow.get("fwd_data_pkts_tot", flow.get("bytes_toserver", 0)),
            "bwd_data_pkts_tot": flow.get("bwd_data_pkts_tot", flow.get("bytes_toclient", 0)),
            "fwd_pkts_per_sec": flow.get("fwd_pkts_per_sec", 0.0),
            "bwd_pkts_per_sec": flow.get("bwd_pkts_per_sec", 0.0),
            "flow_pkts_per_sec": flow.get("flow_pkts_per_sec", 0.0),
            "down_up_ratio": flow.get("down_up_ratio", 0.0),
            "fwd_header_size_tot": flow.get("fwd_header_size_tot", 0.0),
            "fwd_header_size_min": flow.get("fwd_header_size_min", 0.0),
            "fwd_header_size_max": flow.get("fwd_header_size_max", 0.0),
            "bwd_header_size_tot": flow.get("bwd_header_size_tot", 0.0),
            "bwd_header_size_min": flow.get("bwd_header_size_min", 0.0),
            "bwd_header_size_max": flow.get("bwd_header_size_max", 0.0),
            "status": "benign",
            "direction": "inbound",
            "y_attack": 0,
            "y_reason": "benign",
        }
        append_dataset(row, 0, 0.0)
    except Exception as e:
        logger.error(f"Failed to append benign to dataset: {e}")

def handle_flow_event(data: dict):
    # update agregat per-src dari flow
    try:
        _update_aggregates_from_flow({"src_ip": data.get("src_ip")})
    except Exception:
        pass

    if not config.get("capture_benign", True):
        return
    try:
        dst_ip = data.get("dest_ip") or data.get("dst_ip")
        src_ip = data.get("src_ip")
        if not dst_ip or not src_ip:
            return
        if not _dst_in_homenet(dst_ip) or src_ip == dst_ip:
            return

        _merge_flow_features_into_event(data)
        pkts_total, bytes_total = _flow_stats(data.get("flow", {}) or {})
        if pkts_total < int(config.get("flow_min_pkts", 4)):
            return
        if bytes_total < int(config.get("flow_min_bytes", 600)):
            return

        if random.random() <= float(config.get("benign_sample_rate", 0.15)):
            _append_benign_sample(data)
    except Exception as e:
        logger.error(f"handle_flow_event error: {e}")

# ========== IMPROVED EVENT PROCESSING ==========
async def _extract_features(event: dict) -> dict:
    """Ekstrak fitur dari event dengan error handling"""
    try:
        _merge_flow_features_into_event(event)
        
        # agregat per-src saat ini
        fsec, syn_ratio, alerts_5s = _compute_src_aggregates(event["src_ip"])
        event["flows_per_sec"] = fsec
        event["syn_only_ratio"] = syn_ratio
        event["alerts_count_5s"] = alerts_5s
        
        return event
    except (KeyError, ValueError) as e:
        logger.warning(f"Feature extraction error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected feature extraction error: {e}")
        raise

async def _run_drl_prediction(event: dict) -> Tuple[int, float, Optional[str]]:
    """Jalankan prediksi DRL dengan error handling"""
    try:
        return drl_analyzer.predict(event)
    except Exception as e:
        logger.error(f"DRL prediction error: {e}")
        return 0, 0.0, None

async def _apply_blocking_policy(event: dict, drl_action: int, drl_confidence: float, attack_type: Optional[str]) -> Tuple[bool, str]:
    """Terapkan kebijakan blocking dengan error handling"""
    try:
        auto = label_flow(event)
        event["auto_label"] = auto["label"]
        event["auto_label_conf"] = auto["confidence"]
        event["label_sources"] = auto["sources"]

        y_attack, y_reason = binary_from_alert_flow_drl(
            auto, drl_action, drl_confidence, float(config.get("min_confidence", 0.70))
        )
        event["y_attack"] = y_attack
        event["y_reason"] = y_reason

        benign = _is_benign_event(event)
        conf_ok = drl_confidence >= float(config.get("min_confidence", CONFIG_DEFAULTS["min_confidence"]))
        multi_ok = _seen_enough_hits(event["src_ip"])

        srcs = set(auto.get("sources", []))
        auto_strong = (
            auto.get("label") in ATTACK_LABELS
            and float(auto.get("confidence", 0.0)) >= AUTO_BLOCK_CONF_THR
            and AUTO_BLOCK_REQUIRE_SOURCES.issubset(srcs)
        )

        # ==== Post-processing policy khusus SYN flood ====
        sig_l = (event.get("signature") or "").lower()
        synflood_hint = ("syn flood" in sig_l) or ("syn" in sig_l and "flood" in sig_l)
        syn_policy_trigger = synflood_hint and (_seen_enough_hits(event["src_ip"], window_s=2, min_hits=3)
                                                or event["alerts_count_5s"] >= 3)

        # Gating: block hanya saat monitoring ON
        monitoring_on = monitor_event is not None and monitor_event.is_set()
        should_block = False
        reason = ""

        if monitoring_on and drl_action > 0 and conf_ok:
            if auto_strong:
                should_block = True
                reason = (
                    f"DRL=BLOCK fast-track (conf {drl_confidence:.2f}); "
                    f"auto={auto['label']}@{auto['confidence']:.2f} from alert+flow"
                )
            elif (not benign) and (multi_ok or syn_policy_trigger):
                should_block = True
                why = "multi-hit OK" if multi_ok else "SYN-flood policy"
                reason = (
                    f"DRL=BLOCK (conf {drl_confidence:.2f}); auto={auto['label']}@{auto['confidence']:.2f}; {why}; "
                    f"agg(fps={event['flows_per_sec']:.2f}, syn_ratio={event['syn_only_ratio']:.2f}, alerts5s={event['alerts_count_5s']})"
                )

        if should_block:
            if blokir_ip(event["src_ip"], metode="DRL", confidence=drl_confidence, reason=reason):
                return True, reason

        return False, reason

    except Exception as e:
        logger.error(f"Blocking policy error: {e}")
        return False, f"Error: {e}"

async def _send_notification(event: dict, drl_action: int, drl_confidence: float, attack_type: Optional[str], blocked: bool, block_reason: str):
    """Kirim notifikasi dengan error handling"""
    try:
        # de-dup notifikasi 10s per (src, signature)
        key_n = (event["src_ip"], event.get("signature",""))
        now_ts = time.time()
        last = LAST_SENT.get(key_n, 0)
        allow_notify = (now_ts - last) >= 10.0
        if allow_notify:
            LAST_SENT[key_n] = now_ts
        else:
            return

        monitoring_on = monitor_event is not None and monitor_event.is_set()
        if not (config.get("enable_telegram", False) and bot is not None and monitoring_on):
            return

        if blocked:
            message = (
                f"🛑 **INBOUND ATTACK BLOCKED**\n\n"
                f"🕒 {event['timestamp']}\n"
                f"🔍 Signature: `{event.get('signature','')}`\n"
                f"📍 Attacker IP: `{event['src_ip']}`\n"
                f"🎯 Target IP: `{event.get('dst_ip', LOCAL_IP)}`\n"
                f"🤖 DRL: **BLOCK** (conf {drl_confidence:.2f})\n"
                + (f"🧭 Type (clf): `{attack_type}`\n" if attack_type else "")
                + f"📈 Agg: flows/s={event['flows_per_sec']:.2f}, syn_ratio={event['syn_only_ratio']:.2f}, alerts5s={event['alerts_count_5s']}\n"
                + f"📝 Reason: {block_reason}"
            )
            keyboard = [
                [InlineKeyboardButton("✅ Unblock IP", callback_data=f"izin_{event['src_ip']}")],
                [InlineKeyboardButton("📊 Status", callback_data="status_bot")],
            ]
        else:
            title = "🚨 **INBOUND ATTACK DETECTED**" if event.get("y_attack") == 1 else "⚠️ **Suspicious INBOUND Activity**"
            message = (
                f"{title}\n\n"
                f"🕒 {event['timestamp']}\n"
                f"🔍 Signature: `{event.get('signature','')}`\n"
                f"📍 Attacker IP: `{event['src_ip']}`\n"
                f"🎯 Target IP: `{event.get('dst_ip', LOCAL_IP)}`\n"
                f"📊 DRL Confidence: {drl_confidence:.2f}\n"
                + (f"🧭 Type (clf): `{attack_type}`\n" if attack_type else "")
                + f"📈 Agg: flows/s={event['flows_per_sec']:.2f}, syn_ratio={event['syn_only_ratio']:.2f}, alerts5s={event['alerts_count_5s']}"
            )
            keyboard = [
                [InlineKeyboardButton("🛡️ Block IP", callback_data=f"blokir_{event['src_ip']}")],
                [InlineKeyboardButton("📊 Status", callback_data="status_bot")],
            ]

        try:
            await bot.send_message(
                chat_id=config["chat_id"],
                text=message,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown",
            )
        except NetworkError:
            logger.warning("Telegram network error while sending message; continuing.")

    except Exception as e:
        logger.error(f"Notification error: {e}")

async def process_single_event(event: dict):
    """Process single event dengan error handling terpisah per komponen"""
    start_time = time.time()
    try:
        # Ekstrak fitur
        event = await _extract_features(event)
        
        # Jalankan DRL prediction
        drl_action, drl_confidence, attack_type = await _run_drl_prediction(event)
        
        # Terapkan blocking policy
        blocked, block_reason = await _apply_blocking_policy(event, drl_action, drl_confidence, attack_type)
        
        # Simpan ke dataset untuk retraining
        try:
            event["status"] = "detected"
            train_row = convert_to_training_format(event, drl_action, drl_confidence)
            append_dataset(train_row, drl_action, drl_confidence)
        except Exception as e:
            logger.error(f"Failed to append event to dataset: {e}")

        # Kirim notifikasi
        await _send_notification(event, drl_action, drl_confidence, attack_type, blocked, block_reason)

        # Record metrics
        processing_time = time.time() - start_time
        METRICS.record_processing_time(processing_time)

    except (KeyError, ValueError) as e:
        logger.warning(f"Data validation error in event processing: {e}")
    except (IOError, sqlite3.Error) as e:
        logger.error(f"System error in event processing: {e}")
    except Exception as e:
        logger.critical(f"Unexpected error in event processing: {e}")

# ========== SURICATA MONITORING ==========
def pantau_eve(async_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """
    Thread producer: tail Suricata eve.json, dorong event ke asyncio.Queue via loop.call_soon_threadsafe.
    """
    global shutdown_flag, current_log_file

    while not shutdown_flag:
        try:
            log_file = get_latest_suricata_log()
            if not log_file:
                time.sleep(1)
                continue

            current_log_file = log_file
            current_inode = os.stat(current_log_file).st_ino
            logger.info(f"Monitoring INBOUND traffic: {current_log_file}")

            with open(current_log_file, "r") as f:
                f.seek(0, 2)  # tail

                idle_sleep = int(config.get("tail_idle_sleep_ms", 150)) / 1000.0
                err_backoff = int(config.get("tail_error_backoff_ms", 500)) / 1000.0

                while not shutdown_flag:
                    try:
                        if not os.path.exists(current_log_file) or os.stat(current_log_file).st_ino != current_inode:
                            new_file = get_latest_suricata_log()
                            if new_file and new_file != current_log_file:
                                logger.info(f"Log rotated -> switching to {new_file}")
                                current_log_file = new_file
                                current_inode = os.stat(current_log_file).st_ino
                                f.close()
                                f = open(current_log_file, "r")
                                f.seek(0, 2)
                                continue
                    except FileNotFoundError:
                        time.sleep(err_backoff)
                        continue

                    line = f.readline()
                    if not line:
                        time.sleep(idle_sleep)
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    et = data.get("event_type")

                    if et == "flow":
                        try:
                            feats = _compute_flow_features(data)
                            _cache_put(data, feats)
                        except Exception:
                            pass
                        try:
                            handle_flow_event(data)
                        except Exception:
                            pass
                        continue

                    if et != "alert":
                        continue

                    dst_ip = data.get("dest_ip") or data.get("dst_ip")
                    src_ip = data.get("src_ip")

                    if not dst_ip or dst_ip != LOCAL_IP:
                        continue
                    if not data.get("alert", {}).get("signature"):
                        continue

                    # Abaikan signature tertentu
                    sig = data["alert"].get("signature", "")
                    if any(sig.startswith(x) or x in sig for x in IGNORED_SIGNATURES):
                        continue

                    _merge_flow_features_into_event(data)

                    # update agregat per-src dari alert
                    try:
                        _update_aggregates_from_alert({
                            "src_ip": src_ip,
                            "signature": sig
                        })
                    except Exception:
                        pass

                    anomali_data = {
                        "timestamp": data.get("timestamp", datetime.now().isoformat()),
                        "src_ip": data.get("src_ip"),
                        "dst_ip": dst_ip or "",
                        "src_port": data.get("src_port", 0),
                        "dest_port": data.get("dest_port", 0) or data.get("dst_port", 0),
                        "proto": data.get("proto", "unknown"),
                        "signature": data["alert"].get("signature", "Unknown"),
                        "flow": data.get("flow", {}),
                        "alert": data.get("alert", {}),
                        "direction": "inbound",
                        "community_id": data.get("community_id"),
                        "flow_id": data.get("flow_id"),
                    }

                    # dorong ke asyncio queue
                    loop.call_soon_threadsafe(async_queue.put_nowait, anomali_data)

                    simpan_log(
                        anomali_data["timestamp"],
                        anomali_data["src_ip"],
                        anomali_data["dst_ip"],
                        anomali_data["signature"],
                        "detected",
                        direction="inbound",
                    )

        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            time.sleep(1)

# ========== PERIODIC TASKS ==========
async def _periodic_cache_cleanup():
    while not shutdown_flag:
        try:
            cleaned = FLOW_CACHE.cleanup()
            if cleaned > 0:
                logger.debug(f"Cleaned {cleaned} expired cache entries")
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
        await asyncio.sleep(60)

async def _periodic_auto_unblock():
    interval = int(config.get("auto_unblock_check_interval_s", 60))
    while not shutdown_flag:
        try:
            now = datetime.now()
            with ip_lock:
                with db_connection() as conn:
                    c = conn.cursor()
                    c.execute("SELECT ip, unblock_time FROM blocked_ips")
                    rows = c.fetchall()
                    for ip, ts in rows:
                        try:
                            if ts and datetime.fromisoformat(ts) <= now:
                                _nft_del_ip(ip)
                                c.execute("DELETE FROM blocked_ips WHERE ip=?", (ip,))
                                logger.info(f"[✓] Auto-unblock IP {ip} (schedule reached)")
                        except Exception as e:
                            logger.error(f"Auto-unblock error for {ip}: {e}")
        except Exception as e:
            logger.error(f"Periodic auto-unblock loop error: {e}")
        await asyncio.sleep(interval)

async def _periodic_metrics_report():
    """Periodic metrics reporting untuk monitoring"""
    while not shutdown_flag:
        try:
            metrics = METRICS.get_metrics()
            uptime = METRICS.get_uptime()
            
            # Log metrics setiap 5 menit
            if int(uptime) % 300 == 0:  # Setiap 5 menit
                logger.info(
                    f"System Metrics - "
                    f"Uptime: {uptime:.0f}s, "
                    f"Events: {metrics.events_processed}, "
                    f"Queue: {metrics.queue_size}, "
                    f"Memory: {metrics.memory_usage_mb:.1f}MB, "
                    f"Cache: {metrics.cache_size} (hit rate: {metrics.cache_hit_rate:.1%})"
                )
        except Exception as e:
            logger.error(f"Metrics reporting error: {e}")
        
        await asyncio.sleep(60)

# ========== IMPROVED RETRAINING ==========
async def train_with_progress(df: pd.DataFrame, existing_model: Optional[PPO], update: Update) -> str:
    """Training dengan progress reporting"""
    try:
        # Validasi dataset
        if len(df) < 100:
            raise ValueError("Dataset terlalu kecil (<100 samples) untuk training efektif")
        
        # Progressive training - fine tune dari model existing jika ada
        if existing_model:
            logger.info("Using existing model for fine-tuning")
        
        # Training dengan konfigurasi yang sesuai
        out_dir = "models"
        model_name = "ppo_anomali_latest.zip"
        steps = 600_000

        def _do_train():
            return train_ppo(
                train_csv="anomali.csv",  # Gunakan file yang sudah dipersiapkan
                eval_csv=None,
                total_timesteps=steps,
                out_dir=out_dir,
                model_name_latest=model_name,
                seed=42,
                scaler_path=config.get("scaler_path", "models/minmax.pkl"),
            )

        # Jalankan training di thread terpisah
        latest_model_path = await asyncio.to_thread(_do_train)
        return latest_model_path

    except Exception as e:
        logger.error(f"Training error: {e}")
        raise

# ========== MAIN APP ==========
async def run_application():
    global application, bot, shutdown_flag, drl_analyzer, processed_events, current_log_file, LOCAL_IP, config, monitor_event

    shutdown_flag = False
    processed_events = {}
    config = load_config()
    LOCAL_IP = get_local_ip()
    init_db()
    init_dataset()

    ensure_nftables()

    # init event gating
    monitor_event = asyncio.Event()
    monitor_event.clear()  # monitoring OFF sampai user tekan Start

    try:
        drl_analyzer = DRLAnalyzer(config["drl_model_path"])
    except Exception as e:
        logger.critical(f"Failed to initialize DRL analyzer: {e}")
        return

    loop = asyncio.get_running_loop()
    monitoring_queue: asyncio.Queue = asyncio.Queue()

    if config.get("enable_telegram", True):
        try:
            req = HTTPXRequest(
                connect_timeout=10.0,
                read_timeout=35.0,
                write_timeout=10.0,
                pool_timeout=5.0,
            )
            application = Application.builder().token(config["token"]).request(req).build()
            bot = application.bot
            application.add_handler(CommandHandler("start", start))
            application.add_handler(CommandHandler("status", cek_status))
            application.add_handler(CallbackQueryHandler(handle_callback))
            application.add_error_handler(error_handler)
        except Exception as e:
            logger.warning(f"Failed to initialize Telegram bot, switching to headless: {e}")
            application = None
            bot = None
            config["enable_telegram"] = False
    else:
        application = None
        bot = None

    # Producer thread (tail Suricata)
    monitoring_thread = Thread(target=pantau_eve, args=(monitoring_queue, loop,), daemon=True)
    monitoring_thread.start()

    async def process_queue():
        while not shutdown_flag:
            try:
                # Update queue size metric
                METRICS.update_queue_size(monitoring_queue.qsize())
                
                anomali_data = await monitoring_queue.get()
                if anomali_data is None:
                    break
                
                # Process event dengan error handling terpisah
                await process_single_event(anomali_data)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue processing error: {e}")

    # Buat tasks
    process_task = asyncio.create_task(process_queue())
    cache_cleanup_task = asyncio.create_task(_periodic_cache_cleanup())
    unblock_task = asyncio.create_task(_periodic_auto_unblock())
    metrics_task = asyncio.create_task(_periodic_metrics_report())

    logger.info("[*] Starting INBOUND anomaly detection system...")

    app_started = False
    try:
        if config.get("enable_telegram", False) and application is not None:
            await application.initialize()
            await application.start()
            try:
                await application.updater.start_polling(
                    drop_pending_updates=True,
                    allowed_updates=Update.ALL_TYPES,
                    poll_interval=1.0,
                )
            except NetworkError:
                logger.warning("Telegram polling failed to start (network). Bot will keep running headless.")
            app_started = True

            # Kirim welcome (monitoring masih OFF)
            try:
                await send_welcome_message()
            except Exception:
                logger.debug("Welcome message failed; continuing.")

            while not shutdown_flag:
                await asyncio.sleep(1)

        else:
            while not shutdown_flag:
                await asyncio.sleep(1)

    except Exception as e:
        logger.critical(f"Error running application: {e}")

    finally:
        logger.info("Shutting down...")

        try:
            await monitoring_queue.put(None)
        except Exception:
            pass

        # Cancel semua tasks
        tasks = [process_task, cache_cleanup_task, unblock_task, metrics_task]
        for t in tasks:
            if not t.done():
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

        try:
            if app_started and application is not None:
                if getattr(application, "updater", None) is not None:
                    try:
                        await application.updater.stop()
                    except NetworkError:
                        pass
                await application.stop()
                await application.shutdown()
        except Exception as e:
            logger.error(f"Error during Telegram shutdown: {e}")

        try:
            await asyncio.get_running_loop().shutdown_default_executor()
        except Exception:
            pass

        logger.info("[*] System shutdown complete")

# ==== Telegram handlers ====
async def send_welcome_message():
    if not config.get("enable_telegram", False) or bot is None:
        return
    try:
        keyboard = [
            [InlineKeyboardButton("🔄 Update System", callback_data="update_system")],
            [InlineKeyboardButton("🚀 Start Monitoring", callback_data="start_bot")],
        ]
        await bot.send_message(
            chat_id=config["chat_id"],
            text=("🛡️ *Inbound Anomaly Detection Bot* is now active!\n\n"
                  "🔄 Tekan *Update System* untuk retrain model,\n"
                  "atau langsung *Start Monitoring*.\n\n"
                  "_Catatan: Notifikasi baru dikirim setelah Start Monitoring._"),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
    except NetworkError:
        logger.warning("Telegram network error while sending welcome; continuing.")
    except Exception as e:
        logger.error(f"Failed to send welcome message: {e}")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error {context.error}")
    if not config.get("enable_telegram", False) or context is None:
        return
    try:
        chat_id = update.effective_chat.id if update and update.effective_chat else config["chat_id"]
        await context.bot.send_message(
            chat_id=chat_id,
            text="⚠️ An error occurred. Please try again later.",
        )
    except NetworkError:
        logger.warning("Telegram network error in error_handler; continuing.")
    except Exception as e:
        logger.error(f"Error sending error message: {e}")

def _build_blocked_keyboard(blocked_ips: List[str]) -> InlineKeyboardMarkup:
    keyboard = [[InlineKeyboardButton(f"✅ Unblock {ip}", callback_data=f"izin_{ip}")] for ip in blocked_ips]
    keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="start_bot")])
    return InlineKeyboardMarkup(keyboard)

async def _refresh_blocked_buttons(query):
    try:
        with db_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT ip FROM blocked_ips ORDER BY timestamp DESC")
            rows = c.fetchall()
            blocked_ips = [r[0] for r in rows]
        markup = _build_blocked_keyboard(blocked_ips)
        try:
            await query.edit_message_reply_markup(reply_markup=markup)
        except BadRequest as e:
            if "message is not modified" in str(e).lower():
                await query.answer("List already up to date ✅", show_alert=False)
            else:
                raise
    except Exception as e:
        logger.error(f"Error refreshing blocked buttons: {e}")

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global monitor_event
    query = update.callback_query
    data = query.data
    await query.answer()
    try:
        if data == "update_system":
            await run_retraining(update, context)
        elif data == "start_bot":
            if monitor_event is not None:
                monitor_event.set()   # aktifkan monitoring
            await start(update, context)
        elif data == "stop_bot":
            if monitor_event is not None:
                monitor_event.clear() # matikan monitoring (notifikasi berhenti)
            await query.edit_message_text(
                "⏹️ Monitoring stopped. Tekan *Start Monitoring* untuk menyalakan lagi.",
                parse_mode="Markdown"
            )
        elif data == "status_bot":
            await cek_status(update, context)
        elif data == "list_blocked":
            await list_blocked_ips(update, context)
        elif data.startswith("blokir_"):
            ip = data.split("_", 1)[1]
            if blokir_ip(ip):
                await query.answer(f"Blocked {ip} ✅", show_alert=False)
                await _refresh_blocked_buttons(query)
            else:
                await query.answer(f"Gagal block {ip}.", show_alert=True)
        elif data.startswith("izin_"):
            ip = data.split("_", 1)[1]
            if izinkan_ip(ip):
                await query.answer(f"Unblocked {ip} ✅", show_alert=False)
                await _refresh_blocked_buttons(query)
            else:
                await query.answer(f"{ip} tidak terblokir / gagal.", show_alert=True)
    except Exception as e:
        logger.error(f"Error in callback handler: {e}")
        await error_handler(update, context)

async def run_retraining(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("⏳ Retraining model… Mohon tunggu. Bot tetap aktif selama proses.")

    try:
        csv_path = "anomali.csv"
        if not os.path.exists(csv_path):
            await query.edit_message_text("❌ Retraining gagal: `anomali.csv` tidak ditemukan. Jalankan dulu sistem agar file terbentuk.")
            return

        required = [
            "timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "proto",
            "fwd_pkts_tot", "bwd_pkts_tot", "fwd_data_pkts_tot", "bwd_data_pkts_tot",
            "signature",
        ]
        try:
            df = pd.read_csv(csv_path, usecols=lambda c: True)
        except Exception as e:
            await query.edit_message_text(f"❌ Retraining gagal: tidak bisa membaca CSV `{csv_path}`: {e}")
            return

        missing = [c for c in required if c not in df.columns]
        if missing:
            await query.edit_message_text(f"❌ Retraining gagal: kolom hilang: {missing}")
            return

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
            if df.empty:
                await query.edit_message_text("⚠️ Retraining dibatalkan: tidak ada baris dengan timestamp valid.")
                return

        # Coba load model existing untuk fine-tuning
        try:
            existing_model = PPO.load(config["drl_model_path"])
        except Exception:
            existing_model = None

        # Jalankan training dengan progress
        latest_model_path = await train_with_progress(df, existing_model, update)

        try:
            drl_analyzer.model = PPO.load(latest_model_path)
            config["drl_model_path"] = latest_model_path
            logger.info(f"Model reloaded successfully: {latest_model_path}")
        except Exception as e:
            logger.warning(f"Selesai train tapi gagal reload model ke runtime: {e}")

        keyboard = [[InlineKeyboardButton("🚀 Start Monitoring", callback_data="start_bot")]]
        await query.edit_message_text(
            text=(f"✅ Retraining selesai.\n"
                  f"Model: `{latest_model_path}`\n"
                  f"Scaler: `{config.get('scaler_path','models/minmax.pkl')}`\n\n"
                  f"Tekan *Start Monitoring* untuk lanjut."),
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )

    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        await query.edit_message_text("❌ Retraining gagal. Lihat log untuk detail.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        keyboard = [
            [InlineKeyboardButton("📊 Status", callback_data="status_bot")],
            [InlineKeyboardButton("🛡️ Blocked IPs", callback_data="list_blocked")],
            [InlineKeyboardButton("⏹️ Stop Monitoring", callback_data="stop_bot")],
        ]
        message = (
            f"🛡️ **Inbound Traffic Monitoring**\n"
            f"🖥️ Local IP: `{LOCAL_IP}`\n\n"
            "Monitoring all INBOUND network activities.\n"
            "_Notifikasi aktif karena kamu menekan Start Monitoring._"
        )
        if hasattr(update, "callback_query"):
            await update.callback_query.edit_message_text(
                text=message, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown"
            )
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=message,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown",
            )
    except Exception as e:
        logger.error(f"Error in start handler: {e}")
        await error_handler(update, context)

async def cek_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        with db_connection() as conn:
            c = conn.cursor()
            stats = "• No attack data available"
            blocked_count = 0
            try:
                c.execute("SELECT jenis_serangan, COUNT(*) FROM log GROUP BY jenis_serangan")
                stats = "\n".join([f"• {row[0]}: {row[1]} events" for row in c.fetchall()])
                c.execute("SELECT COUNT(*) FROM blocked_ips")
                blocked_result = c.fetchone()
                blocked_count = blocked_result[0] if blocked_result else 0
            except sqlite3.Error as e:
                logger.error(f"Database error in cek_status: {e}")
                stats = "• Error retrieving statistics"
                blocked_count = "N/A"
        
        # Dapatkan metrics system
        metrics = METRICS.get_metrics()
        uptime = METRICS.get_uptime()
        
        status_msg = (
            f"📊 **System Status**\n\n"
            f"⏱️ Uptime: {uptime:.0f} detik\n"
            f"🔒 Blocked IPs: {blocked_count}\n"
            f"📈 Events Processed: {metrics.events_processed}\n"
            f"💾 Memory Usage: {metrics.memory_usage_mb:.1f} MB\n"
            f"📚 Cache Hit Rate: {metrics.cache_hit_rate:.1%}\n"
            f"📊 Event Statistics:\n{stats}\n"
        )
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="start_bot")]]
        if hasattr(update, "callback_query"):
            await update.callback_query.edit_message_text(
                text=status_msg, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown"
            )
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=status_msg,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown",
            )
    except Exception as e:
        logger.error(f"Error in cek_status: {e}")
        await error_handler(update, context)

async def list_blocked_ips(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        with db_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT ip FROM blocked_ips ORDER BY timestamp DESC")
            rows = c.fetchall()
            blocked_ips = [r[0] for r in rows]
        chat_id = update.effective_chat.id if update.effective_chat else config["chat_id"]
        if not blocked_ips:
            msg = "🔒 No blocked IPs"
            keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="start_bot")]]
            if hasattr(update, "callback_query"):
                try:
                    await update.callback_query.edit_message_text(
                        text=msg, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown"
                    )
                except BadRequest as e:
                    if "message is not modified" in str(e).lower():
                        await update.callback_query.answer("List already empty ✅", show_alert=False)
                    else:
                        raise
            else:
                await context.bot.send_message(
                    chat_id=chat_id, text=msg, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown"
                )
            return
        reply_markup = _build_blocked_keyboard(blocked_ips)
        message_text = "🛡️ Blocked IPs"
        if hasattr(update, "callback_query"):
            try:
                await update.callback_query.edit_message_text(
                    text=message_text, reply_markup=reply_markup, parse_mode="Markdown"
                )
            except BadRequest as e:
                if "message is not modified" in str(e).lower():
                    await update.callback_query.edit_message_reply_markup(reply_markup=reply_markup)
                else:
                    raise
        else:
            await context.bot.send_message(
                chat_id=chat_id, text=message_text, reply_markup=reply_markup, parse_mode="Markdown"
            )
    except Exception as e:
        logger.error(f"Error in list_blocked_ips: {e}")
        await error_handler(update, context)

async def kirim_notifikasi(anomali_data):
    # Tidak dipakai lagi; notifikasi terjadi di process_single_event() dengan gating monitor_event
    pass

def shutdown_handler(signum, frame):
    global shutdown_flag
    logger.info("\n🛑 Received shutdown signal...")
    shutdown_flag = True

if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    try:
        asyncio.run(run_application())
    except Exception as e:
        logger.critical(f"[!] Fatal error: {e}")
