#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stream_eval.py — Tail (run-until-stopped) Suricata eve.json dan tulis dataset evaluasi ke CSV.

Fitur utama:
- Baca eve-YYYY-MM-DD.json (berbasis Asia/Jakarta untuk penamaan file harian) atau fallback ke /var/log/suricata/eve.json.
- Tulis baris CSV dengan skema yang kompatibel dengan env_anomali.py (env akan membangun 17 fitur dari kolom-kolom ini).
- Kolom 'dest_port' diset sama dengan 'dst_port' agar env tidak perlu fallback.
- *_data_pkts_tot diisi BYTES (sengaja; selaras dengan runtime/env).
- Label otomatis:
  * Jika ada alert → mapping (scan, bruteforce, dos, dst).
  * Opsional override berdasarkan --attacker-ips dan/atau --time-windows.
  * Kolom biner y_attack: 0 untuk 'benign', 1 untuk label selainnya.
- Default hanya menulis event_type=flow (dataset bersih). Gunakan --all-events bila ingin menulis semua event.
- Stop dengan Ctrl+C.

Contoh:
  python3 stream_eval.py -o eval.csv
  python3 stream_eval.py --log /var/log/suricata/eve-2025-10-17.json -o eval.csv
  python3 stream_eval.py --attacker-ips 192.168.38.200 --time-windows "2025-10-18 04:00:00:2025-10-18 04:10:00" --local-tz Asia/Jakarta -o eval.csv
"""

import os
import time
import json
import csv
import math
import argparse
from datetime import datetime, timezone, timedelta

from dateutil import parser as dateparser
try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # fallback nanti ke UTC bila tidak tersedia

# ==============================
# Skema CSV (kompatibel env_anomali)
# ==============================
CSV_COLS = [
    "timestamp","event_type","src_ip","dst_ip","src_port","dst_port","dest_port","proto",
    "flow_duration","fwd_pkts_tot","bwd_pkts_tot","fwd_data_pkts_tot","bwd_data_pkts_tot",
    "fwd_pkts_per_sec","bwd_pkts_per_sec","flow_pkts_per_sec","down_up_ratio",
    "signature","auto_label","auto_label_conf","label_sources","y_attack"
]

# ==============================
# Helper umum
# ==============================
def first_not_none(*vals):
    """Ambil nilai pertama yang bukan None (boleh 0, '' tetap dianggap sah)."""
    for v in vals:
        if v is not None:
            return v
    return None

# ==============================
# Helper waktu & angka
# ==============================
def _dt(ts):
    try:
        return dateparser.parse(ts)
    except Exception:
        return None

def _sec(a, b, fallback=None):
    return max((b - a).total_seconds(), 0.0) if a and b else (fallback if fallback is not None else 0.0)

def _safe(x, d=0.0):
    try:
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return float(d)
        return xf
    except Exception:
        return float(d)

def _zone_or_utc(tz_name: str):
    """Ambil ZoneInfo(tz_name) jika tersedia, else UTC."""
    if ZoneInfo is not None:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            pass
    return timezone.utc

# ==============================
# Flow feature extraction (ringkas & robust)
# ==============================
def compute_flow_features(ev: dict) -> dict:
    """
    Ekstrak fitur inti flow; robust untuk variasi field top-level/flow.
    Catatan:
      - *_data_pkts_tot diisi BYTES (bukan 'packet count'), ini SENGAJA agar cocok dengan env/runtime kamu.
      - *_pkts_tot diisi PKTS.
    """
    flow = ev.get("flow", {}) or {}

    # Durasi: pakai flow.start/end → fallback ke timestamp event
    start = flow.get("start") or ev.get("flow_start")
    end   = flow.get("end")   or ev.get("flow_end")
    ts_start = _dt(start) or _dt(ev.get("timestamp"))
    ts_end   = _dt(end)   or _dt(ev.get("timestamp"))
    dur = _sec(ts_start, ts_end, 0.0)

    # pkts & bytes (BYTES untuk *_data_pkts_tot)
    try:
        fwd_pkts  = int(flow.get("pkts_toserver", flow.get("fwd_pkts_tot", 0)) or 0)
    except Exception:
        fwd_pkts = 0
    try:
        bwd_pkts  = int(flow.get("pkts_toclient", flow.get("bwd_pkts_tot", 0)) or 0)
    except Exception:
        bwd_pkts = 0
    try:
        fwd_bytes = int(flow.get("bytes_toserver", flow.get("fwd_data_pkts_tot", 0)) or 0)
    except Exception:
        fwd_bytes = 0
    try:
        bwd_bytes = int(flow.get("bytes_toclient", flow.get("bwd_data_pkts_tot", 0)) or 0)
    except Exception:
        bwd_bytes = 0

    denom = dur if dur > 0 else 1.0
    fwd_pps  = _safe(fwd_pkts / denom, 0.0)
    bwd_pps  = _safe(bwd_pkts / denom, 0.0)
    flow_pps = _safe((fwd_pkts + bwd_pkts) / denom, 0.0)
    down_up  = _safe(bwd_pkts / max(1, fwd_pkts), 0.0)

    return {
        "flow_duration": dur,
        "fwd_pkts_tot": fwd_pkts,
        "bwd_pkts_tot": bwd_pkts,
        "fwd_data_pkts_tot": fwd_bytes,
        "bwd_data_pkts_tot": bwd_bytes,
        "fwd_pkts_per_sec": fwd_pps,
        "bwd_pkts_per_sec": bwd_pps,
        "flow_pkts_per_sec": flow_pps,
        "down_up_ratio": down_up,
    }

# ==============================
# Tambahkan di atas (global)
INFO_SIG_IDS_ALLOW = {2016149, 2016150}  # ET INFO STUN Binding Req/Resp
INFO_CATEGORIES_ALLOW = {"misc activity"}  # kategori sering muncul untuk INFO
INFO_PREFIXES = ("et info ",)  # signature yg diawali ini dianggap informasional

ALERT_MAP = {
    "ssh brute force": "bruteforce",
    "bruteforce": "bruteforce",
    "dos": "dos",
    "ddos": "dos",
    "scan": "scan",
    "port scan": "scan",
}

def label_from_alert(alert: dict):
    """
    Mengembalikan:
      - {'label': 'benign_info', 'score': 0.2, 'src': 'alert-info'} untuk alert informasional (severity=3, prefix "ET INFO", kategori allowlist, atau signature_id allowlist)
      - {'label': <mapped>, 'score': ~0.9, 'src': 'alert'} untuk alert yang match ke mapping (scan/bruteforce/dos)
      - {'label': 'anomaly', 'score': 0.80, 'src': 'alert'} untuk alert selain itu (opsional)
      - None jika tidak ada alert
    """
    if not alert:
        return None

    # Normalisasi teks
    sig = (alert.get("signature") or "").strip()
    cat = (alert.get("category") or "").strip()
    sev = int(alert.get("severity", 3))

    sig_lower = sig.lower()
    cat_lower = cat.lower()
    sid = int(alert.get("signature_id", 0))

    # 1) INFO allowlist (STUN/WebRTC, dsb.) -> anggap benign_info
    if (sid in INFO_SIG_IDS_ALLOW) or \
       (sev == 3) or \
       (cat_lower in INFO_CATEGORIES_ALLOW) or \
       sig_lower.startswith(INFO_PREFIXES):
        return {"label": "benign_info", "score": 0.20, "src": "alert-info"}

    # 2) Mapping eksplisit (scan/bruteforce/dos)
    text = f"{sig_lower} {cat_lower}"
    for k, v in ALERT_MAP.items():
        if k in text:
            # bobot sedikit dipengaruhi severity (1>2)
            score = {1: 0.96, 2: 0.92}.get(sev, 0.90)
            return {"label": v, "score": score, "src": "alert"}

    # 3) Alert lain yang tidak jelas -> opsional: tandai anomaly ringan
    return {"label": "anomaly", "score": 0.80, "src": "alert"}


# ==============================
# Tail file (handle truncate/rotate)
# ==============================
def follow(path: str, poll_interval=0.25):
    """
    Tail -F sederhana:
    - Tunggu file jika belum ada.
    - Deteksi pergantian inode (rotate) dan re-open.
    - Deteksi truncate.
    """
    while not os.path.exists(path):
        print(f"[INFO] Waiting for {path}")
        time.sleep(1.0)

    f = open(path, "r", encoding="utf-8", errors="ignore")
    f.seek(0, os.SEEK_END)
    inode = os.fstat(f.fileno()).st_ino
    try:
        while True:
            line = f.readline()
            if line:
                yield line
            else:
                try:
                    st = os.stat(path)
                    if st.st_ino != inode:
                        # rotate/recreate
                        f.close()
                        f = open(path, "r", encoding="utf-8", errors="ignore")
                        inode = os.fstat(f.fileno()).st_ino
                        f.seek(0, os.SEEK_END)
                    elif f.tell() > st.st_size:
                        # truncate
                        f.seek(0, os.SEEK_END)
                except FileNotFoundError:
                    # hilang sebentar, tunggu muncul lagi
                    while not os.path.exists(path):
                        time.sleep(0.5)
                    f = open(path, "r", encoding="utf-8", errors="ignore")
                    inode = os.fstat(f.fileno()).st_ino
                    f.seek(0, os.SEEK_END)
                time.sleep(poll_interval)
    finally:
        f.close()

# ==============================
# CSV Writer
# ==============================
def open_csv(path: str):
    """Buka CSV untuk append; tulis header jika file baru/kosong."""
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(CSV_COLS)
        f.flush()
    return f, w

# ==============================
# Path eve hari ini (Asia/Jakarta) + fallback
# ==============================
def get_today_eve(path_pattern="/var/log/suricata/eve-%Y-%m-%d.json"):
    try:
        path = datetime.now(timezone(timedelta(hours=7))).strftime(path_pattern)
    except Exception:
        path = path_pattern
    if os.path.exists(path):
        return path
    fallback = "/var/log/suricata/eve.json"
    return fallback if os.path.exists(fallback) else path

# ==============================
# Utilities: parse time-windows arg => list of (start_epoch, end_epoch)
# Accept formats:
#  - epoch integer (seconds)
#  - ISO timestamp (e.g., 2025-10-14T17:58:38Z or 2025-10-14 17:58:38)
#  - Tanpa zona → diasumsikan ke --local-tz (default Asia/Jakarta)
# ==============================
def parse_time_token(tok: str, local_tz: str):
    tok = tok.strip()
    if not tok:
        return None
    # epoch?
    try:
        return int(tok)
    except Exception:
        pass
    # ISO?
    dt = dateparser.parse(tok)
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_zone_or_utc(local_tz))
    return int(dt.timestamp())

def parse_time_windows(arg: str, local_tz: str):
    """
    arg example: "1697535600:1697535660,2025-10-14T17:58:38Z:2025-10-14T17:59:00Z"
    returns list of (start_epoch, end_epoch)
    """
    out = []
    if not arg:
        return out
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            continue
        start_s, end_s = p.split(":", 1)
        s = parse_time_token(start_s, local_tz)
        e = parse_time_token(end_s, local_tz)
        if s is not None and e is not None and e >= s:
            out.append((s, e))
    return out

def parse_attacker_ips(arg: str):
    if not arg:
        return set()
    return set([x.strip() for x in arg.split(",") if x.strip()])

def in_attack_window(ts_epoch: int, windows):
    for s, e in windows:
        if s <= ts_epoch <= e:
            return True
    return False

def parse_ts_to_epoch(s: str) -> int:
    """
    Parse Suricata timestamp (ISO-like) to epoch seconds.
    If fails, return 0.
    """
    if not s:
        return 0
    try:
        dt = dateparser.parse(s)
        if dt is None:
            return 0
        if dt.tzinfo is None:
            # default as UTC if event ts no tz (biasanya Suricata pakai Z)
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return 0

# ==============================
# Main
# ==============================
def main():
    ap = argparse.ArgumentParser(
        description="Stream Suricata eve.json → eval.csv (with optional attacker IP/time-window labeling)"
    )
    ap.add_argument("--log", default=None,
                    help="Path ke eve.json (default: eve-YYYY-MM-DD.json atau fallback eve.json)")
    ap.add_argument("-o", "--out", default="evaluation.csv", help="Path output CSV (default: evaluation.csv)")
    ap.add_argument("--attacker-ips", default="",
                    help="Comma-separated attacker IPs (opsional). e.g. 192.168.38.200,10.0.0.5")
    ap.add_argument("--time-windows", default="",
                    help=("Comma-separated windows start:end (epoch/ISO). "
                          "Contoh: 1697535600:1697535660,2025-10-14T17:58:38Z:2025-10-14T17:59:00Z"))
    ap.add_argument("--local-tz", default="Asia/Jakarta",
                    help="TZ untuk parse time-windows tanpa zona (default: Asia/Jakarta)")
    ap.add_argument("--all-events", action="store_true", default=False,
                    help="Jika disetel, tulis SEMUA event (default: hanya event_type=flow).")

    args = ap.parse_args()

    eve_path = args.log if args.log else get_today_eve()
    print(f"[INFO] tailing: {eve_path}")
    print(f"[INFO] writing : {args.out}")
    print("[INFO] stop dengan Ctrl+C")

    # parse attacker metadata
    attacker_ips = parse_attacker_ips(args.attacker_ips)
    time_windows = parse_time_windows(args.time_windows, args.local_tz)
    if attacker_ips:
        print(f"[INFO] attacker IPs      : {attacker_ips}")
    if time_windows:
        print(f"[INFO] time windows(epoch): {time_windows}")

    out_f, writer = open_csv(args.out)
    written = 0

    try:
        for line in follow(eve_path):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                # skip non-json
                continue

            et = ev.get("event_type", "")
            if (not args.all_events) and et != "flow":
                # default: hanya simpan flow agar dataset bersih
                continue

            alert = ev.get("alert", {}) or {}
            sig = alert.get("signature")

            # fitur flow (robust)
            feats = compute_flow_features(ev)

            # label otomatis sederhana (dari alert)
            auto = label_from_alert(alert)
            if not auto:
                auto_label, auto_conf, srcs = "benign", 0.50, "flow"
            else:
                auto_label, auto_conf, srcs = auto["label"], auto["score"], auto["src"]

            # timestamp epoch untuk cek windows
            ev_ts_epoch = parse_ts_to_epoch(ev.get("timestamp", ""))

            # tambahan labeling berdasarkan attacker IP dan/atau time windows
            # logika fleksibel:
            # - jika attacker_ips kosong → ip_match=True (abaikan filter IP)
            # - jika time_windows kosong → time_ok=True (abaikan filter waktu)
            # - jika KEDUANYA kosong → tidak ada override
            override_possible = bool(attacker_ips or time_windows)
            if override_possible:
                src_ip = ev.get("src_ip", "")
                dst_ip = first_not_none(ev.get("dest_ip"), ev.get("dst_ip")) or ""
                ip_match = ((src_ip in attacker_ips) or (dst_ip in attacker_ips)) if attacker_ips else True
                time_ok  = in_attack_window(ev_ts_epoch, time_windows) if time_windows else True
                if ip_match and time_ok:
                    auto_label, auto_conf, srcs = "attacker_window", 0.95, "attacker_ip_window"

            # biner (opsional, memudahkan env bila prioritas label diubah)
            y_attack = 0 if str(auto_label).lower() == "benign" else 1

            # normalisasi ip/port tujuan (hindari or yang salah pada nilai 0)
            dst_port_val = first_not_none(ev.get("dest_port"), ev.get("dst_port"))
            dst_ip_val   = first_not_none(ev.get("dest_ip"), ev.get("dst_ip"))

            row = [
                ev.get("timestamp"), et,
                ev.get("src_ip"),
                dst_ip_val,
                ev.get("src_port"),
                dst_port_val,           # dst_port
                dst_port_val,           # dest_port (disamakan agar env bebas fallback)
                ev.get("proto"),

                feats["flow_duration"],
                feats["fwd_pkts_tot"],
                feats["bwd_pkts_tot"],
                feats["fwd_data_pkts_tot"],
                feats["bwd_data_pkts_tot"],
                feats["fwd_pkts_per_sec"],
                feats["bwd_pkts_per_sec"],
                feats["flow_pkts_per_sec"],
                feats["down_up_ratio"],

                sig,
                auto_label,
                auto_conf,
                srcs,
                y_attack
            ]

            writer.writerow(row)
            written += 1
            # flush berkala
            if written % 50 == 0:
                out_f.flush()

    except KeyboardInterrupt:
        pass
    finally:
        out_f.flush()
        out_f.close()
        print(f"\n[OK] wrote {written} rows to {args.out}")

if __name__ == "__main__":
    main()
