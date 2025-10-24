# quick_stats.py (robust)
import pandas as pd
import numpy as np

DATASET = "datasets/dummy_dataset.csv"

# Fitur yang dipakai untuk threshold
FEATS = ["flow_pkts_per_sec", "fwd_pkts_tot", "bwd_pkts_tot", "flow_duration", "dst_port"]

# Mapping signature -> kelas kasar (untuk bantuan klasifikasi)
SIGMAP = {
    "scan": ["scan", "portscan", "nmap", "masscan"],
    "dos": ["dos", "denial of service", "flood", "syn flood", "udp flood"],
    "brute": ["brute", "dictionary", "guess", "hydra", "ssh brute"],
}

# ---- util ----
def to_num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0)

def qdesc(x):
    x = to_num(x)
    return {
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p98": float(np.quantile(x, 0.98)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(x.max()),
    }

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # lowercase semua nama kolom
    df.columns = [c.strip().lower() for c in df.columns]

    # alias umum
    aliases = {
        "dest_port": "dst_port",
        "destination_port": "dst_port",
        "dport": "dst_port",
        "sport": "src_port",
        "source_port": "src_port",
        "alert.signature": "signature",
        "sig": "signature",
        "signatures": "signature",
        "proto": "protocol",  # hanya alias info; kita tidak pakai di FEATS
    }
    for a, b in aliases.items():
        if a in df.columns and b not in df.columns:
            df.rename(columns={a: b}, inplace=True)

    # jika flow_pkts_per_sec belum ada tapi fwd/bwd ada, kita bisa hitung (opsional)
    if "flow_pkts_per_sec" not in df.columns and {"fwd_pkts_per_sec", "bwd_pkts_per_sec"} <= set(df.columns):
        fps = to_num(df["fwd_pkts_per_sec"]) + to_num(df["bwd_pkts_per_sec"])
        df["flow_pkts_per_sec"] = fps

    return df

def infer_status(df: pd.DataFrame) -> pd.Series:
    # prioritas: y_attack -> label/kelas/class -> signature -> unknown
    if "status" in df.columns:
        return df["status"].astype(str).str.lower()

    # 1) y_attack (0/1)
    for col in ["y_attack", "is_attack", "attack"]:
        if col in df.columns:
            ya = to_num(df[col]).astype(int)
            return ya.replace({0: "benign", 1: "attack"}).astype(str)

    # 2) label tekstual
    for col in ["label", "kelas", "class", "category"]:
        if col in df.columns:
            lab = df[col].astype(str).str.lower()
            # normalisasi beberapa varian
            lab = lab.replace(
                {
                    "normal": "benign",
                    "benign_flow": "benign",
                    "bruteforce": "brute",
                    "bruteforce-ssh": "brute",
                    "dos/hping3": "dos",
                    "attack": "attack",
                }
            )
            return lab

    # 3) turunan dari signature
    sigcol = None
    for c in ["signature", "signature_text", "alert"]:
        if c in df.columns:
            sigcol = c
            break

    if sigcol:
        def _from_sig(s):
            s = str(s) if s is not None else ""
            sl = s.lower()
            if s.upper() == "BENIGN_FLOW":
                return "benign"
            for k, kws in SIGMAP.items():
                if any(kw in sl for kw in kws):
                    # sebut 'attack' agar netral; bucket kelas tetap dari fungsi kelas()
                    return "attack"
            return "benign" if s == "" else "attack"

        return df[sigcol].map(_from_sig).astype(str)

    # 4) default
    return pd.Series(["unknown"] * len(df), index=df.index)

def kelas(sig, status):
    st = str(status).lower()
    if st == "benign":
        return "benign"
    s = str(sig or "").lower()
    # mapping signature → kelas spesifik
    for k, kws in SIGMAP.items():
        if any(kw in s for kw in kws):
            return "bruteforce" if k == "brute" else k
    # heuristik tambahan untuk brute-force SSH
    # dst_port 22 + fwd besar dan bwd kecil akan difilter kemudian
    return "unknown"

# ---- main ----
def main():
    # Baca CSV: dtype=str agar bebas DtypeWarning, dan low_memory=False
    df = pd.read_csv(DATASET, dtype=str, low_memory=False)
    df = normalize_cols(df)

    # Pastikan kolom inti minimal
    required_soft = ["signature"]  # soft-required (bisa diisi alias/None)
    for c in required_soft:
        if c not in df.columns:
            # kalau tidak ada, buat kosong (nanti status/kelas pakai fallback lain)
            df[c] = ""

    # Pastikan dst_port ada (alias sudah ditangani); kalau tetap tidak ada, buat kosong
    if "dst_port" not in df.columns:
        df["dst_port"] = ""

    # Buat/isi kolom status jika hilang
    df["status"] = infer_status(df)

    # Pastikan FEATS ada (kalau tidak ada, buat kolom kosong supaya loop tetap jalan)
    for f in FEATS:
        if f not in df.columns:
            df[f] = ""

    # Kelas spesifik (benign/scan/dos/bruteforce/unknown)
    df["cls"] = [kelas(sig, st) for sig, st in zip(df.get("signature", ""), df["status"])]

    # Filter brute-force: prioritas yang dst_port==22 (SSH)
    is_ssh = to_num(df["dst_port"]) == 22
    df_bf = df[(df["cls"].isin(["bruteforce", "unknown"])) & is_ssh]

    # Buckets
    buckets = {
        "benign": df[df["cls"] == "benign"],
        "scan": df[df["cls"] == "scan"],
        "dos": df[df["cls"] == "dos"],
        "bruteforce": df_bf if not df_bf.empty else df[df["cls"] == "bruteforce"],
    }

    print("\n=== Sample counts ===")
    for k, v in buckets.items():
        print(f"{k:11s}: {len(v)}")

    print("\n=== Quantiles per bucket ===")
    for name, sub in buckets.items():
        if sub is None or sub.empty:
            print(f"\n[{name}] (no data)")
            continue
        print(f"\n[{name}]")
        for f in FEATS:
            print(f"  {f:17s}: {qdesc(sub[f])}")

    # Saran threshold
    ben = buckets["benign"]
    if ben is not None and not ben.empty:
        ben_pps = to_num(ben["flow_pkts_per_sec"])
        ben_fwd = to_num(ben["fwd_pkts_tot"])
        DOS_PPS_THR = float(np.quantile(ben_pps, 0.99))
        DOS_FWD_THR = int(np.quantile(ben_fwd, 0.99))

        scan = buckets["scan"]
        if scan is not None and not scan.empty:
            SCAN_DUR_THR = float(np.quantile(to_num(scan["flow_duration"]), 0.60))
            SCAN_BWD_THR = int(np.quantile(to_num(scan["bwd_pkts_tot"]), 0.50))
        else:
            SCAN_DUR_THR, SCAN_BWD_THR = 1.0, 2

        bf = buckets["bruteforce"]
        if bf is not None and not bf.empty:
            BF_FWD_MIN = int(np.quantile(to_num(bf["fwd_pkts_tot"]), 0.50))
            BF_BWD_MAX = int(np.quantile(to_num(bf["bwd_pkts_tot"]), 0.50))
        else:
            BF_FWD_MIN, BF_BWD_MAX = 12, 4

        print("\n=== Suggested thresholds (paste into main.py > label_from_flow_feats) ===")
        print(f"DOS_PPS_THR   = {DOS_PPS_THR:.2f}")
        print(f"DOS_FWD_THR   = {DOS_FWD_THR}")
        print(f"SCAN_DUR_THR  = {SCAN_DUR_THR:.2f}")
        print(f"SCAN_BWD_THR  = {SCAN_BWD_THR}")
        print(f"BF_FWD_MIN    = {BF_FWD_MIN}")
        print(f"BF_BWD_MAX    = {BF_BWD_MAX}")
    else:
        print("\n[!] No benign rows found; generate benign traffic first.")

if __name__ == "__main__":
    main()
