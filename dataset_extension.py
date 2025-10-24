# dataset_extension.py
import csv
import os
from datetime import datetime
from typing import List

# Kolom dasar (sesuai versi lama)
COLUMNS_BASE: List[str] = [
    'timestamp','src_ip','dst_ip','src_port','dst_port','proto',
    'flow_duration','fwd_pkts_tot','bwd_pkts_tot','fwd_data_pkts_tot','bwd_data_pkts_tot',
    'fwd_pkts_per_sec','bwd_pkts_per_sec','flow_pkts_per_sec','down_up_ratio',
    'fwd_header_size_tot','fwd_header_size_min','fwd_header_size_max',
    'bwd_header_size_tot','bwd_header_size_min','bwd_header_size_max',
    'signature','action','confidence','status','direction'
]

# Kolom opsional baru (label gabungan alert+flow + label biner)
OPTIONAL_COLUMNS: List[str] = [
    'auto_label',         # string kelas otomatis (dos/bruteforce/scan/anomaly/benign)
    'auto_label_conf',    # confidence float 0..1
    'label_sources',      # sumber label: "alert", "flow", atau "alert;flow"
    'y_attack',           # 0/1 biner: normal(0) vs anomali/serangan(1)
    'y_reason',           # asal keputusan biner: 'drl' | 'auto' | 'none'
]

# Urutan kolom kanonik yang kita inginkan di file dataset
COLUMNS: List[str] = COLUMNS_BASE + OPTIONAL_COLUMNS

DATASET_PATH = "anomali.csv"


def _read_header(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            return header
    except Exception:
        return None


def _migrate_header(path: str, new_columns: List[str]):
    """
    Migrasi file CSV agar mengikuti header 'new_columns'.
    Kolom lama dipertahankan, kolom baru diisi '' (string kosong).
    Urutan kolom akan disesuaikan ke 'new_columns'.
    """
    tmp_path = path + ".tmp"
    try:
        with open(path, 'r', newline='', encoding='utf-8') as fin, \
             open(tmp_path, 'w', newline='', encoding='utf-8') as fout:
            reader = csv.DictReader(fin)
            writer = csv.DictWriter(fout, fieldnames=new_columns, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for row in reader:
                out = {col: row.get(col, '') for col in new_columns}
                writer.writerow(out)
        # replace atomically
        os.replace(tmp_path, path)
    except Exception:
        # jika migrasi gagal, jangan tinggalkan tmp yang rusak
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except Exception: pass
        raise


def _ensure_header(path: str, desired_columns: List[str]) -> List[str]:
    """
    Pastikan file ada dan header mencakup semua 'desired_columns'.
    - Jika file belum ada: buat dengan header kanonik 'desired_columns'.
    - Jika file ada dan header berbeda/kurang kolom: migrasi ke header kanonik.
    Return: header final yang dipakai file (list kolom).
    """
    header = _read_header(path)
    if header is None:
        # buat baru
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=desired_columns, quoting=csv.QUOTE_MINIMAL)
            w.writeheader()
        return desired_columns

    # Sudah ada file: cek apakah semua kolom kanonik sudah ada
    header_set = set(header)
    desired_set = set(desired_columns)
    if header != desired_columns or not desired_set.issubset(header_set):
        # Migrasikan ke urutan & superset kanonik
        _migrate_header(path, desired_columns)
        return desired_columns

    # Header sudah sesuai
    return header


def init_dataset(path: str = DATASET_PATH):
    _ensure_header(path, COLUMNS)


def _coerce_scalar(v):
    # Normalisasi nilai untuk CSV
    if v is None:
        return ''
    # Ubah list/set/tuple -> string join dengan ';'
    if isinstance(v, (list, set, tuple)):
        try:
            return ';'.join(str(x) for x in v)
        except Exception:
            return str(v)
    return v


def append_dataset(row: dict, action: int, confidence: float, path: str = DATASET_PATH):
    # Pastikan header up-to-date (akan buat/migrasi bila perlu)
    columns = _ensure_header(path, COLUMNS)

    # normalisasi nilai + sanitasi ringan
    row = dict(row) if row else {}
    row.setdefault('timestamp', datetime.now().isoformat())

    # action & confidence dari argumen → override
    row['action'] = int(action)
    row['confidence'] = float(confidence)

    # hilangkan newline di signature agar tidak memecah baris
    sig = str(row.get('signature', '')) if row.get('signature') is not None else ''
    row['signature'] = sig.replace('\r', ' ').replace('\n', ' ')

    # label_sources bisa berupa list -> jadikan "alert;flow"
    if 'label_sources' in row:
        row['label_sources'] = _coerce_scalar(row.get('label_sources'))

    # Normalisasi kolom biner opsional
    if 'y_attack' in row:
        try:
            row['y_attack'] = int(row.get('y_attack', 0))
            row['y_attack'] = 1 if row['y_attack'] else 0
        except Exception:
            row['y_attack'] = 0
    if 'y_reason' in row and row['y_reason'] is None:
        row['y_reason'] = ''

    # pastikan semua kolom ada; yang tidak ada jadi default kosong/0
    cooked = {}
    for k in columns:
        val = row.get(k, '')
        val = _coerce_scalar(val)
        cooked[k] = val

    # tulis dengan quoting agar koma aman
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_MINIMAL)
        # header sudah dipastikan oleh _ensure_header; tidak perlu writeheader di sini
        w.writerow(cooked)
