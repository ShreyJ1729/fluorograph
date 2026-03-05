"""
Merge fpbase_proteins.csv + sequences.json + FPbase API → fpbase_merged.csv

CSV  (1 row per protein-state): Name, State, Ex max, Em max, spectral metadata
JSON (1 entry per protein):     uuid, name, seq, states[], transitions[], doi
API  (basic endpoint cache):    cofactor, bleach (fields not in CSV/JSON exports)

Output: one row per protein-state with sequence attached.
Spectral data from CSV (authoritative). Sequence from JSON. Cofactor from API.
Proteins missing sequence or both ex+em are flagged but kept.
"""

import csv
import json
from pathlib import Path

CSV_PATH = Path("fpbase_proteins.csv")
JSON_PATH = Path("sequences.json")
API_PATH = Path("api_basic_data.json")
OUT_PATH = Path("fpbase_merged.csv")


def normalize(name: str) -> str:
    return name.strip().lower()


def main():
    # Load JSON → name-keyed dict
    with open(JSON_PATH) as f:
        json_data = json.load(f)

    json_by_name = {}
    for p in json_data:
        key = normalize(p["name"])
        json_by_name[key] = p

    # Load API basic data (cofactor, bleach)
    api_by_name = {}
    if API_PATH.exists():
        with open(API_PATH) as f:
            api_data = json.load(f)
        for p in api_data:
            key = normalize(p["name"])
            api_by_name[key] = p

    # Load CSV
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    n_matched = 0
    n_no_seq = 0
    n_complete = 0  # has seq + ex_max + em_max

    out_fields = [
        "name",
        "uuid",
        "slug",
        "state",
        "seq",
        "seq_len",
        "ex_max",
        "em_max",
        "stokes_shift",
        "ext_coeff",
        "qy",
        "brightness",
        "pka",
        "oligomerization",
        "maturation",
        "lifetime",
        "mw_kda",
        "year",
        "switch_type",
        "aliases",
        "doi",
        "cofactor",
        "bleach",
        "pdb",
        "has_seq",
        "has_ex_em",
        "complete",
    ]

    for row in rows:
        name = row["Name"].strip()
        key = normalize(name)
        jp = json_by_name.get(key)

        seq = jp["seq"].strip() if (jp and jp.get("seq")) else ""
        uuid = jp["uuid"] if jp else ""
        slug = jp["slug"] if jp else ""
        doi = jp.get("doi", "") if jp else ""
        pdb_list = jp.get("pdb") if jp else None
        pdb = ",".join(pdb_list) if isinstance(pdb_list, list) else ""

        # API-sourced fields
        ap = api_by_name.get(key)
        cofactor = ap.get("cofactor", "") if ap else ""
        bleach_val = ap.get("bleach") if ap else None
        bleach = str(bleach_val) if bleach_val is not None else ""

        ex_max_str = row.get("Ex max (nm)", "").strip()
        em_max_str = row.get("Em max (nm)", "").strip()

        try:
            ex_max = float(ex_max_str) if ex_max_str else None
        except ValueError:
            ex_max = None
        try:
            em_max = float(em_max_str) if em_max_str else None
        except ValueError:
            em_max = None

        has_seq = bool(seq)
        has_ex_em = ex_max is not None and em_max is not None
        complete = has_seq and has_ex_em

        if jp:
            n_matched += 1
        if not has_seq:
            n_no_seq += 1
        if complete:
            n_complete += 1

        out_rows.append({
            "name": name,
            "uuid": uuid,
            "slug": slug,
            "state": row.get("State", "").strip(),
            "seq": seq,
            "seq_len": len(seq) if seq else "",
            "ex_max": ex_max if ex_max is not None else "",
            "em_max": em_max if em_max is not None else "",
            "stokes_shift": row.get("Stokes Shift (nm)", "").strip(),
            "ext_coeff": row.get("Extinction Coefficient", "").strip(),
            "qy": row.get("Quantum Yield", "").strip(),
            "brightness": row.get("Brightness", "").strip(),
            "pka": row.get("pKa", "").strip(),
            "oligomerization": row.get("Oligomerization", "").strip(),
            "maturation": row.get("Maturation (min)", "").strip(),
            "lifetime": row.get("Lifetime (ns)", "").strip(),
            "mw_kda": row.get("Molecular Weight (kDa)", "").strip(),
            "year": row.get("Year", "").strip(),
            "switch_type": row.get("Switch Type", "").strip(),
            "aliases": row.get("Aliases", "").strip(),
            "doi": doi,
            "cofactor": cofactor,
            "bleach": bleach,
            "pdb": pdb,
            "has_seq": int(has_seq),
            "has_ex_em": int(has_ex_em),
            "complete": int(complete),
        })

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(out_rows)

    # Cofactor stats
    n_cofactor = sum(1 for r in out_rows if r["cofactor"])
    n_cofactor_complete = sum(1 for r in out_rows if r["cofactor"] and r["complete"])

    print(f"Total rows:            {len(rows)}")
    print(f"Matched to JSON:       {n_matched}")
    print(f"Missing sequence:      {n_no_seq}")
    print(f"Complete (seq+ex+em):  {n_complete}")
    print(f"With cofactor:         {n_cofactor} ({n_cofactor_complete} complete)")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
