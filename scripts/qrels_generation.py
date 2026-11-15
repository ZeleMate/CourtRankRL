#!/usr/bin/env python3
"""
CourtRankRL Qrels Generation - Local Execution (M3 MacBook Air)

Agents.md specifikáció alapján:
- Qrels template generálása hybrid retrieval eredményeiből
- TSV format (query_id, doc_id, relevance)
- Manuális annotálásra optimalizált

Használat:
    python scripts/qrels_generation.py

Előfeltételek:
- pipeline_results.jsonl (előző szkript outputja)

Output:
- baseline_qrels.tsv (relevance=0, manuális kitöltésre vár)
"""

import os
import json
from pathlib import Path
from typing import List, Dict

print("✅ Dependencies betöltve")

# === Konfiguráció ===
BASE_PATH = Path(os.getenv("WORKSPACE_PATH", "/Users/zelenyianszkimate/Documents/CourtRankRL"))

# Input fájl (az előző szkript outputja)
INPUT_PATH = BASE_PATH / "data" / "qrels" / "pipeline_results.jsonl"

# Output fájl
OUTPUT_QRELS = BASE_PATH / "data" / "qrels" / "baseline_qrels.tsv"

print("📂 Workspace és fájlok:")
print(f"  Base path: {BASE_PATH}")
print(f"  Input: {INPUT_PATH}")
print(f"  Output: {OUTPUT_QRELS}")

# Fájl ellenőrzés
if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"❌ Input fájl nem található: {INPUT_PATH}\n"
        "   Először futtasd a 'python scripts/hybrid_retrieval.py' szkriptet!"
    )

print("\n✅ Konfiguráció kész")

# === Retrieval Eredmények Betöltése (pandas optimalizált) ===
print("\n" + "="*60)
print("📥 RETRIEVAL EREDMÉNYEK BETÖLTÉSE")
print("="*60)

results: Dict[str, List[str]] = {}

try:
    # pandas.read_json() gyorsabb mint kézi json.loads() parsing (agents.md szerint)
    import pandas as pd
    df_results = pd.read_json(INPUT_PATH, lines=True, encoding='utf-8')

    # Dict konverzió
    for _, row in df_results.iterrows():
        query = row.get('query', '')
        doc_ids = row.get('doc_ids', [])
        if query and doc_ids:
            results[query] = doc_ids

    print(f"✅ {len(results)} query betöltve")
except (ValueError, FileNotFoundError) as e:
    print(f"❌ Hiba a retrieval eredmények betöltése során: {e}")
    print("   Futtasd a hybrid_retrieval.py scriptet először!")
    sys.exit(1)

# Első néhány query kiírása
print("\nPélda query-k (első 3):")
for i, (query, doc_ids) in enumerate(list(results.items())[:3], 1):
    print(f"  {i}. {query[:60]}{'...' if len(query) > 60 else ''}")
    print(f"     → {len(doc_ids)} dokumentum")

if len(results) > 3:
    print(f"  ... és még {len(results) - 3}")

# === Qrels Template Generálása ===
print("\n" + "="*60)
print("📝 QRELS TEMPLATE GENERÁLÁSA")
print("="*60)

total_pairs = sum(len(docs) for docs in results.values())
print(f"  {len(results)} query")
print(f"  {total_pairs} query-dokumentum pár")
print()

with open(OUTPUT_QRELS, 'w', encoding='utf-8') as f:
    # Header (agents.md format)
    f.write("query_id\tdoc_id\trelevance\n")

    # Pipeline eredmények -> qrels (relevance=0)
    for query, doc_ids in sorted(results.items()):
        for doc_id in doc_ids:
            # Alapértelmezett relevance: 0 (manuális annotálásra vár!)
            f.write(f"{query}\t{doc_id}\t0\n")

print(f"✅ Qrels template létrehozva: {OUTPUT_QRELS}")
print("\n⚠️ FONTOS: A relevance értékeket (0/1/2) manuálisan kell beállítani!")

# === Statisztikák ===
print("\n" + "="*60)
print("📊 STATISZTIKÁK")
print("="*60)

num_queries = len(results)
num_results = [len(docs) for docs in results.values()]
total_pairs = sum(num_results)
avg_results = total_pairs / num_queries if num_queries > 0 else 0

unique_docs = set()
for docs in results.values():
    unique_docs.update(docs)

print(f"  Query-k száma:              {num_queries}")
print(f"  Összes query-doc pár:       {total_pairs}")
print(f"  Átlag találat/query:        {avg_results:.1f}")
print(f"  Egyedi dokumentumok:        {len(unique_docs)}")
print(f"  Min találat/query:          {min(num_results) if num_results else 0}")
print(f"  Max találat/query:          {max(num_results) if num_results else 0}")

# Queries with no results
no_results = [q for q, docs in results.items() if not docs]
if no_results:
    print(f"\n⚠️ {len(no_results)} query-nek nincs találata:")
    for query in no_results[:5]:
        print(f"     - {query}")
    if len(no_results) > 5:
        print(f"     ... és még {len(no_results) - 5}")

print("\n🎉 Qrels generation pipeline sikeresen lefutott!")
print("\nKövetkező lépések:")
print("1. Nyisd meg a baseline_qrels.tsv fájlt")
print("2. Állítsd be a relevance értékeket (0/1/2):")
print("   - 0 = Nem releváns")
print("   - 1 = Releváns")
print("   - 2 = Nagyon releváns")
print("3. Futtasd a GRPO training notebookot a cloud-ban")
