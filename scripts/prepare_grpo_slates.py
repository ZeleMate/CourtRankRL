#!/usr/bin/env python3
"""
GRPO Training Slate Preparation (Optimalizált verzió)
Agents.md specifikáció alapján.

Fő feladatok:
1. Query filtering: csak olyan query-ket tartunk meg ahol a retrieval talál releváns dokumentumot
2. Chunk-level slate készítés teljes szövegekkel
3. Metadata export (court, domain, year)
4. Baseline fusion ranking tárolása (slate order)

Optimalizációk:
- Unified memory cache (text + metadata): ~1.5-1.8 GB RAM, egyszeri beolvasás
- tqdm progress bar: becsült hátralévő idő, throughput tracking
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

# Config
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from configs import config
from scripts.hybrid_retrieval import HybridRetriever

# Relevancia-2 augmentation konfiguráció
MIN_HIGH_REL_CHUNKS = 2  # Minimum high-relevance chunks per slate
HIGH_REL_SCORE_BONUS = 0.15  # Score bonus for relevance==2 chunks


def build_chunk_cache(chunks_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Egyszeri betöltés: chunks.jsonl → unified memory cache.
    
    Cache struktúra: {chunk_id: {"text": str, "court": str, "domain": str, "year": str}}
    
    Returns:
        Chunk cache dictionary
    """
    import pandas as pd
    
    print("\n📦 Unified chunk cache építése (text + metadata)...")
    print(f"   Forrás: {chunks_path}")
    
    chunk_cache = {}
    
    # Egyszeri teljes beolvasás pandas-szal (gyorsabb mint chunked reading erre a célra)
    df = pd.read_json(chunks_path, lines=True)
    
    print(f"   Feldolgozás: {len(df):,} chunk...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cache építés", unit="chunk"):
        chunk_id = row.get('chunk_id')
        if chunk_id:
            chunk_cache[chunk_id] = {
                "text": row.get('text', ''),
                "court": row.get('court', ''),
                "domain": row.get('domain', ''),
                "year": row.get('year', '')
            }
    
    cache_size_mb = sum(len(str(v)) for v in chunk_cache.values()) / (1024 * 1024)
    print(f"   ✅ Cache kész: {len(chunk_cache):,} chunk (~{cache_size_mb:.1f} MB)")
    
    return chunk_cache


def load_qrels(qrels_path: Path) -> Dict[str, Dict[str, int]]:
    """
    Betölti a qrels fájlt.
    
    Returns:
        {query_id: {doc_id: relevance}}
    """
    qrels = defaultdict(dict)
    
    with open(qrels_path, "r", encoding="utf-8") as f:
        next(f)  # Skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            
            query_id, doc_id, relevance_str = parts
            try:
                relevance = int(relevance_str)
                qrels[query_id][doc_id] = relevance
            except ValueError:
                continue
    
    return dict(qrels)


def build_high_rel_mapping(qrels: Dict[str, Dict[str, int]]) -> Dict[str, List[str]]:
    """
    Relevancia==2 dokumentumok query-nként.
    
    Returns:
        {query_id: [doc_id where relevance==2]}
    """
    high_rel_docs = defaultdict(list)
    for query_id, doc_relevances in qrels.items():
        for doc_id, relevance in doc_relevances.items():
            if relevance == 2:
                high_rel_docs[query_id].append(doc_id)
    return dict(high_rel_docs)


def build_doc_chunks_mapping(chunks_path: Path) -> Dict[str, List[str]]:
    """
    Doc ID → chunk ID lista mapping chunks.jsonl alapján.
    
    Returns:
        {doc_id: [chunk_id_0, chunk_id_1, ...]}
    """
    import pandas as pd
    
    print("\n🔗 Doc→Chunk mapping építése...")
    doc_chunks = defaultdict(list)
    
    df = pd.read_json(chunks_path, lines=True)
    for _, row in df.iterrows():
        doc_id = row.get('doc_id')
        chunk_id = row.get('chunk_id')
        if doc_id and chunk_id:
            doc_chunks[doc_id].append(chunk_id)
    
    print(f"   ✅ {len(doc_chunks)} doc mapped")
    return dict(doc_chunks)


def prepare_chunk_slates(
    qrels: Dict[str, Dict[str, int]],
    retriever: HybridRetriever,
    high_rel_docs: Dict[str, List[str]],
    doc_chunks_map: Dict[str, List[str]],
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    Chunk-level slate készítés meglévő qrels-ből.
    
    Agents.md szerint:
    - Használja retriever.get_last_chunk_scores("fused") az RRF baseline orderhez
    - Betölti teljes chunk texteket retriever._load_chunk_texts()-tel
    - Metaadatokat retriever.get_doc_metadata()-val
    - Slate order = chunk-level RRF ranking (konzisztens a doc-level RRF-fel)
    
    Steps:
    1. Minden query-hez: retriever.retrieve(query, top_k=top_k)
    2. Chunk-szintű fused scores lekérése: retriever.get_last_chunk_scores("fused")
    3. Top-K chunk kiválasztása (ez lesz a baseline order)
    4. Teljes szövegek betöltése
    5. Doc-level relevance mapping chunk-ekre
    6. Slate összeállítása metadata-val
    
    Returns:
        List of slate objects for GRPO training
    """
    slates = []
    
    print(f"\n📦 Slate preparation ({len(qrels)} query)...")
    print(f"   Top-K: {top_k}\n")
    
    for query_id, doc_relevances in tqdm(qrels.items(), total=len(qrels), desc="Slate készítés", unit="query"):
        
        # Retrieve candidates (this also stores chunk-level scores)
        _ = retriever.retrieve(query_id)
        
        # Get chunk-level fused scores (RRF baseline order)
        fused_chunks = retriever.get_last_chunk_scores("fused", top_k=top_k)
        
        if not fused_chunks:
            # Edge case: nincs chunk találat
            slates.append({
                "query_id": query_id,
                "slate": [],  # Üres slate
            })
            continue
        
        # ====== ÚJ: HIGH-RELEVANCE AUGMENTÁCIÓ ======
        # 1. Ellenőrizzük: van-e elég relevancia==2 chunk a slate-ben
        current_high_rel_count = 0
        existing_chunk_ids = {chunk_id for chunk_id, _ in fused_chunks}
        existing_doc_ids = {retriever._chunk_id_to_doc_id(cid) for cid in existing_chunk_ids}
        
        query_high_rel_docs = high_rel_docs.get(query_id, [])
        missing_high_rel_docs = [doc_id for doc_id in query_high_rel_docs 
                                 if doc_id not in existing_doc_ids]
        
        # 2. Pótoljuk a hiányzó relevancia==2 chunk-okat
        if len(query_high_rel_docs) > 0:  # Van releváns doc ennél a query-nél
            for chunk_id, _ in fused_chunks:
                doc_id = retriever._chunk_id_to_doc_id(chunk_id)
                if doc_relevances.get(doc_id, 0) == 2:
                    current_high_rel_count += 1
            
            # Ha kevesebb mint MIN_HIGH_REL_CHUNKS, pótoljuk
            if current_high_rel_count < MIN_HIGH_REL_CHUNKS and missing_high_rel_docs:
                needed = MIN_HIGH_REL_CHUNKS - current_high_rel_count
                for doc_id in missing_high_rel_docs[:needed]:
                    # Első chunk választása
                    chunks_for_doc = doc_chunks_map.get(doc_id, [])
                    if chunks_for_doc:
                        # Válasszuk az első chunk-ot
                        added_chunk_id = chunks_for_doc[0]
                        # Adjuk hozzá 0 score-ral (később bonuszolva lesz)
                        fused_chunks.append((added_chunk_id, 0.0))
        
        # 3. Score bonus alkalmazása relevancia==2 chunk-okra
        fused_chunks_with_bonus = []
        for chunk_id, rrf_score in fused_chunks:
            doc_id = retriever._chunk_id_to_doc_id(chunk_id)
            relevance = doc_relevances.get(doc_id, 0)
            
            # Bonus hozzáadása
            if relevance == 2:
                adjusted_score = rrf_score + HIGH_REL_SCORE_BONUS
            else:
                adjusted_score = rrf_score
            
            fused_chunks_with_bonus.append((chunk_id, adjusted_score))
        
        # 4. Újrarendezés az új score-ok alapján
        fused_chunks_with_bonus.sort(key=lambda x: x[1], reverse=True)
        
        # 5. Top-K limitálás (ha pótlás miatt túlmentünk)
        fused_chunks = fused_chunks_with_bonus[:top_k]
        # ====== ÚJ RÉSZ VÉGE ======
        
        # Get individual scores for each chunk
        bm25_scores = {chunk_id: score for chunk_id, score in retriever.get_last_chunk_scores("bm25")}
        faiss_scores = {chunk_id: score for chunk_id, score in retriever.get_last_chunk_scores("dense")}
        
        # Load full chunk texts (agents.md: full text, not preview)
        chunk_ids = [chunk_id for chunk_id, _ in fused_chunks]
        chunk_texts = retriever._load_chunk_texts(chunk_ids)
        
        # Build slate with full metadata
        slate_candidates = []
        for chunk_id, rrf_score in fused_chunks:
            # Get doc_id and metadata
            doc_id = retriever._chunk_id_to_doc_id(chunk_id)
            metadata = retriever.get_doc_metadata(doc_id)
            
            # Map doc-level relevance to chunk
            relevance = doc_relevances.get(doc_id, 0)
            
            # Full chunk text (not preview!)
            text = chunk_texts.get(chunk_id, "")
            
            slate_candidates.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "bm25_score": bm25_scores.get(chunk_id, 0.0),
                "faiss_score": faiss_scores.get(chunk_id, 0.0),
                "rrf_score": rrf_score,
                "relevance": relevance,
                "court": metadata.get("court", ""),
                "domain": metadata.get("domain", ""),
                "year": metadata.get("year", ""),
                "text": text,  # FULL TEXT
            })
        
        slates.append({
            "query_id": query_id,
            "slate": slate_candidates,  # Order = baseline RRF ranking
        })
    
    print(f"   ✅ Feldolgozva: {len(qrels)} query")
    
    return slates


def save_slates(
    slates: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Menti a slate-eket JSONL formátumban."""
    with open(output_path, "w", encoding="utf-8") as f:
        for slate in slates:
            f.write(json.dumps(slate, ensure_ascii=False) + "\n")
    
    print(f"\n💾 Slate-ek mentve: {output_path}")
    print(f"   {len(slates)} query slate")


def print_statistics(
    slates: List[Dict[str, Any]],
    original_count: int,
) -> None:
    """Statisztikák kiírása."""
    print("\n" + "=" * 60)
    print("📊 ÖSSZEFOGLALÓ STATISZTIKÁK")
    print("=" * 60)
    
    print(f"\n📋 Query-k:")
    print(f"   • Eredeti:         {original_count}")
    print(f"   • Feldolgozott:    {len(slates)} ({100 * len(slates) / original_count:.1f}%)")
    
    if slates:
        slate_sizes = [len(slate["slate"]) for slate in slates]
        non_empty_slates = [s for s in slates if s["slate"]]  # Non-empty slates
        
        print(f"\n📦 Slate-ek:")
        print(f"   • Nem üres slate-ek:     {len(non_empty_slates)}")
        print(f"   • Üres slate-ek:         {len(slates) - len(non_empty_slates)}")
        
        if non_empty_slates:
            non_empty_sizes = [len(slate["slate"]) for slate in non_empty_slates]
            print(f"   • Átlag candidates/query: {sum(non_empty_sizes) / len(non_empty_sizes):.1f}")
            print(f"   • Min candidates:         {min(non_empty_sizes)}")
            print(f"   • Max candidates:         {max(non_empty_sizes)}")
            
            # Relevance distribution
            relevance_counts = defaultdict(int)
            for slate in non_empty_slates:
                for candidate in slate["slate"]:
                    relevance_counts[candidate["relevance"]] += 1
            
            total_candidates = sum(relevance_counts.values())
            print(f"\n🎯 Relevancia eloszlás a slate-ekben:")
            for rel in sorted(relevance_counts.keys()):
                count = relevance_counts[rel]
                pct = 100 * count / total_candidates if total_candidates > 0 else 0
                print(f"   • Relevance {rel}: {count:4d} ({pct:5.1f}%)")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="GRPO training slate preparation - chunk-level candidates from existing qrels"
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=config.QRELS_FILE,
        help="Qrels fájl elérési útja",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.DATA_DIR / "models" / "grpo_policy" / "training_slates.jsonl",
        help="Output slate fájl",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Hány candidate chunk kerüljön a slate-be",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 GRPO SLATE PREPARATION")
    print("=" * 60)
    
    # 1. Load qrels
    print(f"\n📖 Qrels betöltése: {args.qrels}")
    qrels = load_qrels(args.qrels)
    print(f"   ✅ {len(qrels)} query betöltve")
    
    # 1.1. Build high-relevance mapping
    print(f"\n🎯 High-relevance doc mapping építése...")
    high_rel_docs = build_high_rel_mapping(qrels)
    high_rel_count = sum(len(docs) for docs in high_rel_docs.values())
    print(f"   ✅ {high_rel_count} relevancia==2 dokumentum {len(high_rel_docs)} query-ben")
    
    # 2. Initialize retriever
    print(f"\n🔧 Retrieval modell inicializálása...")
    retriever = HybridRetriever()
    retriever.initialize()
    print(f"   ✅ Modell kész")
    
    # 2.1. Build unified cache (külön függvény a prepare_grpo_slates.py-ban)
    chunks_path = retriever.base_path / "data" / "processed" / "chunks.jsonl"
    chunk_cache = build_chunk_cache(chunks_path)
    
    # 2.2. Build doc→chunk mapping
    doc_chunks_map = build_doc_chunks_mapping(chunks_path)
    
    # 2.3. Set cache in retriever
    retriever.set_chunk_cache(chunk_cache)
    
    # 3. Prepare slates (no filtering - use all annotated queries)
    slates = prepare_chunk_slates(
        qrels,
        retriever,
        high_rel_docs,
        doc_chunks_map,
        top_k=args.top_k,
    )
    
    # 4. Save outputs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_slates(slates, args.output)
    
    # 5. Statistics
    print_statistics(slates, len(qrels))
    
    print("\n✅ KÉSZ!")
    print(f"\n💡 Következő lépés:")
    print(f"   Töltsd fel a {args.output} fájlt a cloud GPU-ra")
    print(f"   és futtasd a grpo_train_runpod.ipynb notebookot")


if __name__ == "__main__":
    main()

