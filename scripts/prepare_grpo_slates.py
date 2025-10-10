#!/usr/bin/env python3
"""
GRPO Training Slate Preparation
Agents.md specifikáció alapján.

Fő feladatok:
1. Query filtering: csak olyan query-ket tartunk meg ahol a retrieval talál releváns dokumentumot
2. Chunk-level slate készítés teljes szövegekkel
3. Metadata export (court, domain, year)
4. Baseline fusion ranking tárolása (slate order)

Edge case kezelés:
- Ha nincs releváns dok a top-K-ban → kiszűrjük a query-t
- nDCG=0 query-k (csak irreleváns dok-ok) → kiszűrjük
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Config
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from configs import config
from scripts.hybrid_retrieval import HybridRetriever


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


def filter_queries_by_retrievability(
    qrels: Dict[str, Dict[str, int]],
    retriever: HybridRetriever,
    top_k: int = 20,
    min_relevant_in_topk: int = 1,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, str]]:
    """
    Kiszűri azokat a query-ket ahol a retrieval nem talál releváns dokumentumot.
    
    Args:
        qrels: Query relevance judgments
        retriever: HybridRetriever instance
        top_k: Hány dokumentumot kérjünk le
        min_relevant_in_topk: Minimum hány releváns dok kell a top-K-ban
    
    Returns:
        (filtered_qrels, rejection_reasons)
    """
    filtered_qrels = {}
    rejection_reasons = {}
    
    print(f"\n🔍 Query filtering ({len(qrels)} query)...")
    print(f"   Paraméterek: top_k={top_k}, min_relevant={min_relevant_in_topk}\n")
    
    for i, (query_id, doc_relevances) in enumerate(qrels.items(), start=1):
        if i % 10 == 0:
            print(f"   Feldolgozva: {i}/{len(qrels)}...", end="\r")
        
        # Check 1: Van-e releváns dokumentum a qrels-ben?
        relevant_docs = {doc for doc, rel in doc_relevances.items() if rel > 0}
        if not relevant_docs:
            rejection_reasons[query_id] = "Nincs releváns dokumentum a qrels-ben"
            continue
        
        # Check 2: Retrieval találja-e a releváns dokumentumokat?
        try:
            retrieved_docs = retriever.retrieve(query_id, top_k=top_k)
        except Exception as e:
            rejection_reasons[query_id] = f"Retrieval hiba: {e}"
            continue
        
        # Check 3: Van-e legalább N releváns a top-K-ban?
        found_relevant = [doc for doc in retrieved_docs if doc_relevances.get(doc, 0) > 0]
        
        if len(found_relevant) < min_relevant_in_topk:
            rejection_reasons[query_id] = (
                f"Csak {len(found_relevant)} releváns dok a top-{top_k}-ban "
                f"(minimum: {min_relevant_in_topk})"
            )
            continue
        
        # Query passed all checks
        filtered_qrels[query_id] = doc_relevances
    
    print(f"\r   ✅ Feldolgozva: {len(qrels)}/{len(qrels)}")
    
    return filtered_qrels, rejection_reasons


def prepare_chunk_slates(
    filtered_qrels: Dict[str, Dict[str, int]],
    retriever: HybridRetriever,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Készíti a chunk-level slate-eket GRPO training-hez.
    
    Agents.md szerint:
    - Chunk-level candidates (not doc-level aggregation)
    - Full chunk text (~500-800 chars, NOT preview)
    - Metadata: court, domain, year
    - Retrieval scores: bm25, faiss
    - Slate order = baseline fusion ranking
    
    Returns:
        List of slate objects for GRPO training
    """
    slates = []
    
    print(f"\n📦 Slate preparation ({len(filtered_qrels)} query)...")
    print(f"   Top-K: {top_k}\n")
    
    for i, (query_id, doc_relevances) in enumerate(filtered_qrels.items(), start=1):
        if i % 10 == 0:
            print(f"   Feldolgozva: {i}/{len(filtered_qrels)}...", end="\r")
        
        # Retrieve candidates (this also stores chunk-level scores)
        _ = retriever.retrieve(query_id, top_k=top_k)
        
        # Get chunk-level scores from retriever
        bm25_chunks = retriever.get_last_chunk_scores("bm25")
        faiss_chunks = retriever.get_last_chunk_scores("dense")
        
        # Combine and deduplicate chunks
        chunk_scores: Dict[str, Dict[str, float]] = {}
        
        for chunk_id, score in bm25_chunks[:top_k]:
            chunk_scores[chunk_id] = {"bm25": score, "faiss": 0.0}
        
        for chunk_id, score in faiss_chunks[:top_k]:
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id]["faiss"] = score
            else:
                chunk_scores[chunk_id] = {"bm25": 0.0, "faiss": score}
        
        # Get top-K chunks by combined score (this is the baseline fusion order)
        # Use simple average for slate ordering
        top_chunks = sorted(
            chunk_scores.items(),
            key=lambda item: (item[1]["bm25"] + item[1]["faiss"]) / 2,
            reverse=True
        )[:top_k]
        
        if not top_chunks:
            continue
        
        # Load full chunk texts (agents.md: full text, not preview)
        chunk_ids = [chunk_id for chunk_id, _ in top_chunks]
        chunk_texts = retriever._load_chunk_texts(chunk_ids)
        
        # Build slate with full metadata
        slate_candidates = []
        for chunk_id, scores in top_chunks:
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
                "bm25_score": scores["bm25"],
                "faiss_score": scores["faiss"],
                "relevance": relevance,
                "court": metadata.get("court", ""),
                "domain": metadata.get("domain", ""),
                "year": metadata.get("year", ""),
                "text": text,  # FULL TEXT
            })
        
        slates.append({
            "query_id": query_id,
            "slate": slate_candidates,  # Order = baseline fusion ranking
        })
    
    print(f"\r   ✅ Feldolgozva: {len(filtered_qrels)}/{len(filtered_qrels)}")
    
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


def save_rejection_report(
    rejection_reasons: Dict[str, str],
    original_count: int,
    output_path: Path,
) -> None:
    """Menti a kiszűrt query-k jelentését."""
    report = {
        "total_queries": original_count,
        "rejected_count": len(rejection_reasons),
        "accepted_count": original_count - len(rejection_reasons),
        "rejection_rate": len(rejection_reasons) / original_count if original_count > 0 else 0,
        "rejected_queries": rejection_reasons,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 Rejection report mentve: {output_path}")


def print_statistics(
    slates: List[Dict[str, Any]],
    rejection_reasons: Dict[str, str],
    original_count: int,
) -> None:
    """Statisztikák kiírása."""
    print("\n" + "=" * 60)
    print("📊 ÖSSZEFOGLALÓ STATISZTIKÁK")
    print("=" * 60)
    
    print(f"\n📋 Query-k:")
    print(f"   • Eredeti:         {original_count}")
    print(f"   • Elfogadott:      {len(slates)} ({100 * len(slates) / original_count:.1f}%)")
    print(f"   • Elutasított:     {len(rejection_reasons)} ({100 * len(rejection_reasons) / original_count:.1f}%)")
    
    if slates:
        slate_sizes = [len(slate["slate"]) for slate in slates]
        print(f"\n📦 Slate-ek:")
        print(f"   • Átlag candidates/query: {sum(slate_sizes) / len(slate_sizes):.1f}")
        print(f"   • Min candidates:         {min(slate_sizes)}")
        print(f"   • Max candidates:         {max(slate_sizes)}")
        
        # Relevance distribution
        relevance_counts = defaultdict(int)
        for slate in slates:
            for candidate in slate["slate"]:
                relevance_counts[candidate["relevance"]] += 1
        
        total_candidates = sum(relevance_counts.values())
        print(f"\n🎯 Relevancia eloszlás a slate-ekben:")
        for rel in sorted(relevance_counts.keys()):
            count = relevance_counts[rel]
            pct = 100 * count / total_candidates if total_candidates > 0 else 0
            print(f"   • Relevance {rel}: {count:4d} ({pct:5.1f}%)")
    
    if rejection_reasons:
        print(f"\n❌ Elutasítás okai (top 5):")
        reason_counts = defaultdict(int)
        for reason in rejection_reasons.values():
            reason_counts[reason] += 1
        
        for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   • {count:3d}x: {reason}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="GRPO training slate preparation - chunk-level candidates with filtering"
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
        default=10,
        help="Hány candidate chunk kerüljön a slate-be",
    )
    parser.add_argument(
        "--min-relevant",
        type=int,
        default=1,
        help="Minimum hány releváns dok kell a top-K-ban",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Minden query-t megtart (nem ajánlott GRPO-hoz)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 GRPO SLATE PREPARATION")
    print("=" * 60)
    
    # 1. Load qrels
    print(f"\n📖 Qrels betöltése: {args.qrels}")
    qrels = load_qrels(args.qrels)
    print(f"   ✅ {len(qrels)} query betöltve")
    
    # 2. Initialize retriever
    print(f"\n🔧 Retrieval modell inicializálása...")
    retriever = HybridRetriever()
    print(f"   ✅ Modell kész")
    
    # 3. Filter queries (optional)
    if args.no_filter:
        print(f"\n⚠️  Query filtering kihagyva (--no-filter)")
        filtered_qrels = qrels
        rejection_reasons = {}
    else:
        filtered_qrels, rejection_reasons = filter_queries_by_retrievability(
            qrels,
            retriever,
            top_k=args.top_k * 2,  # Check 2x candidates
            min_relevant_in_topk=args.min_relevant,
        )
    
    if not filtered_qrels:
        print("\n❌ Nincs egyetlen query sem ami megfelelt a kritériumoknak!")
        print("   Próbáld ki a --no-filter flag-et vagy csökkentsd a --min-relevant értékét.")
        sys.exit(1)
    
    # 4. Prepare slates
    slates = prepare_chunk_slates(
        filtered_qrels,
        retriever,
        top_k=args.top_k,
    )
    
    # 5. Save outputs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_slates(slates, args.output)
    
    if rejection_reasons:
        rejection_report_path = args.output.parent / "query_filtering_report.json"
        save_rejection_report(rejection_reasons, len(qrels), rejection_report_path)
    
    # 6. Statistics
    print_statistics(slates, rejection_reasons, len(qrels))
    
    print("\n✅ KÉSZ!")
    print(f"\n💡 Következő lépés:")
    print(f"   Töltsd fel a {args.output} fájlt a cloud GPU-ra")
    print(f"   és futtasd a grpo_train_runpod.ipynb notebookot")


if __name__ == "__main__":
    main()

