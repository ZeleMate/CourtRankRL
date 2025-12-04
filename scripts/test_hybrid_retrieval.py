#!/usr/bin/env python3
"""
Interaktív teszt script a hibrid kereső működéséhez.

Használat:
    python scripts/test_hybrid_retrieval.py
    
Vagy interaktív módban:
    python scripts/test_hybrid_retrieval.py --interactive
"""

import sys
import json
from pathlib import Path

# Project root hozzáadása a path-hoz
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.hybrid_retrieval import HybridRetriever, aggregate_chunks_to_docs


def load_chunks_for_display(chunks_path: Path, doc_ids: list, limit: int = 5) -> dict:
    """Betölti a chunk szövegeket megjelenítéshez."""
    import pandas as pd
    
    chunk_texts = {}
    doc_ids_set = set(doc_ids[:limit])
    
    for chunk_df in pd.read_json(chunks_path, lines=True, chunksize=5000):
        for _, row in chunk_df.iterrows():
            chunk_id = row.get('chunk_id', '')
            # Ellenőrizzük, hogy a chunk a keresett dokumentumhoz tartozik-e
            doc_id = chunk_id.rsplit('_', 1)[0] if '_' in chunk_id and chunk_id.rsplit('_', 1)[1].isdigit() else chunk_id
            
            if doc_id in doc_ids_set and doc_id not in chunk_texts:
                chunk_texts[doc_id] = {
                    'text': row.get('text', '')[:500] + '...',  # Első 500 karakter
                    'court': row.get('court', 'N/A'),
                    'domain': row.get('domain', 'N/A'),
                    'year': row.get('year', 'N/A'),
                }
        
        if len(chunk_texts) >= limit:
            break
    
    return chunk_texts


def test_single_query(retriever: HybridRetriever, query: str, show_details: bool = True):
    """Egyetlen query tesztelése részletes kimenettel."""
    print(f"\n{'='*70}")
    print(f"🔍 QUERY: {query}")
    print('='*70)
    
    # Retrieval
    results = retriever.retrieve(query)
    
    print(f"\n📊 Eredmények száma: {len(results)}")
    print(f"\n🏆 Top 10 dokumentum:")
    
    for i, doc_id in enumerate(results[:10], 1):
        print(f"  {i:2d}. {doc_id}")
    
    if show_details:
        # Chunk-szintű részletek
        print(f"\n📈 BM25 Top 5 chunk:")
        bm25_chunks = retriever.get_last_chunk_scores("bm25", top_k=5)
        for chunk_id, score in bm25_chunks:
            print(f"     {chunk_id}: {score:.4f}")
        
        print(f"\n📈 FAISS Top 5 chunk:")
        faiss_chunks = retriever.get_last_chunk_scores("dense", top_k=5)
        for chunk_id, score in faiss_chunks:
            print(f"     {chunk_id}: {score:.4f}")
        
        print(f"\n📈 Fused (RRF) Top 5 chunk:")
        fused_chunks = retriever.get_last_chunk_scores("fused", top_k=5)
        for chunk_id, score in fused_chunks:
            print(f"     {chunk_id}: {score:.4f}")
    
    return results


def test_retrieval_quality(retriever: HybridRetriever, qrels_path: Path):
    """Minőségellenőrzés qrels alapján."""
    if not qrels_path.exists():
        print(f"⚠️  Qrels fájl nem található: {qrels_path}")
        return
    
    print(f"\n{'='*70}")
    print("📋 MINŐSÉGELLENŐRZÉS (qrels alapján)")
    print('='*70)
    
    # Qrels betöltése
    qrels = {}
    with open(qrels_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                query, doc_id, rel = parts[0], parts[1], int(parts[2])
                if query not in qrels:
                    qrels[query] = {}
                qrels[query][doc_id] = rel
    
    print(f"✅ {len(qrels)} query betöltve a qrels-ből")
    
    # Metrikák számítása
    total_queries = 0
    total_hits_at_10 = 0
    total_hits_at_20 = 0
    mrr_sum = 0.0
    
    for query, relevance in list(qrels.items())[:20]:  # Első 20 query
        results = retriever.retrieve(query)
        
        # Hits@K
        rel_docs = {d for d, r in relevance.items() if r > 0}
        hits_10 = len(set(results[:10]) & rel_docs)
        hits_20 = len(set(results[:20]) & rel_docs)
        
        # MRR
        mrr = 0.0
        for i, doc in enumerate(results[:20], 1):
            if doc in rel_docs:
                mrr = 1.0 / i
                break
        
        total_queries += 1
        total_hits_at_10 += 1 if hits_10 > 0 else 0
        total_hits_at_20 += 1 if hits_20 > 0 else 0
        mrr_sum += mrr
        
        print(f"  Query: {query[:50]:50s} | Hits@10: {hits_10} | MRR: {mrr:.3f}")
    
    if total_queries > 0:
        print(f"\n📊 Összesített metrikák ({total_queries} query):")
        print(f"  Hit Rate@10: {total_hits_at_10/total_queries:.2%}")
        print(f"  Hit Rate@20: {total_hits_at_20/total_queries:.2%}")
        print(f"  MRR@20:      {mrr_sum/total_queries:.4f}")


def compare_bm25_vs_dense(retriever: HybridRetriever, query: str):
    """BM25 és Dense eredmények összehasonlítása."""
    print(f"\n{'='*70}")
    print(f"⚖️  BM25 vs DENSE összehasonlítás: {query}")
    print('='*70)
    
    # Retrieval futtatása (betölti a cache-t)
    _ = retriever.retrieve(query)
    
    # BM25 doc-level
    bm25_chunks = retriever.get_last_chunk_scores("bm25")
    bm25_docs = aggregate_chunks_to_docs(bm25_chunks)[:10]
    
    # Dense doc-level
    dense_chunks = retriever.get_last_chunk_scores("dense")
    dense_docs = aggregate_chunks_to_docs(dense_chunks)[:10]
    
    # Fused
    fused_chunks = retriever.get_last_chunk_scores("fused")
    fused_docs = aggregate_chunks_to_docs(fused_chunks)[:10]
    
    print(f"\n{'BM25 Top 10':<30} | {'Dense Top 10':<30} | {'Fused Top 10':<30}")
    print('-' * 95)
    
    for i in range(10):
        bm25_doc = bm25_docs[i][0] if i < len(bm25_docs) else '-'
        dense_doc = dense_docs[i][0] if i < len(dense_docs) else '-'
        fused_doc = fused_docs[i][0] if i < len(fused_docs) else '-'
        print(f"{bm25_doc:<30} | {dense_doc:<30} | {fused_doc:<30}")
    
    # Overlap elemzés
    bm25_set = {d[0] for d in bm25_docs}
    dense_set = {d[0] for d in dense_docs}
    fused_set = {d[0] for d in fused_docs}
    
    print(f"\n📊 Overlap elemzés (Top 10):")
    print(f"  BM25 ∩ Dense:  {len(bm25_set & dense_set)} közös")
    print(f"  BM25 ∩ Fused:  {len(bm25_set & fused_set)} közös")
    print(f"  Dense ∩ Fused: {len(dense_set & fused_set)} közös")
    print(f"  Csak BM25:     {len(bm25_set - dense_set - fused_set)}")
    print(f"  Csak Dense:    {len(dense_set - bm25_set - fused_set)}")


def interactive_mode(retriever: HybridRetriever):
    """Interaktív keresési mód."""
    print("\n" + "="*70)
    print("🔄 INTERAKTÍV MÓD")
    print("Írj be egy keresési kifejezést, vagy 'q' a kilépéshez")
    print("Parancsok: 'compare <query>' - BM25 vs Dense összehasonlítás")
    print("="*70)
    
    while True:
        try:
            query = input("\n🔍 Query: ").strip()
            
            if query.lower() == 'q':
                print("👋 Kilépés...")
                break
            
            if not query:
                continue
            
            if query.lower().startswith('compare '):
                compare_bm25_vs_dense(retriever, query[8:])
            else:
                test_single_query(retriever, query)
                
        except KeyboardInterrupt:
            print("\n👋 Kilépés...")
            break


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Hibrid kereső tesztelése")
    parser.add_argument('--interactive', '-i', action='store_true', 
                        help="Interaktív mód")
    parser.add_argument('--query', '-q', type=str, 
                        help="Egyetlen query tesztelése")
    parser.add_argument('--compare', '-c', type=str,
                        help="BM25 vs Dense összehasonlítás")
    parser.add_argument('--quality', action='store_true',
                        help="Minőségellenőrzés qrels alapján")
    args = parser.parse_args()
    
    # Retriever inicializálása
    print("🚀 HybridRetriever inicializálása...")
    retriever = HybridRetriever()
    retriever.initialize()
    print("✅ Retriever kész!")
    
    # Tesztek futtatása
    if args.query:
        test_single_query(retriever, args.query)
    
    elif args.compare:
        compare_bm25_vs_dense(retriever, args.compare)
    
    elif args.quality:
        qrels_path = Path(retriever.base_path) / "data" / "qrels" / "baseline_qrels.tsv"
        test_retrieval_quality(retriever, qrels_path)
    
    elif args.interactive:
        interactive_mode(retriever)
    
    else:
        # Default: néhány példa query tesztelése
        test_queries = [
            "munkáltatói kártérítési felelősség",
            "gyermektartásdíj megállapítás",
            "lopás vétség eljárás megszüntetése",
            "biztosítási szerződés érvénytelenség",
        ]
        
        print("\n📝 Példa query-k tesztelése...")
        for query in test_queries:
            test_single_query(retriever, query, show_details=False)
        
        print("\n💡 További opciók:")
        print("  --interactive (-i): Interaktív mód")
        print("  --query 'keresés': Egyetlen query részletes tesztelése")
        print("  --compare 'keresés': BM25 vs Dense összehasonlítás")
        print("  --quality: Minőségellenőrzés qrels alapján")


if __name__ == "__main__":
    main()
