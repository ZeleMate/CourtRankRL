#!/usr/bin/env python3
"""
CourtRankRL CLI Interface
Agents.md specifikáció alapján implementálva.

Parancsok:
- build: Docling feldolgozás → chunking → BM25 index
- query: hibrid (BM25+FAISS) keresés, opcionális GRPO reranking
- train: GRPO reranker policy tanítása (baseline → feature export → GRPO)
"""

import argparse
import sys
from pathlib import Path
from typing import List

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from configs import config
from src.data_loader.preprocess_documents import main as build_docs
from src.data_loader.build_bm25_index import main as build_bm25
from src.search.hybrid_search import HybridRetriever
from src.search.grpo_reranker import GRPOReranker, load_qrels, export_slates_for_grpo_training

def build_command():
    """Build pipeline: Docling → chunking → BM25."""
    print("=== COURTRANKRL BUILD PIPELINE ===")

    try:
        # 1. lépés: Docling feldolgozás
        print("1. lépés: Docling feldolgozás és normalizálás...")
        print("   (DOCX parsing, minimális normalizálás)")
        build_docs()

        # 2. lépés: Chunkolás
        print("2. lépés: Intelligens chunkolás...")
        print("   (Docling dönti el a chunk méretet és átfedést)")

        # 3. lépés: BM25 indexelés
        print("3. lépés: BM25S index (bm25s.tokenize + cache)...")
        build_bm25()

        print("\nBUILD PIPELINE KÉSZ!")
        print("Generált artifaktumok:")
        print(f"   Chunks: {config.CHUNKS_JSONL}")
        print(f"   BM25 Index: {config.BM25_INDEX_PATH}")

        print("\nKész a lekérdezésre! Használat: uv run courtrankrl query \"kérdés\"")
        print("Megjegyzés: FAISS index generálása szükséges a gemma_embedding_runpod.ipynb futtatásával")

    except Exception as e:
        print(f"\nBUILD SIKERTELEN: {e}")
        print("Ellenőrizd a hibaüzenetet és próbáld újra.")
        sys.exit(1)

def query_command(query: str, top_k: int = 10, rerank: bool = True):
    """Query pipeline: embed query → BM25 + dense → fusion → RL reranking → doc IDs."""
    print("=== COURTRANKRL QUERY PIPELINE ===")
    print(f"Lekérdezés: {query}")
    print(f"Top-K: {top_k}")
    print(f"Reranking: {'bekapcsolva' if rerank else 'kikapcsolva'}")

    try:
        # Retriever inicializálása
        retriever = HybridRetriever()

        if rerank:
            # 1. lépés: Jelöltek lekérése rerankinghez (agents.md 4. lépés)
            print("1. lépés: Jelöltek lekérése rerankinghez...")
            retriever.retrieve_candidates(query, top_k=config.TOP_K_BASELINE)
            bm25_results = retriever.get_last_doc_scores("bm25")
            dense_results = retriever.get_last_doc_scores("dense")

            print(f"   BM25 jelöltek: {len(bm25_results)}")
            print(f"   Dense jelöltek: {len(dense_results)}")

            # 2. lépés: GRPO reranking alkalmazása (agents.md 5. lépés)
            print("2. lépés: GRPO reranking alkalmazása...")
            try:
                reranker = GRPOReranker()

                # Eredmények konvertálása jelölt formátumra
                candidates = []
                all_docs = set()
                for doc_id, score in bm25_results:
                    all_docs.add(doc_id)
                    candidates.append({
                        "doc_id": doc_id,
                        "bm25_score": score,
                        "faiss_score": dense_results.get(doc_id, 0.0) if doc_id in [d[0] for d in dense_results] else 0.0
                    })

                # Reranking
                reranked = reranker.rerank(query, candidates)[:top_k]
                print(f"   Újrendezett lista: {len(reranked)} dokumentum")

                print("\nRERANKELT EREDMÉNYEK:")
                for idx, candidate in enumerate(reranked, start=1):
                    print(f"{idx}. {candidate['doc_id']}")

            except Exception as e:
                print(f"Figyelmeztetés: Reranker nem elérhető ({e}), baseline használata...")
                rerank = False

        if not rerank:
            # 1. lépés: Hibrid baseline retrieval (agents.md 4. lépés)
            print("1. lépés: Hibrid baseline retrieval...")
            doc_ids = retriever.retrieve(query, top_k=top_k, fusion_method="rrf")

            print(f"   Találatok száma: {len(doc_ids)}")

            print("\nBASELINE EREDMÉNYEK:")
            for idx, doc_id in enumerate(doc_ids, start=1):
                print(f"{idx}. {doc_id}")

        print(f"\nQuery sikeresen befejezve!")

    except Exception as e:
        print(f"\nQUERY SIKERTELEN: {e}")
        print("Ellenőrizd, hogy az indexek elkészültek: uv run courtrankrl build")
        sys.exit(1)

def train_command():
    """Train GRPO reranker: load qrels → baseline candidates → features → GRPO training."""
    print("=== COURTRANKRL GRPO TRAINING PIPELINE ===")
    print("Agents.md spec: 5) RL Reranking (GRPO-style)")
    print("Cél: a baseline javítása megerősítéses tanulással")

    try:
        # 1. lépés: Qrels betöltése
        print("1. lépés: Qrels betöltése...")
        qrels = load_qrels(config.QRELS_FILE)

        if not qrels:
            print(f"Nincs qrels adat: {config.QRELS_FILE}")
            return

        print(f"   Betöltött query-k: {len(qrels)}")

        # 2. lépés: Retriever inicializálása
        print("2. lépés: Retriever inicializálása...")
        retriever = HybridRetriever()

        # 3. lépés: Slate export cloud traininghez
        print("3. lépés: Slate export cloud traininghez...")
        slates = export_slates_for_grpo_training(retriever, qrels, config.SLATE_EXPORT_PATH)

        print("4. lépés: Cloud training notebook futtatása...")
        print(f"   Notebook: {config.GRPO_TRAIN_NOTEBOOK}")
        print("   RunPod-on futtassa: grpo_train_runpod.ipynb")
        print("   Input: training_slates.jsonl (kész)")

        print("\nGRPO TRAINING ELŐKÉSZÍTVE!")
        print("Következő lépések:")
        print("   1. Másolja a slates fájlt RunPod-ra")
        print(f"   2. Futtassa: {config.GRPO_TRAIN_NOTEBOOK}")
        print("   3. Töltse le az adapter fájlokat")
        print("   4. Helyezze el: data/models/grpo_policy/")

        print("\nArtifactumok:")
        print(f"   Slates: {config.SLATE_EXPORT_PATH}")
        print(f"   Adapter: {config.GRPO_ADAPTER_PATH} (RunPod-ról)")
        print(f"   Tokenizer: {config.GRPO_TOKENIZER_PATH} (RunPod-ról)")
        print(f"   Metrics: {config.GRPO_METRICS_PATH} (RunPod-ról)")

    except Exception as e:
        print(f"\nGRPO TRAINING ELŐKÉSZÍTÉS SIKERTELEN: {e}")
        print("Ellenőrizd, hogy a qrels fájl létezik és az indexek elkészültek.")
        sys.exit(1)

def main():
    """Main CLI entry point for CourtRankRL."""
    parser = argparse.ArgumentParser(
        description="CourtRankRL - Magyar bírósági döntések retrieval rendszer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Példák használatra:

  # Build pipeline futtatása
  uv run courtrankrl build

  # Keresés baseline módban
  uv run courtrankrl query "családi jogi ügy" --no-rerank

  # Keresés GRPO reranking-gal
  uv run courtrankrl query "szerződéses jog" --top-k 5

  # GRPO policy tanítása (cloud előkészítés)
  uv run courtrankrl train

Használat előtt:
  1. uv run courtrankrl build
  2. Generate FAISS index using gemma_embedding_runpod.ipynb
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Elérhető parancsok')

    # Build command
    build_parser = subparsers.add_parser(
        'build',
        help='Build pipeline: Docling → chunking → BM25 indexing'
    )

    # Query command
    query_parser = subparsers.add_parser(
        'query',
        help='Keresés a rendszerben dokumentum azonosítókkal'
    )
    query_parser.add_argument(
        'query',
        help='Keresési lekérdezés'
    )
    query_parser.add_argument(
        '--top-k', type=int, default=10,
        help='Visszaadandó dokumentumok száma (alap: 10)'
    )
    query_parser.add_argument(
        '--fusion-method', choices=['rrf', 'zscore'], default='rrf',
        help='Fusion method: rrf vagy zscore (alap: rrf)'
    )
    query_parser.add_argument(
        '--no-rerank', action='store_true',
        help='GRPO reranking kikapcsolása (csak baseline keresés)'
    )

    # Train command
    train_parser = subparsers.add_parser(
        'train',
        help='GRPO reranker policy tanítása'
    )

    # Parse arguments
    args = parser.parse_args()

    # Route to appropriate command
    if args.command == 'build':
        build_command()
    elif args.command == 'query':
        query_command(args.query, args.top_k, not args.no_rerank)
    elif args.command == 'train':
        train_command()
    else:
        parser.print_help()
        print("\nKezdéshez: uv run courtrankrl build")

if __name__ == '__main__':
    main()
