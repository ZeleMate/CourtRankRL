#!/usr/bin/env python3
"""
Qrels validáló script
Ellenőrzi a qrels fájl formátumát és konzisztenciáját.
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

# Config
project_root = Path(__file__).resolve().parent.parent
qrels_path = project_root / "data" / "qrels" / "baseline_qrels.tsv"
chunks_path = project_root / "data" / "processed" / "chunks.jsonl"

VALID_RELEVANCES = {0, 1, 2}
MIN_QUERIES = 50
MIN_DOCS_PER_QUERY = 3
MIN_RECALL_THRESHOLD = 0.3  # Minimum 30% recall a retrieval-ből

class QrelsValidator:
    def __init__(self, qrels_path: Path, chunks_path: Path):
        self.qrels_path = qrels_path
        self.chunks_path = chunks_path
        self.errors = []
        self.warnings = []
        self.qrels_data = []
        self.available_doc_ids = set()

    def validate_all(self, check_retrieval: bool = False) -> bool:
        """
        Futtat minden validációt és visszaadja az eredményt.

        Args:
            check_retrieval: Ha True, ellenőrzi a retrieval recall-t (lassabb)
        """
        print("=" * 60)
        print("🔍 Qrels Validáció")
        print("=" * 60)

        checks = [
            ("Fájl létezés", self.check_file_exists),
            ("Fájl formátum (encoding, newlines)", self.check_file_format),
            ("TSV struktúra és header", self.check_tsv_structure),
            ("Relevancia értékek", self.check_relevance_values),
            ("Doc ID-k elérhetősége", self.check_doc_ids_exist),
            ("Minimum követelmények", self.check_minimum_requirements),
            ("Query-dokumentum eloszlás", self.check_distribution),
        ]

        # Opcionális retrieval check (lassú, de hasznos)
        if check_retrieval:
            checks.append(("Retrieval recall (opcionális)", self.check_retrievability))

        all_passed = True

        for check_name, check_func in checks:
            print(f"\n📋 {check_name}...", end=" ")
            try:
                result = check_func()
                if result:
                    print("✅")
                else:
                    print("❌")
                    all_passed = False
            except Exception as e:
                print(f"❌ (Hiba: {e})")
                self.errors.append(f"{check_name}: {e}")
                all_passed = False

        # Eredmények összefoglalása
        self.print_summary()

        return all_passed

    def check_file_exists(self) -> bool:
        """Ellenőrzi, hogy a qrels fájl létezik-e."""
        if not self.qrels_path.exists():
            self.errors.append(f"Qrels fájl nem található: {self.qrels_path}")
            return False
        return True

    def check_file_format(self) -> bool:
        """Ellenőrzi az encoding-ot és newline karaktereket."""
        try:
            with open(self.qrels_path, 'rb') as f:
                content = f.read()

            # BOM check
            if content.startswith(b'\xef\xbb\xbf'):
                self.warnings.append("BOM található a fájl elején (UTF-8 BOM), távolítsd el")

            # Newline check
            if b'\r\n' in content:
                self.warnings.append("Windows newline (CRLF) karakterek, használj Unix newline-t (LF)")

            # UTF-8 encoding check
            try:
                content.decode('utf-8')
            except UnicodeDecodeError as e:
                self.errors.append(f"Nem UTF-8 kódolású fájl: {e}")
                return False

            return True
        except Exception as e:
            self.errors.append(f"Fájl olvasási hiba: {e}")
            return False

    def check_tsv_structure(self) -> bool:
        """Ellenőrzi a TSV struktúrát és header-t."""
        try:
            with open(self.qrels_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if not lines:
                self.errors.append("Üres fájl")
                return False

            # Header check
            header = lines[0].strip()
            expected_header = "query_id\tdoc_id\trelevance"
            if header != expected_header:
                self.errors.append(
                    f"Helytelen header. Várt: '{expected_header}', Kapott: '{header}'"
                )
                return False

            # Parse qrels
            for line_num, line in enumerate(lines[1:], start=2):
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) != 3:
                    self.errors.append(
                        f"Sor {line_num}: 3 oszlop szükséges (TAB-bal elválasztva), {len(parts)} található"
                    )
                    return False

                query_id, doc_id, relevance_str = parts

                try:
                    relevance = int(relevance_str)
                except ValueError:
                    self.errors.append(
                        f"Sor {line_num}: relevance nem egész szám: '{relevance_str}'"
                    )
                    return False

                self.qrels_data.append({
                    'line_num': line_num,
                    'query_id': query_id,
                    'doc_id': doc_id,
                    'relevance': relevance
                })

            return True
        except Exception as e:
            self.errors.append(f"TSV parsing hiba: {e}")
            return False

    def check_relevance_values(self) -> bool:
        """Ellenőrzi, hogy a relevancia értékek 0, 1, vagy 2."""
        invalid = []
        for entry in self.qrels_data:
            if entry['relevance'] not in VALID_RELEVANCES:
                invalid.append(
                    f"Sor {entry['line_num']}: érvénytelen relevance={entry['relevance']} "
                    f"(query: {entry['query_id'][:50]}...)"
                )

        if invalid:
            self.errors.extend(invalid[:10])  # Csak az első 10-et mutatjuk
            if len(invalid) > 10:
                self.errors.append(f"... és még {len(invalid) - 10} hiba")
            return False

        return True

    def check_doc_ids_exist(self) -> bool:
        """
        Ellenőrzi, hogy a doc_id-k léteznek-e a chunks.jsonl-ben.

        Optimalizálva pandas.read_json() használatával (agents.md szerint) - 10-30x gyorsabb
        mint kézi json.loads() parsing 2.9M soros fájlnál.
        """
        import pandas as pd

        if not self.chunks_path.exists():
            self.warnings.append(f"Chunks fájl nem található, doc_id validáció kihagyva: {self.chunks_path}")
            return True

        # Load available doc_ids pandas-szal (chunked reading memória-hatékonyság miatt)
        print("\n   🔄 Chunks fájl beolvasása (pandas optimalizált)...", end=" ")
        try:
            for i, chunk_df in enumerate(pd.read_json(self.chunks_path, lines=True, encoding='utf-8', chunksize=50000)):
                if i > 0 and i % 10 == 0:
                    print(f"\r   🔄 Feldolgozva: {i * 50000:,} chunk...", end=" ")

                # Egyedi doc_id-k kinyerése a chunk DataFrame-ből
                doc_ids = chunk_df['doc_id'].dropna().unique()
                self.available_doc_ids.update(doc_ids)

            print(f"\r   ✅ {len(self.available_doc_ids):,} egyedi doc_id betöltve")
        except Exception as e:
            self.warnings.append(f"Chunks betöltési hiba: {e}")
            return True

        # Check qrels doc_ids
        qrels_doc_ids = {entry['doc_id'] for entry in self.qrels_data}
        missing = qrels_doc_ids - self.available_doc_ids

        if missing:
            self.errors.append(
                f"{len(missing)} doc_id nem található a chunks.jsonl-ben. Példák:"
            )
            for doc_id in list(missing)[:5]:
                self.errors.append(f"  - {doc_id}")
            return False

        return True

    def check_minimum_requirements(self) -> bool:
        """Ellenőrzi a minimum követelményeket."""
        # Query count
        queries = {entry['query_id'] for entry in self.qrels_data}
        if len(queries) < MIN_QUERIES:
            self.warnings.append(
                f"Kevés query: {len(queries)} (minimum: {MIN_QUERIES})"
            )

        # Docs per query
        docs_per_query = defaultdict(set)
        for entry in self.qrels_data:
            docs_per_query[entry['query_id']].add(entry['doc_id'])

        insufficient = []
        for query, docs in docs_per_query.items():
            if len(docs) < MIN_DOCS_PER_QUERY:
                insufficient.append((query, len(docs)))

        if insufficient:
            self.warnings.append(
                f"{len(insufficient)} query-nek kevesebb mint {MIN_DOCS_PER_QUERY} dokumentuma van:"
            )
            for query, count in insufficient[:5]:
                self.warnings.append(f"  - '{query[:50]}...': {count} dokumentum")

        # At least one relevant doc per query
        relevant_per_query = defaultdict(int)
        for entry in self.qrels_data:
            if entry['relevance'] > 0:
                relevant_per_query[entry['query_id']] += 1

        no_relevant = [q for q in queries if relevant_per_query[q] == 0]
        if no_relevant:
            self.errors.append(
                f"{len(no_relevant)} query-nek nincs releváns dokumentuma (relevance > 0):"
            )
            for query in no_relevant[:5]:
                self.errors.append(f"  - '{query[:50]}...'")
            return False

        return True

    def check_distribution(self) -> bool:
        """Ellenőrzi a relevancia értékek eloszlását."""
        relevance_counts = Counter(entry['relevance'] for entry in self.qrels_data)

        total = sum(relevance_counts.values())
        print(f"\n   📊 Relevancia eloszlás:")
        for rel in sorted(VALID_RELEVANCES):
            count = relevance_counts.get(rel, 0)
            pct = 100 * count / total if total > 0 else 0
            print(f"      {rel}: {count:4d} ({pct:5.1f}%)")

        # Warning if all are 0
        if relevance_counts[0] == total:
            self.warnings.append(
                "Minden dokumentum relevance=0! Módosítsd a valós értékekre."
            )

        return True

    def check_retrievability(self) -> bool:
        """
        Ellenőrzi hogy a retrieval megtalálja-e a releváns dokumentumokat.
        Ez egy opcionális, lassabb ellenőrzés ami betölti a retrieval modellt.

        Edge case amit kezel:
        - Ha a qrels-ben VAN releváns dokumentum, de a retrieval NEM találja meg
        - Ez nem bug, hanem gyenge retrieval performance jele
        """
        try:
            print("\n   🔄 Retrieval modell betöltése (ez eltarthat)...", end=" ")

            # Lazy import hogy ne lassítson minden validációt
            sys.path.insert(0, str(project_root / "src"))
            from src.search.hybrid_search import HybridRetriever

            retriever = HybridRetriever()
            print("✅")

            # Query-k csoportosítása relevancia szerint
            queries_by_relevance = defaultdict(lambda: {'relevant_docs': set(), 'all_docs': set()})
            for entry in self.qrels_data:
                query_id = entry['query_id']
                doc_id = entry['doc_id']
                queries_by_relevance[query_id]['all_docs'].add(doc_id)
                if entry['relevance'] > 0:
                    queries_by_relevance[query_id]['relevant_docs'].add(doc_id)

            # Ellenőrzés query-nként
            low_recall_queries = []
            zero_recall_queries = []
            total_queries = len(queries_by_relevance)

            print(f"   🔍 {total_queries} query ellenőrzése...", end=" ")

            for i, (query_id, data) in enumerate(queries_by_relevance.items()):
                if i > 0 and i % 10 == 0:
                    print(f"\r   🔍 Feldolgozva: {i}/{total_queries}...", end=" ")

                relevant_docs = data['relevant_docs']
                if not relevant_docs:
                    continue  # Skip queries without relevant docs

                # Retrieve top-20 documents
                try:
                    retrieved_docs = set(retriever.retrieve(query_id, top_k=20))
                except Exception as e:
                    self.warnings.append(f"Retrieval hiba query-re '{query_id[:50]}...': {e}")
                    continue

                # Calculate recall
                found_relevant = relevant_docs & retrieved_docs
                recall = len(found_relevant) / len(relevant_docs) if relevant_docs else 0.0

                if recall == 0:
                    zero_recall_queries.append((query_id, len(relevant_docs)))
                elif recall < MIN_RECALL_THRESHOLD:
                    low_recall_queries.append((query_id, recall, len(relevant_docs)))

            print(f"\r   ✅ {total_queries} query ellenőrizve")

            # Report findings
            if zero_recall_queries:
                self.errors.append(
                    f"{len(zero_recall_queries)} query-nél NULLA recall (egyetlen releváns dok sem találva top-20-ban):"
                )
                for query, num_relevant in zero_recall_queries[:5]:
                    self.errors.append(
                        f"  - '{query[:50]}...' ({num_relevant} releváns dok a qrels-ben)"
                    )
                if len(zero_recall_queries) > 5:
                    self.errors.append(f"  ... és még {len(zero_recall_queries) - 5} query")

                self.errors.append(
                    "\n  💡 Ez azt jelenti hogy a retrieval nem találja meg a releváns dokumentumokat!"
                )
                self.errors.append(
                    "     Lehetséges okok:"
                )
                self.errors.append(
                    "     - Query és dokumentum szövege túl különböző (vocabulary mismatch)"
                )
                self.errors.append(
                    "     - BM25/FAISS index minősége gyenge"
                )
                self.errors.append(
                    "     - Releváns dokumentumok hibásan vannak annotálva a qrels-ben"
                )
                return False

            if low_recall_queries:
                self.warnings.append(
                    f"{len(low_recall_queries)} query-nél alacsony recall (< {MIN_RECALL_THRESHOLD*100:.0f}%):"
                )
                for query, recall, num_relevant in low_recall_queries[:5]:
                    self.warnings.append(
                        f"  - '{query[:50]}...' (recall={recall:.1%}, {num_relevant} releváns dok)"
                    )
                if len(low_recall_queries) > 5:
                    self.warnings.append(f"  ... és még {len(low_recall_queries) - 5} query")

            # Success message
            good_recall_count = total_queries - len(zero_recall_queries) - len(low_recall_queries)
            if good_recall_count > 0:
                print(f"   ✅ {good_recall_count} query megfelelő recall-lal (≥ {MIN_RECALL_THRESHOLD*100:.0f}%)")

            return True

        except ImportError as e:
            self.warnings.append(
                f"Retrieval check kihagyva (import hiba): {e}"
            )
            return True
        except Exception as e:
            self.warnings.append(
                f"Retrieval check kihagyva (hiba): {e}"
            )
            return True

    def print_summary(self):
        """Összefoglaló jelentés kiírása."""
        print("\n" + "=" * 60)
        print("📊 ÖSSZEFOGLALÁS")
        print("=" * 60)

        print(f"\n✅ Sikeres ellenőrzések: {len(self.qrels_data)} sor feldolgozva")
        print(f"   • {len({e['query_id'] for e in self.qrels_data})} egyedi query")
        print(f"   • {len({e['doc_id'] for e in self.qrels_data})} egyedi doc_id")

        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} figyelmeztetés:")
            for warning in self.warnings:
                print(f"   - {warning}")

        if self.errors:
            print(f"\n❌ {len(self.errors)} hiba:")
            for error in self.errors:
                print(f"   - {error}")
            print("\n❌ VALIDÁCIÓ SIKERTELEN")
            return False
        else:
            print("\n✅ VALIDÁCIÓ SIKERES!")
            print("\n🚀 Következő lépés:")
            print("   Futtasd a baseline_evaluation.ipynb notebookot")
            return True

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Qrels validáló - ellenőrzi a qrels fájl formátumát és konzisztenciáját"
    )
    parser.add_argument(
        "--qrels-file",
        type=str,
        default=None,
        help="Qrels fájl path (alapértelmezett: baseline_qrels.tsv)",
    )
    parser.add_argument(
        "--check-retrieval",
        action="store_true",
        help="Ellenőrzi hogy a retrieval megtalálja-e a releváns dokumentumokat (lassabb, opcionális)",
    )
    args = parser.parse_args()

    # Use custom qrels file if provided
    if args.qrels_file:
        custom_qrels_path = project_root / args.qrels_file
        if not custom_qrels_path.exists():
            custom_qrels_path = Path(args.qrels_file)
        validator = QrelsValidator(custom_qrels_path, chunks_path)
    else:
        validator = QrelsValidator(qrels_path, chunks_path)

    success = validator.validate_all(check_retrieval=args.check_retrieval)

    if args.check_retrieval:
        print("\n💡 TIP: A retrieval check opcionális és lassú.")
        print("   Normál validációhoz futtasd paraméter nélkül: python validate_qrels.py")

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
