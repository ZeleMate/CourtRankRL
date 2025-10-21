#!/usr/bin/env python3
"""
AI Annotátor - GPT-5 mini relevancia pontozás

Egyszerűsített script, amely csak annotálást végez:
- Input: baseline_qrels.tsv (1000 query × 20 doc = 20,000 sor)
- Output: annotated_qrels.tsv (0/1/2 relevancia értékekkel)
- Few-shot példák: baseline_qrels_golden_set.tsv
- OpenAI Responses API (nem Batch API)

Futtatás: python scripts/ai_annotation.py
"""

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Konfiguráció
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_QRELS = PROJECT_ROOT / "data/qrels/baseline_qrels_golden_set.tsv"
INPUT_QRELS = PROJECT_ROOT / "data/qrels/baseline_qrels.tsv"
OUTPUT_QRELS = PROJECT_ROOT / "data/qrels/annotated_qrels.tsv"
CHUNKS_JSONL = PROJECT_ROOT / "data/processed/chunks.jsonl"

MODEL_NAME = "gpt-5-mini-2025-08-07"
BATCH_SIZE = 15
FEWSHOT_SAMPLES = 3  # 3-3-3 = 9 példa

# Kulcsszó extrakcióhoz
KEYWORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)

# Environment setup
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ HIBA: OPENAI_API_KEY környezeti változó nincs beállítva!")
    print("Állítsd be a .env fájlban: OPENAI_API_KEY=sk-...")
    exit(1)

def normalize_qrels_file(path: Path) -> None:
    """Garantálja, hogy a qrels TSV pontosan 3 oszlopot tartalmazzon."""
    if not path.exists():
        return

    normalized_lines: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, raw_line in enumerate(f):
            line = raw_line.strip("\n\r")
            if not line:
                continue

            parts = [part.strip() for part in line.split("\t") if part is not None]

            if idx == 0 and parts and parts[0].lower() == "query_id":
                normalized_lines.append("query_id\tdoc_id\trelevance")
                continue

            if len(parts) < 2:
                continue

            query_id = parts[0]
            doc_id = parts[1]
            relevance = parts[2] if len(parts) > 2 and parts[2] else "0"

            try:
                relevance_int = int(str(relevance).strip())
            except (TypeError, ValueError):
                relevance_int = 0

            normalized_lines.append(f"{query_id}\t{doc_id}\t{relevance_int}")

    if not normalized_lines:
        return

    current_content = path.read_text(encoding="utf-8").strip()
    normalized_content = "\n".join(normalized_lines)

    if current_content == normalized_content:
        return

    path.write_text(normalized_content + "\n", encoding="utf-8")


def load_qrels(path: Path) -> pd.DataFrame:
    """Betölti a qrels TSV fájlt."""
    print(f"   📖 {path.name} betöltése...")

    # Normalizálás: biztosítjuk a 3 oszlopos formátumot
    normalize_qrels_file(path)

    df = pd.read_csv(
        path,
        sep="\t",
        usecols=[0, 1, 2],
        names=["query_id", "doc_id", "relevance"],
        header=0,
        dtype={"query_id": str, "doc_id": str, "relevance": "Int64"},
    )

    # Ha valahol hiányzó érték maradt, töltsük 0-val
    df["relevance"] = df["relevance"].fillna(0).astype(int)

    print(f"   ✅ {len(df)} sor, {df['query_id'].nunique()} query")
    return df

    #TODO: implement keyword search for each query so that we select the appropriate chunks for each document


def load_chunks_mapping() -> dict:
    """Betölti a chunks.jsonl-t és létrehozza a doc_id → chunk lista mapping-et."""
    print(f"   📖 {CHUNKS_JSONL.name} betöltése...")

    doc_id_to_chunks = defaultdict(list)
    chunk_count = 0

    for chunk_df in pd.read_json(CHUNKS_JSONL, lines=True, chunksize=50000):
        for _, row in chunk_df.iterrows():
            doc_id = row["doc_id"]

            # Store all chunks per doc
            doc_id_to_chunks[doc_id].append({
                "chunk_id": row["chunk_id"],
                "text": row["text"]
            })

            chunk_count += 1
            if chunk_count % 100000 == 0:
                print(f"   🔄 Feldolgozva: {chunk_count:,} chunk...")

    print(f"   ✅ {len(doc_id_to_chunks):,} egyedi doc_id betöltve")
    return dict(doc_id_to_chunks)


def extract_keywords(text: str) -> List[str]:
    """Egyszerű kulcsszó extrakció: kisbetűs, rövid szavak szűrése."""
    tokens = [token.lower() for token in KEYWORD_PATTERN.findall(text)]
    keywords = [token for token in tokens if len(token) > 2]
    return keywords if keywords else tokens


def select_relevant_chunk(query_text: str, chunks: List[dict]) -> tuple[str, str]:
    """Kulcsszó alapú chunk kiválasztás egy dokumentumhoz."""
    if not chunks:
        return "", ""

    keywords = extract_keywords(query_text)
    if not keywords:
        best_chunk = chunks[0]
        return best_chunk["chunk_id"], best_chunk["text"]

    keyword_set = set(keywords)
    best_chunk = chunks[0]
    best_score = -1
    best_first_pos = float("inf")

    for chunk in chunks:
        text = chunk["text"]
        lower_text = text.lower()
        match_count = 0
        first_pos = float("inf")

        for keyword in keyword_set:
            pos = lower_text.find(keyword)
            if pos != -1:
                match_count += 1
                if pos < first_pos:
                    first_pos = pos

        if (
            match_count > best_score
            or (match_count == best_score and first_pos < best_first_pos)
        ):
            best_chunk = chunk
            best_score = match_count
            best_first_pos = first_pos

        if best_score == len(keyword_set):
            break

    return best_chunk["chunk_id"], best_chunk["text"]


def generate_fewshot_examples(golden_df: pd.DataFrame, chunks_map: dict, samples: int = 3) -> str:
    """
    Stratifikált mintavételezés: 3-3-3 példa (0,1,2 relevancia).

    Args:
        golden_df: baseline_qrels_golden_set.tsv DataFrame
        chunks_map: {doc_id: List[chunk_dict]}
        samples: példák száma/szint

    Returns:
        Formázott few-shot string
    """
    print(f"   📝 Few-shot példák generálása ({samples} példa/relevancia szint)...")

    examples = []
    relevance_labels = {0: "NOT RELEVANT", 1: "PARTIALLY RELEVANT", 2: "HIGHLY RELEVANT"}

    for rel in [0, 1, 2]:
        subset = golden_df[golden_df["relevance"] == rel]
        if len(subset) >= samples:
            sample_rows = subset.sample(samples, random_state=42)

            for _, row in sample_rows.iterrows():
                query = row["query_id"]
                doc_id = row["doc_id"]
                rel_val = row["relevance"]

                # Select most relevant chunk for this query-doc pair
                chunks = chunks_map.get(doc_id, [])
                _, text = select_relevant_chunk(query, chunks)

                # Truncate if too long but keep meaningful content
                if len(text) > 800:
                    text = text[:800] + "..."

                examples.append(
                    f"--- EXAMPLE (Relevance {rel_val} - {relevance_labels[rel_val]}) ---\n"
                    f"Query: \"{query}\"\n"
                    f"Document ID: {doc_id}\n"
                    f"Document Content: \"{text}\"\n"
                    f"Correct Output: {query}\\t{doc_id}\\t{rel_val}\n"
                )

    print(f"   ✅ {len(examples)} few-shot példa generálva")
    return "\n\n".join(examples)


def get_annotation_system_prompt(fewshot_examples: str) -> str:
    """System prompt annotáláshoz - optimalizált magyar jogi kontextusra."""
    return f"""You are an expert legal document relevance annotator specializing in Hungarian court decisions. Your task is to evaluate the relevance of court decision excerpts to legal search queries with high precision.

TASK: Annotate query-document pairs with relevance scores (0/1/2) based on substantive legal relevance.

RELEVANCE SCALE - DETAILED CRITERIA:

0 - NOT RELEVANT:
  • Different legal domain (e.g., query about criminal law, document about civil contracts)
  • Different legal issue (e.g., query about employment termination, document about product liability)
  • No semantic overlap in legal concepts
  • Document might share 1-2 generic terms but addresses completely different matter
  • Example: Query "munkaviszony megszüntetés" vs Document about "adásvételi szerződés"

1 - PARTIALLY RELEVANT:
  • Same or related legal domain (e.g., both criminal law or both civil law)
  • Shares important legal concepts but NOT the specific issue queried
  • Contains relevant procedural elements (e.g., appeal, evidence) but different substantive matter
  • Related legal remedy or consequence but different factual situation
  • Example: Query "munkaviszony azonnali megszüntetés" vs Document about "munkaviszony rendes felmondás"
  • Example: Query "tulajdonjog bejegyzés" vs Document about "használati jog bejegyzés"

2 - HIGHLY RELEVANT:
  • DIRECTLY addresses the specific legal issue in the query
  • Same legal domain AND same substantive legal question
  • Document provides applicable legal reasoning, rules, or precedent for the query
  • Contains the key legal institution/concept AND its specific application context
  • User would find this document highly useful for understanding the queried legal issue
  • Example: Query "munkaviszony jogellenes megszüntetés kártérítés" vs Document discussing unlawful employment termination and damages
  • Example: Query "fellebbezés elkésett benyújtás" vs Document about late appeal submission and consequences

CRITICAL EVALUATION PRINCIPLES:

1. SEMANTIC DEPTH OVER KEYWORD MATCHING:
   - Don't assign relevance 2 just because several words match
   - Focus on whether the LEGAL ISSUE and its CONTEXT match
   - "kártérítés" (damages) appears in many cases - check if the underlying cause matches

2. LEGAL DOMAIN CONSISTENCY:
   - Criminal law (büntetőjog): bűncselekmény, bűntett, vétség, szabadságvesztés, óvadék
   - Civil law (polgári jog): szerződés, kártérítés, tulajdonjog, felmondás
   - Administrative law (közigazgatási jog): hatósági határozat, jogorvoslat
   - Labor law (munkajog): munkaviszony, felmondás, végkielégítés

3. PROCEDURAL VS SUBSTANTIVE:
   - Procedural terms (eljárási): fellebbezés, jogorvoslat, bizonyítás, határidő
   - These can indicate relevance 1 if procedural context matches but substance differs
   - Assign relevance 2 only if BOTH procedure AND substance align

4. SPECIFICITY MATTERS:
   - Generic query + Generic document = likely 1 (not 2)
   - Specific query + Matching specific document = 2
   - Specific query + Generic document = likely 0 or 1

5. HUNGARIAN LEGAL TERMINOLOGY:
   - Understand compound legal terms: "munkaviszony megszüntetés" is ONE concept
   - "azonnali hatályú megszüntetés" vs "rendes felmondás" are DIFFERENT concepts
   - "előzetes letartóztatás" vs "óvadék ellenében szabadlábon hagyás" are related but distinct

CONTEXT: Hungarian court decisions covering civil, criminal, labor, administrative, and commercial law matters.

OUTPUT FORMAT: TSV rows ONLY (no explanations, no headers, no extra text)
query_id<TAB>doc_id<TAB>relevance

Each line must contain exactly 3 fields separated by TAB characters.

FEW-SHOT EXAMPLES:
{fewshot_examples}

EVALUATION CHECKLIST FOR EACH DOCUMENT:
1. Does the document address the SAME legal issue as the query?
2. Is the legal domain/field consistent?
3. Are the key legal concepts and their application context aligned?
4. Would a legal professional find this document useful for the query?
5. If uncertain between 1 and 2, prefer 1 (be conservative with score 2)

IMPORTANT: Base your judgment on LEGAL SUBSTANCE and CONTEXT, not superficial keyword overlap. Think like a legal researcher seeking applicable precedent or guidance."""


def get_annotation_user_prompt(query_text: str, documents: list) -> str:
    """
    User prompt annotáláshoz.
    
    Args:
        query_text: Query szöveg
        documents: Lista dict-ekről: {doc_id, chunk_text}
    
    Returns:
        Formázott user prompt
    """
    prompt = f"LEGAL SEARCH QUERY TO EVALUATE:\n\"{query_text}\"\n\n"
    prompt += f"TASK: Evaluate the relevance of {len(documents)} court decision excerpts to this query.\n\n"
    prompt += "=" * 80 + "\n\n"
    
    for i, doc in enumerate(documents, 1):
        prompt += f"DOCUMENT {i}:\n"
        prompt += f"  ID: {doc['doc_id']}\n"
        if 'chunk_id' in doc and doc['chunk_id']:
            prompt += f"  Chunk ID: {doc['chunk_id']}\n"
        prompt += f"  Content:\n  \"\"\"\n  {doc['chunk_text']}\n  \"\"\"\n\n"
        prompt += "-" * 80 + "\n\n"
    
    prompt += "OUTPUT INSTRUCTIONS:\n"
    prompt += "Return EXACTLY one TSV row per document in this format:\n"
    prompt += "query_text<TAB>doc_id<TAB>relevance_score\n\n"
    prompt += "Where relevance_score is 0, 1, or 2 based on the criteria provided.\n"
    prompt += "NO headers, NO explanations, NO extra text.\n\n"
    prompt += "TSV OUTPUT:"
    return prompt


def annotate_all(input_df: pd.DataFrame, chunks_map: dict, fewshot: str, output_path: Path) -> list:
    """
    Annotálja az összes query-doc párt batch-ekben streaming kimenettel.

    Returns:
        [(query_id, doc_id, relevance), ...]
    """
    print(f"   🔍 Annotálás indítása ({len(input_df)} sor)...")

    client = OpenAI(api_key=OPENAI_API_KEY)
    system_prompt = get_annotation_system_prompt(fewshot)

    results = []
    total_cost = 0.0
    processed_queries = 0

    # Inicializálja a kimeneti fájlt
    save_results([], output_path, mode="w")

    # Group by query
    total_queries = input_df["query_id"].nunique()
    for query_id, group in input_df.groupby("query_id"):
        doc_ids = group["doc_id"].tolist()
        processed_queries += 1

        print(f"   🔄 Query {processed_queries}/{total_queries}: '{query_id[:50]}...' ({len(doc_ids)} docs)")

        query_results = []

        # Batch documents
        for i in range(0, len(doc_ids), BATCH_SIZE):
            batch_docs = doc_ids[i:i+BATCH_SIZE]

            # Prepare docs with relevant chunk selection
            documents = []
            for doc_id in batch_docs:
                chunk_list = chunks_map.get(doc_id, [])
                chunk_id, chunk_text = select_relevant_chunk(query_id, chunk_list)
                documents.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_text
                })

            user_prompt = get_annotation_user_prompt(query_id, documents)

            # API call
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=2000
                )

                # Parse TSV output
                content = response.choices[0].message.content
                for line in content.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split("\t")
                    if len(parts) == 3:
                        q, d, r = parts
                        try:
                            relevance = int(r)
                            if relevance in {0, 1, 2}:
                                query_results.append((q, d, relevance))
                                results.append((q, d, relevance))
                        except ValueError:
                            continue

                # Cost tracking
                cost = (response.usage.prompt_tokens * 0.025 / 1_000_000 +
                        response.usage.completion_tokens * 0.2 / 1_000_000)
                total_cost += cost

                print(f"     ✓ {len(batch_docs)} docs annotálva (${cost:.3f})")

            except Exception as e:
                print(f"     ✗ Hiba: {e}")

        # Query feldolgozása után azonnal mentés
        if query_results:
            save_results(query_results, output_path, mode="a")
            print(f"   💾 Query {processed_queries} eredményei mentve ({len(query_results)} sor)")

    print(f"\n   💰 Összes költség: ${total_cost:.2f}")
    return results


def save_results(results: list, output_path: Path, mode: str = "w") -> None:
    """TSV mentés: query_id\tdoc_id\trelevance"""
    if not results:
        return

    df = pd.DataFrame(results, columns=["query_id", "doc_id", "relevance"])

    # Header csak új fájl esetén
    header = mode == "w"
    df.to_csv(output_path, sep="\t", index=False, encoding="utf-8", lineterminator="\n",
              mode=mode, header=header)

    print(f"   💾 {len(results)} sor {'mentve' if mode == 'a' else 'inicializálva'}: {output_path}")


def print_stats(results: list) -> None:
    """Egyszerű statisztika printelés."""
    df = pd.DataFrame(results, columns=["query_id", "doc_id", "relevance"])
    
    print(f"\n📊 Statisztikák:")
    print(f"  Összes sor: {len(df)}")
    print(f"  Unique query: {df['query_id'].nunique()}")
    print(f"  Relevancia eloszlás:")
    for rel, count in df["relevance"].value_counts().sort_index().items():
        pct = 100 * count / len(df)
        print(f"    {rel}: {count} ({pct:.1f}%)")


def main():
    print("🚀 AI Annotátor indítás")
    print("=" * 60)
    
    # 1. Load data
    print("📖 Adatok betöltése...")
    golden_df = load_qrels(GOLDEN_QRELS)
    input_df = load_qrels(INPUT_QRELS)
    chunks_map = load_chunks_mapping()
    
    # 2. Few-shot examples
    print("\n📝 Few-shot példák generálása...")
    fewshot = generate_fewshot_examples(golden_df, chunks_map, FEWSHOT_SAMPLES)
    
    # 3. Annotate
    print(f"\n🔍 Annotálás ({len(input_df)} sor)...")
    results = annotate_all(input_df, chunks_map, fewshot, OUTPUT_QRELS)

    # 4. Végső összesítés (minden eredmény már mentve)
    print("\n💾 Összes eredmény már streamingben mentve!")
    print(f"   ✅ Teljes eredmény: {len(results)} annotáció")
    
    # 5. Stats
    print_stats(results)
    
    print("\n✅ Kész!")
    print(f"\n💡 Következő lépés:")
    print(f"   python scripts/validate_qrels.py --qrels-file {OUTPUT_QRELS}")


if __name__ == "__main__":
    main()
