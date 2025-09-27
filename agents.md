Agent Specification

Project Goal

Build a compute‑light, locally executable (M3 Macbook Air 16 GB RAM) retrieval pipeline for Hungarian court decisions. The system must use a real embedding model and demonstrate measurable ranking improvements via GRPO‑style RL reranking. Keep the solution minimal, reproducible, and easy to extend.

Out of Scope
- No PDF parsing.
- No stemming/lemmatization.
- No query rewrite.
- No tokenization libraries like tiktoken.
- No logging, testing, or deployment infrastructure.

Inputs
- Raw documents: DOCX court decisions (directory structure: court/domain/case_id/file).

Outputs
- Chunks JSONL: normalized, segmented text with minimal metadata.
- Indexes: BM25 (sparse) and FAISS (dense) with FAISS↔chunk ID mapping.
- BM25 artifacts: bm25s model directory (scores/indices/indptr/vocab/params), serialized chunk_id list, token statistics (length distribution, total docs), optional token-id cache for fast rebuilds.
- Policy: parameters of the RL reranking model.
- Query result: top‑k list of identifiers only (baseline and reranked).

High‑Level Pipeline
1) Ingestion with Docling
- Docling parses DOCX to plain text.
- Minimal normalization (whitespace, control characters); keep legal clauses.
- Metadata extraction: court, domain, year, case identifier from filename/folders.

2) Chunking
- Let Docling decide on chunking and overlap size, use it's capabilites for that.
- Preserve metadata on chunks.
- Output: chunks JSONL.

3) Indexing
- BM25: use the BM25s library (https://bm25s.github.io/, https://huggingface.co/blog/xhluca/bm25s) with its native tokenizer; cache token ids, vocabulary, and length metadata alongside the bm25s index files so rebuilds do not re-tokenize the corpus.
- FAISS: HF EmbeddingGemma - https://huggingface.co/google/embeddinggemma-300m (HF token from .env); L2‑normalized embeddings with IP metric; adaptive IVF training with a train buffer; `nlist` matched to sample size; add pending batches post‑training.
- ID mapping: FAISS row index ↔ chunk_id in a separate file.

4) Hybrid Retrieval (baseline)
- Embed the query with the same model.
- Retrieve top-K candidates from BM25 (chunk-level scoring via bm25s) and FAISS.
- Fuse via RRF or z‑score weighted sum (handle zero variance).
- Convert chunk IDs to document IDs consistently, keep per-source score features, output top-k list of Document IDs (baseline).

5) RL Reranking (GRPO‑style)
- Features per candidate: dense similarity, normalized BM25 score (pulled from cached bm25s scores), rank difference, token-length features, optionally simple metadata features.
- Policy: linear or shallow MLP head; groupwise softmax over candidates.
- Reward: nDCG@10 at group level; group‑relative reward versus the baseline.
- Output: saved policy; reranked results; numeric comparison baseline vs rerank.

BM25S Implementation Notes
- Tokenization: rely on `bm25s.tokenize` with configurable stopword handling (default none for Hungarian); persist the returned token-id structures for reuse.
- Index persistence: always call `BM25.save()` into `data/index/bm25/`, store `chunk_ids.json` plus `bm25_stats.json` (total_docs, avg_length, histogram) for downstream consumers.
- Scoring utilities: prefer `get_scores` / `get_scores_from_ids` to avoid redundant retrieval when exporting features for GRPO.
- Performance: allow enabling `activate_numba_scorer()` and thread count tuning via CLI flags for local CPU optimization on M3 hardware.

6) CLI Workflows
- Build: Docling parse → normalization → chunking → BM25.
- Embedding: Generate FAISS index using gemma_embedding_runpod.ipynb.
- Query: embed → BM25+FAISS → fusion → top‑k ID list (optional RL rerank).
- RL Train: load qrels → baseline candidates → features → GRPO training → save policy → evaluate.

Acceptance Criteria
- Always use the full dataset for the final pipeline.
- Build produces: chunks JSONL, BM25 index.
- Notebook produces: FAISS index and ID mapping.
- Query returns: fused top‑k list of chunk IDs.
- RL training saves a policy and improves metrics vs baseline (e.g., nDCG@10).
- A real embedding model is used (HF token from .env), not a stub.

Pitfalls
- Always L2‑normalize embeddings (required for FAISS with IP).
- Handle zero variance in z‑score normalization.
- Maintain correct FAISS row ↔ chunk_id ↔ relevance mapping.
- Return 0 for nDCG when no relevant results exist.
- Keep chunk size/overlap, top‑K, fusion, and RL hyperparameters configurable.
- Ensure FAISS index is generated using gemma_embedding_runpod.ipynb before querying.

Configuration & Execution
- Fully local execution; EmbeddingGemma model for document and query embeddings.
- Optimized for M3 MacBook Air with MPS (Metal Performance Shaders) acceleration.
- Centralized configuration in `configs/config.py`; values tuned for PoC.
- Reproducible outputs; simple JSON/JSONL artifacts.

Data Formats (outline)
- chunks.jsonl: {chunk_id, doc_id, text, court, domain, year, source_path}
- bm25_index.json: minimal postings and statistics structure
- faiss_index.bin: FAISS index file
- chunk_id_map.json: FAISS row index → chunk_id

Language Policy
- This file (AGENTS.md) must be written in English only.
- All other project outputs and user‑facing text must be in Hungarian.
- All CLI messages, result lists, reports, and printed strings must be Hungarian.
- Query responses must return only Hungarian content where applicable; when returning identifiers, return IDs only as requested.

Sources to implement GRPO based RL
- https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide
- https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo
- https://github.com/wangqinsi1/GAINRL?tab=readme-ov-file
- https://huggingface.co/learn/llm-course/en/chapter12/4
- https://huggingface.co/docs/trl/main/en/grpo_trainer