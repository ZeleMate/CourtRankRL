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
- BM25: tokenization via simple `split()`; store postings, doc_len/avg_len, idf cache.
- FAISS: HF EmbeddingGemma (HF token from .env); L2‑normalized embeddings with IP metric; adaptive IVF training with a train buffer; `nlist` matched to sample size; add pending batches post‑training.
- ID mapping: FAISS row index ↔ chunk_id in a separate file.

4) Hybrid Retrieval (baseline)
- Embed the query with the same model.
- Retrieve top‑K candidates from BM25 and FAISS.
- Fuse via RRF or z‑score weighted sum (handle zero variance).
- Output: top‑k list of Document IDs (baseline).

5) RL Reranking (GRPO‑style)
- Features per candidate: dense similarity, normalized BM25 score, rank difference; optionally simple metadata features.
- Policy: linear or shallow MLP head; groupwise softmax over candidates.
- Reward: nDCG@10 at group level; group‑relative reward versus the baseline.
- Output: saved policy; reranked results; numeric comparison baseline vs rerank.

6) CLI Workflows
- Build: Docling parse → normalization → chunking → BM25.
- Embedding: Generate FAISS index using qwen_embedding_runpod.ipynb.
- Query: embed → BM25+FAISS → fusion → top‑k ID list (optional RL rerank).
- RL Train: load qrels → baseline candidates → features → GRPO training → save policy → evaluate.

Acceptance Criteria
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
- Ensure FAISS index is generated using qwen_embedding_runpod.ipynb before querying.

Configuration & Execution
- Fully local execution; Qwen3-Embedding-0.6B model for document and query embeddings.
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
