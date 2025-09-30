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
- Indexes: BM25 (sparse) and FAISS (dense) with FAISS↔doc ID mapping.
- BM25 artifacts: bm25s model directory (scores/indices/indptr/vocab/params), serialized doc_id list, token statistics (length distribution, total docs), optional token-id cache for fast rebuilds.
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
- ID mapping: FAISS row index ↔ doc_id in a separate file.

4) Hybrid Retrieval (baseline)
- Embed the query with the same model.
- Retrieve top-K candidates from BM25 (document-level scoring via bm25s) and FAISS.
- Fuse via RRF or z‑score weighted sum (handle zero variance).
- Aggregate chunk-level results to document level, keep per-source score features, output top-k list of Document IDs (baseline).

5) RL Reranking (GRPO‑style)
- Candidate context: for every slate element collect document text (trimmed to config length), document metadata (court, domain, year), RRF/BM25/FAISS scores, and rank positions; surface these as structured JSON snippets embedded in the prompt alongside the Hungarian query.
- Slate preparation: merge the top‑`k` BM25 and FAISS hits into a fixed-length slate per query, pad with neutral placeholders when necessary, and attach ground-truth relevance labels pulled from the qrels file.
- Policy model: fine-tune `Qwen/Qwen3-4B-Instruct-2507` with QLoRA adapters (bf16 compute, 4-bit weights) so it can score slate items via autoregressive logits; groupwise softmax over candidate scores yields the reranking distribution.
- Trainer: run Hugging Face TRL `GRPOTrainer` with group size matched to the slate length, KL penalty disabled, learning-rate warmup + gradient clipping, and per-step logging of Hungarian status messages; rewards are computed on-device from the provided relevance labels.
- Reward shaping: evaluate nDCG@10 for both the baseline order and the policy's sampled order at document level, use their difference as the reward, clamp negative rewards when no relevant documents exist, and default to zero if the slate lacks annotations.
- Regularization: apply entropy bonus and normalize rewards by the query-level variance (fallback 1.0) to keep QLoRA training stable on limited VRAM.
- Outputs: save LoRA adapter weights and tokenizer configs under `data/models/grpo_policy/`, emit reranked identifier lists per query, and log baseline vs. reranked nDCG/MAP metrics to `data/models/grpo_policy/metrics.json`.

GRPO Implementation Plan (TRL-based)
- Environment setup: install/update `trl`, `transformers`, `peft`, and `bitsandbytes`; keep configs ready for CPU/MPS inference while enabling bf16 + 4bit QLoRA training on cloud GPUs without altering the CLI surface.
- Data pipeline: reuse hybrid retrieval exports to materialize per-query candidate slates and save them as JSONL batches that the cloud GRPO notebook can ingest (IDs stay in sync with `chunks.jsonl` and qrels).
- Model preparation: load `Qwen/Qwen3-4B-Instruct-2507` with `load_in_4bit=True`, attach QLoRA adapters (rank/alpha set in config), and register reward/eval prompts that stay within the model’s context window.
- Cloud training notebook: create `notebooks/grpo_train_runpod.ipynb` that loads the slate JSONL, configures `GRPOTrainer` (group size = slate length, custom nDCG-based `reward_fn`), streams Hungarian progress logs, and checkpoints adapters into `/workspace/artifacts/grpo_policy/` for download.
- Artifact handover: after cloud training, download the adapter weights, tokenizer files, and metrics JSON into `data/models/grpo_policy/` locally; commit to a deterministic filename schema for seamless loading.
- Local inference integration: update `GRPOReranker` so it fetches the downloaded adapters, runs forward passes on CPU/MPS with low-memory settings, and gracefully falls back to baseline ordering if artifacts are missing.
- Portability: document the Runpod workflow (environment variables, mixed-precision flags, artifact sync back to the repo) to ensure the same policy runs locally post-training.
- Baseline compatibility: no changes required for ingestion, chunking, or hybrid retrieval modules (`src/data_loader/build_bm25_index.py`, `src/search/hybrid_search.py`); only the RL layer consumes the new artifacts.
- CLI alignment: adapt `src/cli.py` so training invocations dispatch to the cloud notebook (artifact sync) and querying loads the LoRA adapters before reranking, while maintaining Hungarian console output.
- Slate export: add a utility that serializes baseline candidate slates (document text + metadata + scores) into JSONL for GRPO training, ensuring document IDs stay aligned with qrels and FAISS mappings.
- Qrels consistency: harmonize configuration (`configs/config.py`) with the canonical `data/qrels/baseline_qrels.tsv` path to avoid mismatched formats between CLI and notebook.

BM25S Implementation Notes
- Tokenization: rely on `bm25s.tokenize` with configurable stopword handling (default none for Hungarian); persist the returned token-id structures for reuse.
- Index persistence: always call `BM25.save()` into `data/index/bm25/`, store `doc_ids.json` plus `bm25_stats.json` (total_docs, avg_length, histogram) for downstream consumers.
- Corpus structure: BM25S stores corpus documents as `{'id': index, 'text': chunk_id}` dictionaries; retrieval results must extract chunk_id from the 'text' field, not the 'id' field (which is just a numeric index).
- Search results: the `search()` method returns chunk_id strings that are later aggregated to document-level doc_id identifiers via suffix stripping (e.g., "doc_0" → "doc").
- Scoring utilities: prefer `get_scores` / `get_scores_from_ids` to avoid redundant retrieval when exporting features for GRPO.
- Performance: allow enabling `activate_numba_scorer()` and thread count tuning via CLI flags for local CPU optimization on M3 hardware.

6) CLI Workflows
- Build: Docling parse → normalization → chunking → BM25.
- Embedding: Generate FAISS index using gemma_embedding_runpod.ipynb.
- Query: embed → BM25+FAISS → document aggregation → fusion → top‑k document ID list (optional RL rerank).
- RL Train: load qrels → baseline candidates → features → GRPO training → save policy → evaluate.

Relevance Judgments (Qrels)
- Location: store in `data/qrels/baseline_qrels.tsv` and keep the filename stable for the RL training CLI.
- Format: tab-separated with a header row `query_id\tdoc_id\trelevance`; use numeric `query_id` values, document identifiers that match the `doc_id` field in `chunks.jsonl`, and relevance grades in {0,1,2}.
- Coverage: provide at least 50 unique queries, each with three or more annotated documents; ensure every relevant document references the same underlying document IDs used elsewhere in the pipeline.
- Consistency: maintain UTF-8 encoding, no BOM, Unix newlines, and keep the file sorted first by `query_id`, then by descending `relevance` to simplify deterministic loading.

Acceptance Criteria
- Always use the full dataset for the final pipeline.
- Build produces: chunks JSONL, BM25 index.
- Notebook produces: FAISS index and ID mapping.
- Query returns: fused top‑k list of document IDs (e.g., 0302-G_20416_2019_11).
- RL training saves a policy and improves metrics vs baseline (e.g., nDCG@10 at document level).
- A real embedding model is used (HF token from .env), not a stub.

Pitfalls
- Always L2‑normalize embeddings (required for FAISS with IP).
- Handle zero variance in z‑score normalization.
- Maintain correct FAISS row ↔ doc_id ↔ relevance mapping at document level.
- Return 0 for nDCG when no relevant results exist.
- Keep document aggregation, top‑K, fusion, and RL hyperparameters configurable.
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
- doc_id_map.json: FAISS row index → doc_id

Language Policy
- This file (AGENTS.md) must be written in English only.
- All other project outputs and user‑facing text must be in Hungarian.
- All CLI messages, result lists, reports, and printed strings must be Hungarian.
- Query responses must return only Hungarian content where applicable; when returning identifiers, return IDs only as requested.

Sources to implement GRPO based RL
- https://huggingface.co/learn/llm-course/en/chapter12/4
- https://huggingface.co/docs/trl/main/en/grpo_trainer
- https://huggingface.co/learn/llm-course/en/chapter12/5?fw=pt
