Agent Specification

Project Goal

Build a compute‑light, locally executable (M3 Macbook Air 16 GB RAM) retrieval pipeline for Hungarian court decisions. The system must use a real embedding model and demonstrate measurable ranking improvements via GRPO‑style RL reranking. Keep the solution minimal, reproducible, and easy to extend.

Agent Summaries

Whenever an AI agent creates a summary of the proposed changes, put that files into the /CourtRankRL/data/Agent summaries folder. Read the necessarry docs from that folder.

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
1) Ingestion & Chunking
- **Parsing**: Use Docling to convert DOCX court decisions to plain text with minimal normalization (whitespace, control characters); preserve legal clause structure.
- **Metadata extraction**: Extract court, domain, year, case identifier from directory structure and filenames.
- **Chunking**: Leverage Docling's native chunking capabilities for automatic segmentation with appropriate overlap; preserve all metadata on each chunk.
- **Output**: `chunks.jsonl` with complete document and chunk metadata.

2) Indexing
- **BM25**: Use BM25s library (https://bm25s.github.io/) with native tokenizer; apply case-insensitive tokenization (lowercase corpus and queries for Hungarian); persist token IDs, vocabulary, and statistics alongside index for fast rebuilds without re-tokenization.
- **FAISS Embedding Generation** (Cloud GPU): Use `google/embeddinggemma-300m` (HF token from .env) via **Sentence Transformers library** (>=5.1.0); CRITICAL: EmbeddingGemma requires Sentence Transformers (not AutoModel from transformers); use `model.encode(texts, prompt_name="document", normalize_embeddings=True)` which automatically adds prompt "title: none | text: {chunk}", performs mean pooling with attention mask, and L2-normalizes embeddings; model max_seq_length is 2048 tokens (auto-handled); process chunks in batches on RTX 5090 GPU; output consolidated `embeddings.npy` (float32, L2-normalized) and `embedding_chunk_ids.json` for artifact download. IMPORTANT: Do NOT use transformers.AutoModel - it produces zero-vector embeddings that break FAISS indexing.
- **FAISS Index Building** (Local/Cloud): Load pre-computed embeddings from artifacts; use OPQ64_256,IVF65536,PQ64x4fsr (https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index) for optimal memory/accuracy balance (product quantization for memory efficiency); configure nlist=65536 for 1M-10M vectors, nprobe=256-1024 for recall optimization; GPU-accelerated training on RunPod, but index now fits in local memory (M3 MacBook Air) for hybrid retrieval; separate notebooks enable faster iteration (embedding generation ~1-2 hours, index building ~5-10 minutes).
- **ID mapping**: Maintain FAISS row index ↔ chunk_id mapping in separate .npy file for efficient storage and fast loading (3M+ mappings esetén JSON túl lassú és memóriaigényes).

3) Hybrid Retrieval (baseline)
- **Query processing**: Lowercase and normalize whitespace to match BM25 preprocessing; embed with EmbeddingGemma via Sentence Transformers using `model.encode(query, prompt_name="query", normalize_embeddings=True)` which automatically adds prompt "task: search result | query: {query}" and L2-normalizes output.
- **Retrieval**: Obtain top-K candidates from both BM25 (chunk-level via bm25s) and FAISS (dense similarity); aggregate chunk scores to document level using max-score strategy.
- **Fusion**: Apply RRF (Reciprocal Rank Fusion) to combine BM25 and FAISS rankings in a robust, parameter-free manner.
- **Output**: Top-k document IDs with per-source score features retained for downstream RL training.

4) RL Reranking (GRPO‑style)
- **Slate preparation**: Use chunk-level retrieval to create slates with top-scoring chunks from BM25 and FAISS (not doc-level aggregation); each slate contains the most relevant chunks as determined by retrieval scores; include full chunk text (~500-800 chars, not preview), metadata (court, domain, year), chunk_id, doc_id, and retrieval scores; map doc-level relevance from qrels to chunks; format as structured text (not JSON) for clarity; the slate order represents the baseline fusion ranking.
- **Policy model**: Fine-tune `Qwen/Qwen3-4B-Instruct-2507` using QLoRA adapters (4-bit quantization, moderate LoRA rank) for document scoring via autoregressive generation; model receives full chunk context to understand relevance reasoning; use groupwise softmax over candidate scores to produce reranking distribution.
- **Training framework**: Use TRL `GRPOTrainer` with **Unsloth acceleration** and **vLLM inference** for GRPO algorithm configuration: `loss_type="dapo"` (eliminates length bias), `scale_rewards="batch"` (robust normalization), `mask_truncated_completions=True` (stability), `importance_sampling_level="sequence"` (GSPO-style), `kl_coef=0.0` (no KL penalty); Unsloth provides 2x faster training and 50%+ memory savings through optimized gradient checkpointing and 4-bit quantization; vLLM (`fast_inference=True`) accelerates generation during GRPO rollouts by 2-3x (critical performance requirement); configure for RTX 5090 GPU (24GB VRAM) with optimized batch settings (batch=4, generations=10), gradient accumulation (2), RSLoRA for stability (`use_rslora=True`), and automatic dtype optimization.
- **Reward function**: TRL-compatible signature `(completions, prompts, **kwargs)`; extract query_id from prompt via regex, lookup slate metadata in global dict; baseline order is the slate's initial fusion ranking (index 0,1,2...); compute nDCG@10 difference between baseline fusion order and policy order as core reward; add small entropy bonus for exploration; assign negative reward for parse failures (not zero); clip rewards to stable range for gradient stability.
- **Dataset strategy**: Split full 98-query dataset into train/eval sets (80/20) with shuffled deterministic seed for reproducibility; each query annotated with 20 documents in qrels for sufficient supervision signal.
- **Outputs**: Save LoRA adapter weights and tokenizer to `data/models/grpo_policy/`; emit reranked document IDs per query; log nDCG@10 improvements to `metrics.json`.

GRPO Implementation Plan (TRL-based, RTX 5090 optimized, chunk-based, Unsloth-accelerated)
- **Environment**: Cloud training on RTX 5090 GPU (24GB VRAM) using `unsloth`, `vllm`, `trl`, `transformers`, `peft`, `bitsandbytes`; local M3 MacBook Air for baseline retrieval and evaluation only.
- **Acceleration framework**: Use Unsloth for model loading (`FastLanguageModel.from_pretrained`) with integrated 4-bit quantization, LoRA adapters, and vLLM inference; Unsloth gradient checkpointing (`use_gradient_checkpointing="unsloth"`) provides 50%+ memory savings enabling larger batch sizes and more generations per prompt; vLLM accelerates completions during GRPO rollouts by 2-3x through efficient batched inference and KV-cache optimization; RSLoRA (`use_rslora=True`) ensures training stability; automatic dtype selection (`dtype=None`) optimizes for GPU; total training time reduced on RTX 5090.
- **Data preparation**: Export chunk-level candidate slates from hybrid retrieval as JSONL; use top-scoring chunks (not doc-level aggregation) to provide relevant context; load full chunk text from `chunks.jsonl` for each candidate; include chunk_id, doc_id, metadata, retrieval scores; map doc-level relevance from qrels; maintain ID consistency across pipeline.
- **Model configuration**: Load `unsloth/Qwen3-4B-Instruct-2507` with 4-bit quantization; attach QLoRA adapters targeting all attention and MLP projection layers (q,k,v,o,gate,up,down); use moderate LoRA rank for quality/memory balance.
- **Prompt strategy**: Structured text format (not JSON) displaying chunk-level candidates with full text (~500-800 chars); include doc_id, chunk_id, court/domain/year, BM25/FAISS scores; format for clarity while accepting higher token count (~5000-6000) for meaningful context; max_seq_length=8192 to prevent truncation of full chunk texts.
- **Training notebook**: `notebooks/grpo_train_runpod.ipynb` loads chunk-based slate JSONL, configures `GRPOTrainer` with best-practice parameters (GRPO algorithm via `loss_type="dapo"`, batch-level reward scaling, sequence-level importance sampling, disabled KL penalty), implements TRL-compatible reward function with regex-based query_id extraction and global slate lookup, performs shuffled train/eval split (80/20, seed=42), logs Hungarian progress messages, checkpoints to `/workspace/artifacts/grpo_policy/`.
- **Reward computation**: Extract query_id from prompt via regex (robust parsing), retrieve slate metadata from global dict, baseline order is slate's fusion ranking [0,1,2,...], compute nDCG@10 for both baseline and policy orderings, use difference as primary signal, add exploration bonus, assign negative penalty for parse failures, clip to stable range.
- **Hyperparameters**: Configure for RTX 5090 with Unsloth optimizations: batch size 4 (Unsloth memory savings), gradient accumulation 2 (effective batch 8), 10 generations per prompt, learning rate 1e-5, warmup schedule 50 steps, RSLoRA enabled (`use_rslora=True`), automatic dtype (`dtype=None`); higher token count (~5000-6000/prompt) with max_seq_length=8192 and larger batch (8 effective) fits within 24GB VRAM due to Unsloth gradient checkpointing; training completes in 15-25 minutes with vLLM (500 steps, 40 generations/step) maintaining quality with increased exploration.
- **Resource management**: Memory footprint includes quantized model, LoRA parameters, optimizer state, activation buffers, GRPO trajectory storage, longer prompts (~5-6k tokens with 8192 max_seq_length); fits within 24GB VRAM with appropriate batch sizing and Unsloth gradient checkpointing.
- **Deployment strategy**: Training and inference occur on cloud; download metrics and evaluation results to `data/models/grpo_policy/` for local analysis; adapter weights remain on cloud environment.
- **Pipeline integration**: GRPO layer operates independently; slate preparation uses chunk-level retrieval (via `get_last_chunk_scores()`) and full text loading (via `_load_chunk_texts()`); no modifications to preprocessing, BM25 indexing, or base retrieval modules.
- **Qrels format**: Ensure qrels contains doc_id values matching `chunks.jsonl` structure; doc-level relevance mapped to chunk-level slates; provide sufficient query coverage for train/eval split.

BM25S Implementation Notes
- Tokenization: **case-insensitive** - lowercase corpus text before tokenization, lowercase queries before search; rely on `bm25s.tokenize` with configurable stopword handling (default none for Hungarian); persist the returned token-id structures for reuse.
- Index persistence: always call `BM25.save()` into `data/index/bm25/`, store `doc_ids.json` plus `bm25_stats.json` (total_docs, avg_length, histogram) for downstream consumers.
- Corpus structure: BM25S stores corpus documents as `{'id': index, 'text': chunk_id}` dictionaries; retrieval results must extract chunk_id from the 'text' field, not the 'id' field (which is just a numeric index).
- Search results: the `search()` method returns chunk_id strings that are later aggregated to document-level doc_id identifiers via suffix stripping (e.g., "doc_0" → "doc").
- Scoring utilities: prefer `get_scores` / `get_scores_from_ids` to avoid redundant retrieval when exporting features for GRPO.
- Performance: allow enabling `activate_numba_scorer()` and thread count tuning via CLI flags for local CPU optimization on M3 hardware.

5) CLI Workflows
- **Build** (Local): Docling parsing → text normalization → chunking → BM25 index creation with token cache.
- **Embedding Generation** (Cloud): Notebook (`gemma_embedding_runpod.ipynb`) generates embedding vectors from chunks using EmbeddingGemma via Sentence Transformers library (>=5.1.0) with document prompts (`prompt_name="document"`) on RTX 5090 GPU; outputs consolidated `embeddings.npy` (float32, L2-normalized) and `embedding_chunk_ids.json` artifacts for download.
- **FAISS Index Building** (Local/Cloud): Notebook (`faiss_index_builder.ipynb`) loads pre-computed embeddings from artifacts → builds OPQ64_256,IVF65536,PQ64x4fsr index → saves `faiss_index.bin` and `chunk_id_map.npy`; runs locally on M3 MacBook Air or cloud GPU for faster training.
- **Hybrid Retrieval** (Local): Script (`scripts/hybrid_retrieval.py`) runs BM25+FAISS hybrid search on query list → RRF fusion → saves pipeline results; memory-optimized for M3 MacBook Air (16GB RAM) with PQ-compressed FAISS index.
  - Note: `chunk_id_map.npy` is loaded via NumPy with `allow_pickle=True`. Both array-based (row-indexed `np.ndarray`) and dict-based (`{row_index: chunk_id}`) formats are supported for compatibility.
- **Qrels Generation** (Local): Script (`scripts/qrels_generation.py`) creates qrels template from retrieval results → saves TSV for manual annotation; runs locally after hybrid retrieval.
- **RL Training** (Cloud): Notebook (`grpo_train_runpod.ipynb`) trains GRPO policy from slates → saves adapter weights and metrics on RTX 5090 GPU.
- **Evaluation** (Local): Notebook (`baseline_evaluation.ipynb`) computes nDCG@10 metrics from qrels and results.

Relevance Judgments (Qrels)
- **Location**: `data/qrels/baseline_qrels.tsv` with stable filename for pipeline consistency.
- **Format**: Tab-separated with header `query_id\tdoc_id\trelevance`; query_id as Hungarian query text, doc_id matching `chunks.jsonl` structure (document-level, not chunk-level), relevance grades in {0,1,2}.
- **Coverage**: Minimum 50 unique queries with multiple annotated documents each; sufficient for train/eval split and statistical significance.
- **Consistency**: UTF-8 encoding, Unix newlines, sorted by query_id and relevance for deterministic loading; maintain document ID alignment across all pipeline components.

Acceptance Criteria
- **Dataset scale**: Use full corpus and complete qrels dataset (98 queries) for final pipeline and evaluation.
- **Build outputs**: Chunks JSONL, BM25 index with token cache, document metadata accessible for slate preparation.
- **Cloud outputs**: FAISS OPQ64_256,IVF65536,PQ64x4fsr with efficient .npy ID mapping (memory-optimized), GRPO policy adapters with metrics.
- **Query functionality**: Return fused top-k document IDs (e.g., `0302-G_20416_2019_11`) from hybrid retrieval locally using memory-optimized FAISS index.
- **RL learning signal**: Demonstrate measurable improvement or consistent reward increase over training.
- **Model requirements**: Use real EmbeddingGemma model (from .env HF token) with no fallback; execute baseline retrieval locally, RL training on cloud.

Critical Requirements & Pitfalls
- **EmbeddingGemma library**: CRITICAL - MUST use Sentence Transformers library (>=5.1.0), NOT AutoModel from transformers; EmbeddingGemma requires specific prompt formatting ("title: none | text: " for documents, "task: search result | query: " for queries), mean pooling with attention mask, and proper L2 normalization, all handled automatically by Sentence Transformers; using AutoModel produces zero-vector embeddings that break retrieval and cause FAISS training failures; model does NOT support float16, use float32 or bfloat16.
- **Unsloth integration**: CRITICAL - Use Unsloth `FastLanguageModel` for GRPO training, NOT standard transformers `AutoModelForCausalLM`; Unsloth provides integrated 4-bit quantization, LoRA, and vLLM inference with significant performance improvements (2x faster, 50% less memory); must set `fast_inference=True` to enable vLLM for GRPO generations; must use `use_gradient_checkpointing="unsloth"` for optimal memory efficiency; standard transformers loading will work but miss critical optimizations for RTX 5090 capacity.
- **vLLM acceleration**: CRITICAL - Always set `fast_inference=True` in `FastLanguageModel.from_pretrained()`; vLLM provides 2-3x faster inference during GRPO rollouts, reducing training time from 45-60 min to 15-25 min; forgetting this flag causes severe performance degradation; add `dtype=None` for automatic optimization (bfloat16/float16 selection based on GPU).
- **RSLoRA stability**: RECOMMENDED - Enable `use_rslora=True` in `FastLanguageModel.get_peft_model()` for rank ≥32; RSLoRA (Rank-Stabilized LoRA) provides better training stability, smoother convergence, and prevents NaN/Inf issues with higher LoRA ranks; no performance penalty, only benefits.
- **Tokenizer optimization**: Set `tokenizer.padding_side = "right"` for Qwen models in generation tasks; right-padding is recommended for GRPO training to ensure consistent output formatting and better parse success rates in the reward function.
- **FAISS embeddings**: Always L2-normalize before indexing (required for inner product metric); Sentence Transformers handles this automatically with `normalize_embeddings=True`.
- **ID consistency**: Maintain correct mappings across FAISS row indices ↔ chunk_id ↔ doc_id ↔ qrels relevance at document level throughout pipeline.
- **Metric edge cases**: Return 0 for nDCG when no relevant documents exist in results.
- **Evaluation metrics**: Use `ranx` library (https://amenra.github.io/ranx/) for IR metrics computation; ranx provides TREC-validated implementations of NDCG, MAP, MRR, Precision@K, Recall@K with Numba-accelerated batch evaluation; replace manual metric implementations with `ranx.evaluate(qrels, run, metrics)` for cleaner code and 5-10x performance improvement; qrels format `{query_id: {doc_id: relevance}}` is directly compatible; GRPO evaluation uses ranx for comprehensive metrics (NDCG@5/10/20, MAP, MRR, Precision@5/10/20, Recall@5/10/20) on train/eval splits with per-query CSV exports.
- **RRF Fusion**: Use `ranx.fuse()` for Reciprocal Rank Fusion; `fuse([bm25_run, faiss_run], method="rrf")` replaces manual RRF implementation; ranx supports multiple fusion strategies (RRF, CombSUM, CombMNZ, etc.) enabling easy experimentation; fusion occurs at query level with Run objects `{query_id: {doc_id: score}}`.
- **GRPO NDCG calculation**: For GRPO reward computation during training, use sklearn `ndcg_score` (faster for training loop); for final evaluation, use `ranx.evaluate()` with comprehensive metrics on baseline vs policy rankings; convert ranking indices to Run format, compute full metric suite; export train_per_query_results.csv and eval_per_query_results.csv with all metrics per query; maintain edge-case handling (return 0.0 when no relevant docs).
- **Reward clipping**: Use `numpy.clip` with configured bounds for numerical stability.
- **Entropy metric**: Use `scipy.stats.entropy` on a normalized probability vector with small epsilon.
- **Baseline configuration**: RRF (Reciprocal Rank Fusion) provides robust, parameter-free baseline for GRPO comparison; no training required.
- **Build dependencies**: Embedding vectors generated via `gemma_embedding_runpod.ipynb` (cloud, faster training), FAISS index built via `faiss_index_builder.ipynb` (local/cloud); hybrid retrieval and qrels generation now run locally; GRPO training occurs on cloud environment only.
- **Hyperparameter flexibility**: Keep document aggregation strategies, top-K values, and RL training parameters configurable for experimentation.
- **JSONL Parsing**: Use `pandas.read_json(file, lines=True)` for JSONL files instead of manual line-by-line json.loads() parsing; pandas provides 10-30x faster C-optimized parsing with automatic type inference and memory-efficient chunked reading via `chunksize` parameter; especially critical for large files like chunks.jsonl (~2.9M rows); pandas already a project dependency via notebooks.

Configuration & Execution
- **Local execution**: M3 MacBook Air (16GB RAM) runs document preprocessing (Docling parsing, chunking), BM25 index building (CPU-only, no GPU needed), hybrid retrieval (`hybrid_retrieval.py`), qrels generation (`qrels_generation.py`), and evaluation (metrics computation from qrels); hybrid retrieval now possible locally with PQ-compressed FAISS index (~2-3GB memory usage fits within 16GB RAM).
- **Cloud execution**: RTX 5090 GPU (24GB VRAM) for embedding generation (`gemma_embedding_runpod.ipynb`) and GRPO training (`grpo_train_runpod.ipynb`) via RunPod notebooks; Unsloth-accelerated GRPO training with vLLM inference (`fast_inference=True`) for 2-3x faster training (15-25 min with vLLM vs 45-60 min without, originally 1 hour pre-Unsloth) and 50%+ memory efficiency; GPU acceleration for embedding generation and RL training; FAISS index building and hybrid retrieval now run locally.
- **Configuration**: Centralized in `configs/config.py` with values tuned for PoC scale; all hyperparameters configurable for experimentation.
- **Artifacts**: Simple JSON/JSONL format for reproducibility; cloud notebooks export results for local analysis; adapter weights remain on cloud.

Data Formats
- **chunks.jsonl**: One chunk per line with fields: `{chunk_id, doc_id, text, court, domain, year, source_path}`
- **BM25 artifacts**: bm25s model directory (scores/indices/indptr/vocab/params), chunk_id list, token statistics, optional token cache
- **Embedding artifacts**: `embeddings.npy` (float32, L2-normalized vectors generated via Sentence Transformers), `embedding_chunk_ids.json` (chunk_id order matching embedding rows)
- **FAISS artifacts**: `faiss_index.bin` (OPQ64_256,IVF65536,PQ64x4fsr binary), `chunk_id_map.npy` (row index → chunk_id mapping in efficient .npy format)
- **Training slates**: JSONL with chunk-level candidates per query: `{query_id, slate: [{chunk_id, doc_id, bm25_score, faiss_score, relevance, court, domain, year, text (full chunk)}]}`; slate order represents baseline fusion ranking; chunk_id mapping loaded from .npy file for efficiency
- **Results**: Document ID lists (baseline and reranked), metrics JSON with nDCG@10 comparisons

Language Policy
- This file (AGENTS.md) must be written in English only.
- All other project outputs and user‑facing text must be in Hungarian.
- All CLI messages, result lists, reports, and printed strings must be Hungarian.
- Query responses must return only Hungarian content where applicable; when returning identifiers, return IDs only as requested.

Sources to implement GRPO based RL
- https://huggingface.co/learn/llm-course/en/chapter12/4
- https://huggingface.co/docs/trl/main/en/grpo_trainer
- https://huggingface.co/learn/llm-course/en/chapter12/5?fw=pt
- https://huggingface.co/learn/llm-course/en/chapter12/6 (Unsloth + GRPO integration)
- https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-Instruct.ipynb (Unsloth Qwen3 fine-tuning)
- https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507 (Unsloth Qwen3 model support)
- https://github.com/unslothai/unsloth (Unsloth documentation)
- https://docs.unsloth.ai/ (Unsloth API reference)
