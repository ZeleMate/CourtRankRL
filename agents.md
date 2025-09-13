Agent Specification

Goal

Implement a compute-light retrieval pipeline for Hungarian court decisions. The pipeline must demonstrate improvements in ranking quality through RL-based reranking (GRPO-style). The system should remain minimal, reproducible, and locally executable, while relying on a real embedding model (local or API-based). Write everything in Hungarian.

Non-Goals
	•	No PDF parsing.
	•	No stemming.
	•	No query rewrite.
	•	No tokenization libraries like tiktoken.
	•	No logging, test frameworks, or deployment infrastructure.

Inputs
	•	Raw documents: RTF and DOCX files containing court decisions.
	•	Dev qrels file: JSONL where each line has a query and relevant chunk IDs.

    • The raw documents' path looks like this: '/Users/zelenyianszkimate/Documents/CourtRankRL/data/raw/Ceglédi Járásbíróság/gazdasági jog/1403-G_20821_2011_89/1403-G_20821_2011_89.RTF' or this '/Users/zelenyianszkimate/Documents/CourtRankRL/data/raw/Békéscsabai Járásbíróság/gazdasági jog/0403-G_20163_2020_27/0403-G_20163_2020_27.DOCX'

Outputs
	•	Chunks JSONL: normalized and segmented text with metadata.
	•	Indexes: FAISS dense index and BM25 sparse index.
	•	Policy file: trained RL reranking model parameters.
	•	Query results: ranked list of chunks (baseline and reranked).

Capabilities
	1.	Ingestion:
	•	Use Docling to parse RTF/DOCX files into plain text.
	•	Attach minimal metadata (e.g., court, year, title, citation).
	•	Normalize text minimally (whitespace, control characters).
	2.	Chunking:
	•	Split documents into overlapping text segments of configurable length.
	•	Preserve metadata across chunks.
	3.	Indexing:
	•	Build a BM25 sparse index from tokenized chunk texts (simple split).
	•	Build a FAISS dense index from chunk embeddings.
	•	Use a real embedding model (local or API) to generate embeddings.
	•	Store an ID map to align FAISS rows with chunk IDs.
	4.	Hybrid Retrieval:
	•	Embed query text with the same embedding model.
	•	Retrieve candidates from BM25 and FAISS.
	•	Combine results via Reciprocal Rank Fusion (RRF) or weighted fusion (z-score normalization).
	•	Return a top-k baseline ranked list.
	5.	RL Reranking:
	•	Extract features for each candidate: dense similarity score, BM25 normalized score, rank difference.
	•	Train a linear or shallow neural network policy using GRPO-style group-relative rewards.
	•	Reward function: ranking metrics such as nDCG@10.
	•	Save the trained policy.
	6.	CLI Commands:
	•	Build: parse → clean → chunk → embed → build indexes.
	•	Query: embed query → retrieve candidates → fuse → return ranked chunks.
	•	Train RL: load qrels → retrieve baseline → compute features → train policy → save policy.

Acceptance Criteria
	•	The build command produces chunks, FAISS index, BM25 index, and ID mapping.
	•	The query command returns a fused ranked list of chunks.
	•	The RL training command produces a policy file and reranking improves metrics compared to baseline.
	•	The system works with a real embedding model, not just a stub.

Pitfalls
	•	Always normalize embeddings before adding to FAISS for cosine similarity.
	•	Handle zero variance in z-score normalization.
	•	Ensure correct mapping between FAISS indices, chunk IDs, and relevance labels.
	•	Return 0 for nDCG if no relevant documents are found.
	•	Keep chunk size, overlap, and retrieval cutoffs configurable, not hard-coded.

Future Extensions
	•	Swap in different embedding models (e.g., Gemini, bge-m3, OpenAI).
	•	Add domain priors (court level, recency, citation matching).
	•	Introduce richer features for RL.
	•	Extend with logging, testing, or deployment pipelines after prototype stage.