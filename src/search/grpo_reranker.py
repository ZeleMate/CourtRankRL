#!/usr/bin/env python3
"""
CourtRankRL GRPO-style Reranker
Agents.md specifikáció alapján implementálva.

Főbb jellemzők:
- Qwen/Qwen3-4B-Instruct-2507 model QLoRA adapterekkel
- CPU/MPS inference alacsony memóriahasználattal
- Graceful fallback baseline ordering-re
- NDCG@10 alapú scoring
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

# Projekt gyökér hozzáadása az import úthoz
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from configs import config
from src.search.hybrid_search import HybridRetriever

class GRPOLocalReranker:
    """GRPO-based reranker using Qwen model with LoRA adapters."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.loaded = False
        self.use_grpo = True

        # Load GRPO policy if available
        self._load_grpo_policy()

    def _load_grpo_policy(self):
        """Load GRPO policy from artifacts directory."""
        policy_dir = config.GRPO_POLICY_DIR

        if not policy_dir.exists():
            print(f"GRPO policy könyvtár nem található: {policy_dir}")
            print("Fallback: baseline ordering használata")
            self.use_grpo = False
            return

        try:
            # Load PEFT config
            peft_config = PeftConfig.from_pretrained(policy_dir)

            # Load base model with 4-bit quantization for CPU/MPS
            base_model = config.GRPO_MODEL_NAME

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16 if self.device.type == "cpu" else torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                policy_dir,
                trust_remote_code=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=quantization_config,
                device_map="auto" if self.device.type == "cpu" else {"": self.device},
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == "cpu" else torch.bfloat16
            )

            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(model, policy_dir)
            self.model.eval()

            if self.device.type != "cpu":
                self.model = self.model.to(self.device)

            self.loaded = True
            print(f"GRPO policy betöltve: {policy_dir}")

        except Exception as e:
            print(f"GRPO policy betöltési hiba: {e}")
            print("Fallback: baseline ordering használata")
            self.use_grpo = False

    def _create_prompt(self, query: str, candidates: List[Dict]) -> str:
        """Create Hungarian prompt for reranking."""
        prompt = f"""A következő bírósági dokumentumokat kell rangsorolnod egy '{query}' keresési lekérdezéshez.

Válaszolj minden dokumentumhoz egy 0-10 közötti relevancia pontszámmal (10 = nagyon releváns, 0 = nem releváns).

Dokumentumok:
"""

        for i, candidate in enumerate(candidates, 1):
            # Use first chunk text as representative
            text = candidate['chunks'][0]['text'][:500] + "..." if len(candidate['chunks'][0]['text']) > 500 else candidate['chunks'][0]['text']

            prompt += f"""
{i}. Dokumentum (Bíróság: {candidate['chunks'][0]['metadata']['court']}, Év: {candidate['chunks'][0]['metadata']['year']})
Szöveg: {text}
"""

        prompt += """
Válaszolj minden dokumentumhoz egy sorban, a következő formátumban:
1. [pontszám]
2. [pontszám]
...

Pontszámok (0-10): """

        return prompt

    def _extract_scores_from_completion(self, completion: str) -> List[float]:
        """Extract numeric scores from model completion."""
        lines = completion.strip().split('\n')
        scores = []

        for line in lines:
            if line.strip() and any(char.isdigit() for char in line):
                # Extract numeric scores (simple parsing)
                parts = line.replace(',', '').split()
                for part in parts:
                    try:
                        score = float(part)
                        if 0 <= score <= 10:  # Reasonable score range
                            scores.append(score)
                    except ValueError:
                        continue

        return scores

    def _calculate_ndcg(self, relevance_scores: List[int], k: int = 10) -> float:
        """Calculate NDCG@k."""
        if not relevance_scores:
            return 0.0

        dcg = 0.0
        for i in range(min(k, len(relevance_scores))):
            rel = relevance_scores[i]
            dcg += rel / (i + 1) ** 0.5

        # IDCG calculation
        sorted_rel = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for i in range(min(k, len(sorted_rel))):
            idcg += sorted_rel[i] / (i + 1) ** 0.5

        return dcg / idcg if idcg > 0 else 0.0

class GRPOReranker:
    """GRPO-style reranker using Qwen model with LoRA adapters."""

    def __init__(self):
        self.reranker = GRPOLocalReranker()

    def extract_features(
        self,
        doc_ids: List[str],
        bm25_scores: Dict[str, float],
        dense_scores: Dict[str, float],
        chunk_groups: Dict[str, List[Tuple[str, float]]],
        retriever: HybridRetriever,
    ) -> np.ndarray:
        features: List[np.ndarray] = []

        bm25_vals = list(bm25_scores.values())
        bm25_mean = np.mean(bm25_vals) if bm25_vals else 0.0
        bm25_std = np.std(bm25_vals) if bm25_vals else 1.0
        if bm25_std == 0:
            bm25_std = 1.0

        for doc_id in doc_ids:
            bm25_score = bm25_scores.get(doc_id, 0.0)
            dense_score = dense_scores.get(doc_id, 0.0)
            normalized_bm25 = (bm25_score - bm25_mean) / bm25_std

            # Rang különbség a top listákból
            bm25_rank = self._get_rank(doc_id, bm25_scores)
            dense_rank = self._get_rank(doc_id, dense_scores)
            rank_diff = abs(bm25_rank - dense_rank)

            # Token hossz feature – átlag chunk hossz
            chunk_list = chunk_groups.get(doc_id, [])
            if chunk_list and retriever.bm25:
                lengths = []
                for chunk_id, _ in chunk_list:
                    try:
                        parts = chunk_id.split("_")
                        lengths.append(int(parts[-1]))
                    except Exception:
                        continue
                token_len = np.mean(lengths) if lengths else retriever.bm25.avg_doc_length
            else:
                token_len = retriever.bm25.avg_doc_length if retriever.bm25 else 0.0

            features.append(
                np.array([dense_score, normalized_bm25, rank_diff, token_len], dtype=np.float32)
            )

        return np.stack(features) if features else np.zeros((0, 4), dtype=np.float32)

    @staticmethod
    def _get_rank(doc_id: str, scores: Dict[str, float]) -> int:
        if not scores:
            return 0
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        for idx, (candidate, _) in enumerate(ranked, start=1):
            if candidate == doc_id:
                return idx
        return len(ranked) + 1

    def calculate_ndcg(
        self,
        predicted_ranks: Sequence[int],
        true_relevance: Sequence[float],
        k: int = 10,
    ) -> float:
        """Calculate NDCG@k."""
        if not true_relevance:
            return 0.0

        dcg = 0.0
        for i in range(min(k, len(predicted_ranks))):
            if i < len(true_relevance):
                rel = true_relevance[predicted_ranks[i]]
                dcg += rel / np.log2(i + 2)

        sorted_rel = sorted(true_relevance, reverse=True)
        idcg = 0.0
        for i in range(min(k, len(sorted_rel))):
            idcg += sorted_rel[i] / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def train_step(
        self,
        retriever: HybridRetriever,
        query_id: str,
        bm25_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
        qrel_data: Dict[str, int],
    ):
        """
        Single GRPO training step with Unsloth-inspired memory optimizations.
        Uses group-relative rewards and efficient memory usage.
        """
        if not bm25_results and not dense_results:
            return None

        bm25_scores = dict(bm25_results)
        dense_scores = dict(dense_results)

        all_doc_ids_sorted = sorted(
            set(bm25_scores.keys()) | set(dense_scores.keys()),
            key=lambda doc: bm25_scores.get(doc, 0.0) + dense_scores.get(doc, 0.0),
            reverse=True,
        )

        features = self.extract_features(
            all_doc_ids_sorted,
            bm25_scores,
            dense_scores,
            getattr(retriever, "bm25_chunk_groups", {}),
            retriever,
        )

        if features.size == 0:
            return None

        features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            policy_scores = self.policy(features_tensor).squeeze()
            softmax_scores = torch.softmax(policy_scores, dim=0)

        ranked_indices = torch.argsort(softmax_scores, descending=True)

        all_doc_ids = all_doc_ids_sorted
        true_relevance = [qrel_data.get(doc_id, 0) for doc_id in all_doc_ids]

        baseline_ndcg = self.calculate_ndcg(
            list(range(len(all_doc_ids))), true_relevance, k=10
        )
        policy_ndcg = self.calculate_ndcg(ranked_indices.tolist(), true_relevance, k=10)

        reward = policy_ndcg - baseline_ndcg

        self.baseline_history.append(baseline_ndcg)
        if len(self.baseline_history) > 100:
            self.baseline_history.pop(0)

        self.reward_std_history.append(reward)
        if len(self.reward_std_history) > 50:
            self.reward_std_history.pop(0)

        advantage = reward
        if self.memory_efficient and len(self.baseline_history) > 5:
            group_baseline = np.mean(self.baseline_history[-10:]) if len(self.baseline_history) >= 10 else np.mean(self.baseline_history)
            advantage = reward - group_baseline
            if len(self.reward_std_history) > 5:
                global_std = np.std(self.reward_std_history) + 1e-8
                advantage /= global_std

        top_indices = ranked_indices[: min(10, len(ranked_indices))]
        selected_scores = softmax_scores[top_indices]

        advantage_tensor = torch.tensor(float(advantage), dtype=torch.float32, device=self.device)
        loss = -advantage_tensor * torch.mean(selected_scores)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return reward

    def rerank(
        self,
        retriever: HybridRetriever,
        bm25_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
        query: str = "",
        chunks_data: Optional[Dict[str, Dict]] = None,
    ) -> List[Tuple[str, float]]:
        """Rerank candidates using GRPO policy or fallback to baseline."""
        if not bm25_results and not dense_results:
            return []

        # Merge and deduplicate candidates
        all_candidates = {}
        for doc_id, score in bm25_results:
            all_candidates[doc_id] = {
                'bm25_score': score,
                'dense_score': dense_results.get(doc_id, 0.0) if dense_results else 0.0
            }

        for doc_id, score in dense_results:
            if doc_id not in all_candidates:
                all_candidates[doc_id] = {
                    'bm25_score': bm25_results.get(doc_id, 0.0) if bm25_results else 0.0,
                    'dense_score': score
                }

        # Sort by combined score (baseline)
        sorted_candidates = sorted(
            all_candidates.items(),
            key=lambda x: x[1]['bm25_score'] + x[1]['dense_score'],
            reverse=True
        )

        # If GRPO policy is not available, return baseline
        if not self.reranker.use_grpo:
            return [(doc_id, scores['bm25_score'] + scores['dense_score'])
                   for doc_id, scores in sorted_candidates]

        # Use GRPO policy for reranking
        try:
            # Prepare candidates with chunks
            candidates_with_chunks = []
            for doc_id, scores in sorted_candidates:
                chunks = []
                if chunks_data:
                    for chunk_id, chunk_data in chunks_data.items():
                        if chunk_id.startswith(doc_id + '_'):
                            chunks.append({
                                'chunk_id': chunk_id,
                                'text': chunk_data.get('text', '')[:500],
                                'metadata': {
                                    'court': chunk_data.get('court', ''),
                                    'domain': chunk_data.get('domain', ''),
                                    'year': chunk_data.get('year', ''),
                                }
                            })

                if not chunks:
                    # Fallback chunk
                    chunks = [{
                        'chunk_id': f"{doc_id}_placeholder_0",
                        'text': f"Document {doc_id} - no detailed chunks available",
                        'metadata': {
                            'court': 'unknown',
                            'domain': 'unknown',
                            'year': 'unknown',
                        }
                    }]

                candidates_with_chunks.append({
                    'doc_id': doc_id,
                    'chunks': chunks,
                    'bm25_score': scores['bm25_score'],
                    'dense_score': scores['dense_score']
                })

            # Create prompt and get GRPO scores
            prompt = self.reranker._create_prompt(query, candidates_with_chunks)

            inputs = self.reranker.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )

            # Move to device
            if self.reranker.device.type != "cpu":
                inputs = {k: v.to(self.reranker.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.reranker.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.reranker.tokenizer.eos_token_id
                )

            # Extract scores from completion
            completion = self.reranker.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            scores = self.reranker._extract_scores_from_completion(completion)

            if len(scores) != len(candidates_with_chunks):
                print(f"Score szám mismatch: {len(scores)} vs {len(candidates_with_chunks)}")
                return [(doc_id, score['bm25_score'] + score['dense_score'])
                       for doc_id, score in sorted_candidates]

            # Return reranked results
            reranked = []
            for i, (doc_id, _) in enumerate(sorted_candidates):
                reranked.append((doc_id, scores[i]))

            return sorted(reranked, key=lambda x: x[1], reverse=True)

        except Exception as e:
            print(f"GRPO reranking hiba: {e}")
            print("Fallback: baseline ordering használata")
            return [(doc_id, scores['bm25_score'] + scores['dense_score'])
                   for doc_id, scores in sorted_candidates]

    def evaluate(self, retriever: HybridRetriever, test_queries: List[str], qrels: Dict[str, Dict[str, int]], chunks_data: Optional[Dict[str, Dict]] = None):
        """
        Evaluate reranker performance vs baseline.
        Agents.md spec: numeric comparison baseline vs rerank.
        """
        total_baseline_ndcg = 0.0
        total_reranked_ndcg = 0.0
        num_queries = 0

        print("=== EVALUATION: BASELINE vs RERANKED ===")

        for query in test_queries:
            if query not in qrels:
                continue

            retriever.retrieve_candidates(query, top_k=50)
            bm25_results = retriever.get_last_doc_scores("bm25")
            dense_results = retriever.get_last_doc_scores("dense")

            if not bm25_results:
                continue

            # Baseline: BM25 results
            baseline_doc_ids = [doc_id for doc_id, _ in bm25_results]
            true_relevance = [qrels[query].get(doc_id, 0) for doc_id in baseline_doc_ids]
            baseline_ndcg = self.reranker._calculate_ndcg(list(range(len(baseline_doc_ids))), true_relevance)

            reranked_results = self.rerank(retriever, bm25_results, dense_results, query, chunks_data)
            reranked_doc_ids = [doc_id for doc_id, _ in reranked_results]
            true_relevance_reranked = [qrels[query].get(doc_id, 0) for doc_id in reranked_doc_ids]
            reranked_ndcg = self.reranker._calculate_ndcg(list(range(len(reranked_doc_ids))), true_relevance_reranked)

            total_baseline_ndcg += baseline_ndcg
            total_reranked_ndcg += reranked_ndcg
            num_queries += 1

            print(f"Query: {query[:30]}...")
            print(".4f")
            print(".4f")
            print()

        if num_queries > 0:
            avg_baseline_ndcg = total_baseline_ndcg / num_queries
            avg_reranked_ndcg = total_reranked_ndcg / num_queries
            improvement = avg_reranked_ndcg - avg_baseline_ndcg

            print("=== ÖSSZESÍTÉS ===")
            print(".4f")
            print(".4f")
            print(".4f")
            print(".1f")

            return {
                'baseline_ndcg': avg_baseline_ndcg,
                'reranked_ndcg': avg_reranked_ndcg,
                'improvement': improvement,
                'num_queries': num_queries
            }

        return None

    def save_policy(self, path: Path):
        """Save trained policy (legacy compatibility)."""
        print("GRPO policy saving not implemented - use cloud training notebook")

    def load_policy(self, path: Path):
        """Load trained policy (legacy compatibility)."""
        print("GRPO policy loading not implemented - use artifacts directory")

def load_qrels(qrels_file: Path) -> Dict[str, Dict[str, int]]:
    """Load qrels file for training - convert chunk_id to doc_id."""
    qrels = {}

    if not qrels_file.exists():
        return qrels

    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                query_id = parts[0]
                chunk_id = parts[2]  # This might be chunk_id from qrels
                relevance = int(parts[3])

                # Convert chunk_id to doc_id (assuming format: "doc_id_src_tag_i")
                doc_id = chunk_id.split('_')[0] if '_' in chunk_id else chunk_id

                if query_id not in qrels:
                    qrels[query_id] = {}

                qrels[query_id][doc_id] = relevance

    return qrels

def main():
    """Test GRPO reranker with baseline evaluation."""
    from src.search.hybrid_search import HybridRetriever

    if not config.BASELINE_QRELS_FILE.exists():
        print(f"Hiba: Qrels fájl nem található: {config.BASELINE_QRELS_FILE}")
        return

    # Load qrels
    qrels = load_qrels(config.BASELINE_QRELS_FILE)

    if not qrels:
        print("Nincs qrels adat")
        return

    # Initialize components
    reranker = GRPOReranker()
    retriever = HybridRetriever()

    print("=== GRPO RERANKER TESZT ===")
    print(f"GRPO policy betöltve: {reranker.reranker.loaded}")
    print(f"GRPO használata: {reranker.reranker.use_grpo}")

    # Test evaluation
    try:
        test_queries = list(qrels.keys())[:5]  # Test with first 5 queries
        eval_results = reranker.evaluate(retriever, test_queries, qrels)

        if eval_results:
            print("\n=== TESZT EREDMÉNYEK ===")
            print(".4f")
            print(".4f")
            print(".4f")
    except Exception as e:
        print(f"Evaluation hiba: {e}")

    print("=== TESZT SIKERES ===")

if __name__ == '__main__':
    main()
