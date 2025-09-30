#!/usr/bin/env python3
"""
CourtRankRL GRPO-style Reranker
Based on agents.md specification: Qwen3-4B-Instruct with QLoRA and TRL GRPOTrainer.

Főbb jellemzők:
- Qwen/Qwen3-4B-Instruct-2507 model with QLoRA adapters
- TRL GRPOTrainer for memory-efficient training
- NDCG@10 reward calculation with entropy bonus
- Hungarian status messages and logging
- Local inference with CPU/MPS support
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
try:
    from trl import GRPOTrainer, GRPOConfig  # type: ignore[attr-defined]
except ImportError:
    try:
        from trl.trainer.grpo_trainer import GRPOTrainer  # type: ignore[import-not-found]
        from trl.trainer.grpo_config import GRPOConfig  # type: ignore[import-not-found]
    except ImportError:
        GRPOTrainer = None  # type: ignore
        GRPOConfig = None  # type: ignore
import logging

# Projekt gyökér hozzáadása az import úthoz
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from configs import config
from src.search.hybrid_search import HybridRetriever


# GRPO tréninghez és inference-hez használt alapértelmezett értékek.
GRPO_MODEL_NAME = getattr(config, "GRPO_MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
GRPO_SLATE_SIZE = getattr(config, "GRPO_SLATE_SIZE", 20)
GRPO_GROUP_SIZE = getattr(config, "GRPO_GROUP_SIZE", GRPO_SLATE_SIZE)
GRPO_LEARNING_RATE = getattr(config, "GRPO_LEARNING_RATE", 1e-5)
GRPO_NUM_GENERATIONS = getattr(config, "GRPO_NUM_GENERATIONS", 4)
GRPO_WARMUP_STEPS = getattr(config, "GRPO_WARMUP_STEPS", 100)
GRPO_MAX_STEPS = getattr(config, "GRPO_MAX_STEPS", 1000)
GRPO_SAVE_STEPS = getattr(config, "GRPO_SAVE_STEPS", 100)
GRPO_EVAL_STEPS = getattr(config, "GRPO_EVAL_STEPS", 50)
GRPO_NDCG_K = getattr(config, "GRPO_NDCG_K", 10)
GRPO_ENTROPY_BONUS = getattr(config, "GRPO_ENTROPY_BONUS", 0.01)
GRPO_NEGATIVE_REWARD_CLAMP = getattr(config, "GRPO_NEGATIVE_REWARD_CLAMP", -1.0)
GRPO_VARIANCE_NORMALIZATION = getattr(config, "GRPO_VARIANCE_NORMALIZATION", True)
GRPO_LORA_RANK = getattr(config, "GRPO_LORA_RANK", 64)
GRPO_LORA_ALPHA = getattr(config, "GRPO_LORA_ALPHA", 128)
GRPO_LORA_DROPOUT = getattr(config, "GRPO_LORA_DROPOUT", 0.05)

def export_slates_for_grpo_training(retriever: HybridRetriever, qrels: Dict[str, Dict[str, int]], output_path: Path):
    """
    Export baseline candidate slates for GRPO training.
    Agents.md: serializes baseline candidate slates into JSONL for GRPO training.
    """
    print("Slate export indítása...")

    slates = []

    for query_id, relevance_data in qrels.items():
        try:
            # Get baseline candidates (same as training data preparation)
            retriever.retrieve_candidates(query_id, top_k=GRPO_SLATE_SIZE * 2)
            bm25_results = retriever.get_last_doc_scores("bm25")
            dense_results = retriever.get_last_doc_scores("dense")

            if not bm25_results:
                continue

            # Merge and create fixed-size slate
            all_candidates = {}
            for doc_id, score in bm25_results:
                all_candidates[doc_id] = {"doc_id": doc_id, "bm25_score": score}

            for doc_id, score in dense_results:
                if doc_id in all_candidates:
                    all_candidates[doc_id]["faiss_score"] = score
                else:
                    all_candidates[doc_id] = {"doc_id": doc_id, "faiss_score": score}

            # Sort by combined score and take top slate_size
            sorted_candidates = sorted(
                all_candidates.values(),
                key=lambda x: x.get("bm25_score", 0) + x.get("faiss_score", 0),
                reverse=True
            )[:GRPO_SLATE_SIZE]

            # Pad with neutral placeholders if needed
            while len(sorted_candidates) < GRPO_SLATE_SIZE:
                sorted_candidates.append({
                    "doc_id": f"placeholder_{len(sorted_candidates)}",
                    "bm25_score": 0.0,
                    "faiss_score": 0.0
                })

            # Add metadata and relevance labels
            for candidate in sorted_candidates:
                doc_id = candidate["doc_id"]
                candidate["relevance"] = relevance_data.get(doc_id, 0)
                candidate["chunk_id"] = doc_id  # Simplified mapping
                candidate["query_id"] = query_id

            slates.append({
                "query_id": query_id,
                "slate": sorted_candidates
            })

        except Exception as e:
            print(f"Hiba a {query_id} slate export során: {e}")
            continue

    # Save to JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for slate in slates:
            f.write(json.dumps(slate, ensure_ascii=False) + '\n')

    print(f"Slates exportálva: {len(slates)} slate -> {output_path}")
    return slates

class GRPOReranker:
    """
    GRPO-style reranker using Qwen3-4B-Instruct-2507 with QLoRA.
    Agents.md specification: Qwen/Qwen3-4B-Instruct-2507 with QLoRA adapters.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training state
        self.is_trained = False
        self.baseline_scores = []
        self.reranked_scores = []

        # Load policy if available
        self._load_policy_if_available()

    def _load_policy_if_available(self):
        """Load trained GRPO policy if available."""
        if config.GRPO_ADAPTER_PATH.exists():
            try:
                print("GRPO adapter betöltése...")
                self._initialize_model()
                if self.model is not None and hasattr(self.model, 'load_adapter'):
                    self.model.load_adapter(str(config.GRPO_ADAPTER_PATH.parent), adapter_name="grpo_policy")
                self.is_trained = True
                print("GRPO adapter sikeresen betöltve.")
            except Exception as e:
                print(f"Hiba az adapter betöltésekor: {e}")
                self.is_trained = False
        else:
            print("Nincs betanított GRPO adapter.")

    def _initialize_model(self):
        """Initialize Qwen3-4B-Instruct model with QLoRA."""
        if self.model is None:
            print("Qwen3-4B-Instruct modell inicializálása...")

            # 4-bit quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                GRPO_MODEL_NAME,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                GRPO_MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )

            # Configure LoRA
            lora_config = LoraConfig(
                r=GRPO_LORA_RANK,
                lora_alpha=GRPO_LORA_ALPHA,
                lora_dropout=GRPO_LORA_DROPOUT,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )

            self.model = get_peft_model(self.model, lora_config)
            print("Qwen3-4B-Instruct modell QLoRA adapterekkel inicializálva.")

    def _prepare_slate_context(self, query: str, candidates: List[Dict]) -> str:
        """
        Prepare structured JSON context for GRPO model.
        Agents.md: structured JSON snippets embedded in the prompt.
        """
        slate_items = []
        for i, candidate in enumerate(candidates):
            slate_items.append({
                "index": i,
                "chunk_id": candidate.get("chunk_id", ""),
                "doc_id": candidate.get("doc_id", ""),
                "text": candidate.get("text", "")[:500],  # Trim to config length
                "court": candidate.get("court", ""),
                "domain": candidate.get("domain", ""),
                "year": candidate.get("year", 0),
                "bm25_score": candidate.get("bm25_score", 0.0),
                "faiss_score": candidate.get("faiss_score", 0.0),
                "rrf_score": candidate.get("rrf_score", 0.0)
            })

        context = {
            "query": query,
            "slate": slate_items,
            "slate_size": len(slate_items)
        }

        return json.dumps(context, ensure_ascii=False, indent=2)

    def _calculate_ndcg(self, ranked_indices: List[int], true_relevance: List[float], k: int = 10) -> float:
        """Calculate NDCG@k as per agents.md specification."""
        if not true_relevance or not ranked_indices:
            return 0.0

        dcg = 0.0
        for i in range(min(k, len(ranked_indices))):
            if i < len(true_relevance):
                rel = true_relevance[ranked_indices[i]]
                dcg += rel / np.log2(i + 2)

        sorted_rel = sorted(true_relevance, reverse=True)
        idcg = 0.0
        for i in range(min(k, len(sorted_rel))):
            idcg += sorted_rel[i] / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def _reward_function(self, completions, **kwargs):
        """
        Custom reward function for GRPO trainer.
        Agents.md: nDCG@10 difference with entropy bonus and variance normalization.
        """
        rewards = []

        for completion, slate_data in zip(completions, kwargs.get("slate_data", [])):
            try:
                # Parse model output to get rankings
                # This is a simplified version - in practice you'd need proper parsing
                predicted_order = self._parse_model_ranking(completion)

                true_relevance = [item.get("relevance", 0) for item in slate_data]
                baseline_order = list(range(len(slate_data)))

                baseline_ndcg = self._calculate_ndcg(baseline_order, true_relevance, GRPO_NDCG_K)
                policy_ndcg = self._calculate_ndcg(predicted_order, true_relevance, GRPO_NDCG_K)

                # NDCG difference as reward
                reward = policy_ndcg - baseline_ndcg

                # Clamp negative rewards
                if reward < 0:
                    reward = max(reward, GRPO_NEGATIVE_REWARD_CLAMP)

                # Entropy bonus
                entropy = self._calculate_entropy(predicted_order)
                reward += GRPO_ENTROPY_BONUS * entropy

                # Variance normalization
                if GRPO_VARIANCE_NORMALIZATION and len(self.baseline_scores) > 0:
                    variance = np.var(self.baseline_scores)
                    if variance > 0:
                        reward /= np.sqrt(variance)

                rewards.append(reward)

            except Exception as e:
                print(f"Hiba a reward számításakor: {e}")
                rewards.append(0.0)

        return rewards

    def _parse_model_ranking(self, completion: str) -> List[int]:
        """Parse model completion to extract ranking."""
        # Simplified parsing - in practice you'd implement proper ranking extraction
        # This is a placeholder implementation
        try:
            # Extract indices from completion (example: "1,3,2,4,0" -> [1,3,2,4,0])
            numbers = [int(x.strip()) for x in completion.split(",") if x.strip().isdigit()]
            return numbers[:GRPO_SLATE_SIZE]
        except:
            return list(range(GRPO_SLATE_SIZE))

    def _calculate_entropy(self, ranking: List[int]) -> float:
        """Calculate entropy of ranking distribution."""
        if not ranking:
            return 0.0

        # Convert to probability distribution
        counts = {}
        for idx in ranking:
            counts[idx] = counts.get(idx, 0) + 1

        probs = [count / len(ranking) for count in counts.values()]
        entropy = -sum(p * np.log(p) for p in probs if p > 0)

        return entropy

    def prepare_training_data(self, retriever: HybridRetriever, qrels: Dict[str, Dict[str, int]]) -> List[Dict]:
        """
        Prepare training data for GRPO trainer.
        Agents.md: materialize per-query candidate slates and save them as JSONL.
        """
        training_data = []

        for query_id, relevance_data in qrels.items():
            try:
                # Get baseline candidates
                retriever.retrieve_candidates(query_id, top_k=GRPO_SLATE_SIZE * 2)
                bm25_results = retriever.get_last_doc_scores("bm25")
                dense_results = retriever.get_last_doc_scores("dense")

                if not bm25_results:
                    continue

                # Merge and create fixed-size slate
                all_candidates = {}
                for doc_id, score in bm25_results:
                    all_candidates[doc_id] = {"doc_id": doc_id, "bm25_score": score}

                for doc_id, score in dense_results:
                    if doc_id in all_candidates:
                        all_candidates[doc_id]["faiss_score"] = score
                    else:
                        all_candidates[doc_id] = {"doc_id": doc_id, "faiss_score": score}

                # Sort by combined score and take top slate_size
                sorted_candidates = sorted(
                    all_candidates.values(),
                    key=lambda x: x.get("bm25_score", 0) + x.get("faiss_score", 0),
                    reverse=True
                )[:GRPO_SLATE_SIZE]

                # Pad with neutral placeholders if needed
                while len(sorted_candidates) < GRPO_SLATE_SIZE:
                    sorted_candidates.append({
                        "doc_id": f"placeholder_{len(sorted_candidates)}",
                        "bm25_score": 0.0,
                        "faiss_score": 0.0
                    })

                # Add metadata and relevance labels
                for candidate in sorted_candidates:
                    doc_id = candidate["doc_id"]
                    candidate["relevance"] = relevance_data.get(doc_id, 0)
                    candidate["chunk_id"] = doc_id  # Simplified mapping

                training_data.append({
                    "query": query_id,
                    "slate": sorted_candidates
                })

            except Exception as e:
                print(f"Hiba a {query_id} training adat előkészítésekor: {e}")
                continue

        return training_data

    def train_grpo(self, training_data: List[Dict]):
        """
        Train GRPO model using TRL GRPOTrainer.
        Agents.md: run GRPOTrainer with group size matched to slate length.
        """
        print("GRPO training indítása...")
        self._initialize_model()

        # Prepare dataset for GRPO trainer
        def data_generator():
            for item in training_data:
                yield {
                    "prompt": self._create_training_prompt(item["query"], item["slate"]),
                    "slate_data": item["slate"]
                }

        # GRPO configuration
        if GRPOConfig is None:
            raise ImportError("GRPOConfig not available, install trl package")
        
        grpo_config = GRPOConfig(
            output_dir=str(config.GRPO_POLICY_DIR),
            num_generations=GRPO_NUM_GENERATIONS,
            max_steps=GRPO_MAX_STEPS,
            save_steps=GRPO_SAVE_STEPS,
            eval_steps=GRPO_EVAL_STEPS,
            logging_steps=10,
            learning_rate=GRPO_LEARNING_RATE,
            warmup_steps=GRPO_WARMUP_STEPS,
            max_completion_length=256,
        )

        # Initialize trainer
        if GRPOTrainer is None or self.model is None:
            raise ImportError("GRPOTrainer not available or model not initialized")
        
        self.trainer = GRPOTrainer(
            model=self.model,  # type: ignore
            reward_funcs=self._reward_function,  # type: ignore
            args=grpo_config,
            train_dataset=list(data_generator()),  # type: ignore
        )

        # Train
        print("GRPO training fut...")
        self.trainer.train()

        # Save artifacts
        self._save_training_artifacts(training_data)

        self.is_trained = True
        print("GRPO training befejezve.")

    def _create_training_prompt(self, query: str, slate: List[Dict]) -> List[Dict]:
        """Create training prompt for GRPO."""
        context = self._prepare_slate_context(query, slate)

        messages = [
            {"role": "system", "content": "Rangsorold a következő bírósági dokumentumokat relevancia szerint. Válaszolj számokkal vesszővel elválasztva (pl. '1,3,2,4,0')."},
            {"role": "user", "content": f"Kérjük, rangsorolja ezeket a bírósági dokumentumokat:\n\n{context}"}
        ]

        return messages

    def _save_training_artifacts(self, training_data: List[Dict]):
        """Save training artifacts as per agents.md specification."""
        print("Training artifactumok mentése...")

        # Save LoRA adapter
        if self.model is not None:
            self.model.save_pretrained(str(config.GRPO_POLICY_DIR))

        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(config.GRPO_POLICY_DIR))

        # Save metrics
        metrics = {
            "training_samples": len(training_data),
            "slate_size": GRPO_SLATE_SIZE,
            "model_name": GRPO_MODEL_NAME,
            "baseline_ndcg": np.mean(self.baseline_scores) if self.baseline_scores else 0.0,
            "reranked_ndcg": np.mean(self.reranked_scores) if self.reranked_scores else 0.0,
        }

        with open(config.GRPO_METRICS_PATH, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        print(f"Artifactumok mentve: {config.GRPO_POLICY_DIR}")

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
    ) -> List[Dict]:
        """
        Rerank candidates using trained GRPO policy.
        Agents.md: emit reranked identifier lists per query.
        """
        if not self.is_trained or self.model is None:
            print("Nincs betanított GRPO modell, baseline sorrend használata.")
            return candidates

        try:
            # Prepare context for inference
            context = self._prepare_slate_context(query, candidates)

            # Create prompt
            messages = [
                {"role": "system", "content": "Rangsorold a bírósági dokumentumokat relevancia szerint. Válaszolj számokkal (pl. '1,3,2,4,0')."},
                {"role": "user", "content": f"Rangsorolja ezeket a dokumentumokat:\n\n{context}"}
            ]

            # Tokenize
            if self.tokenizer is None or self.model is None:
                return candidates
            
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.device)

            # Generate
            with torch.no_grad():
                eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=eos_token_id
                )

            # Decode and parse ranking
            generated_text = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            predicted_order = self._parse_model_ranking(generated_text)

            # Reorder candidates
            reranked = []
            for idx in predicted_order:
                if idx < len(candidates):
                    reranked.append(candidates[idx])

            # Add remaining candidates
            used_indices = set(predicted_order)
            for i, candidate in enumerate(candidates):
                if i not in used_indices:
                    reranked.append(candidate)

            return reranked

        except Exception as e:
            print(f"Hiba a reranking során: {e}")
            return candidates

    def evaluate(self, retriever: HybridRetriever, test_queries: List[str], qrels: Dict[str, Dict[str, int]]):
        """
        Evaluate reranker performance vs baseline.
        Agents.md spec: numeric comparison baseline vs rerank.
        """
        baseline_scores = []
        reranked_scores = []
        num_queries = 0

        print("=== GRPO RERANKER KIÉRTÉKELÉS ===")

        for query in test_queries:
            if query not in qrels:
                continue

            try:
                # Get baseline candidates
                retriever.retrieve_candidates(query, top_k=config.TOP_K_BASELINE)
                bm25_results = retriever.get_last_doc_scores("bm25")
                dense_results = retriever.get_last_doc_scores("dense")

                if not bm25_results:
                    continue

                # Convert to candidate format
                baseline_candidates = []
                for doc_id, score in bm25_results:
                    baseline_candidates.append({
                        "doc_id": doc_id,
                        "bm25_score": score,
                        "relevance": qrels[query].get(doc_id, 0)
                    })

                # Calculate baseline NDCG
                baseline_order = list(range(len(baseline_candidates)))
                baseline_relevance = [c["relevance"] for c in baseline_candidates]
                baseline_ndcg = self._calculate_ndcg(baseline_order, baseline_relevance, GRPO_NDCG_K)
                baseline_scores.append(baseline_ndcg)

                # Rerank with GRPO policy
                reranked_candidates = self.rerank(query, baseline_candidates)

                # Calculate reranked NDCG
                reranked_relevance = [c["relevance"] for c in reranked_candidates]
                reranked_ndcg = self._calculate_ndcg(list(range(len(reranked_candidates))), reranked_relevance, GRPO_NDCG_K)
                reranked_scores.append(reranked_ndcg)

                num_queries += 1

                if num_queries <= 5:  # Show first few examples
                    print(f"Query: {query[:50]}...")
                    print(f"  Baseline nDCG@10: {baseline_ndcg:.4f}")
                    print(f"  Reranked nDCG@10: {reranked_ndcg:.4f}")
                    print()

            except Exception as e:
                print(f"Hiba a {query} kiértékelésekor: {e}")
                continue

        if num_queries > 0:
            avg_baseline_ndcg = np.mean(baseline_scores)
            avg_reranked_ndcg = np.mean(reranked_scores)
            improvement = avg_reranked_ndcg - avg_baseline_ndcg

            print("=== KIÉRTÉKELÉS ÖSSZESÍTÉSE ===")
            print(f"Átlagos baseline nDCG@10: {avg_baseline_ndcg:.4f}")
            print(f"Átlagos reranked nDCG@10: {avg_reranked_ndcg:.4f}")
            print(f"Javulás: {improvement:.4f}")
            print(f"Feldolgozott query-k: {num_queries}")

            # Store scores for metrics
            self.baseline_scores = baseline_scores
            self.reranked_scores = reranked_scores

            return {
                'baseline_ndcg': avg_baseline_ndcg,
                'reranked_ndcg': avg_reranked_ndcg,
                'improvement': improvement,
                'num_queries': num_queries
            }

        return None

def load_qrels(qrels_file: Path) -> Dict[str, Dict[str, int]]:
    """Load qrels file for training - convert chunk_id to doc_id."""
    qrels = {}

    if not qrels_file.exists():
        return qrels

    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')  # TSV format as per agents.md
            if len(parts) >= 3:
                query_id = parts[0]
                chunk_id = parts[1]
                relevance = int(parts[2])

                # Convert chunk_id to doc_id (assuming format: "doc_id_src_tag_i")
                doc_id = chunk_id.split('_')[0] if '_' in chunk_id else chunk_id

                if query_id not in qrels:
                    qrels[query_id] = {}

                qrels[query_id][doc_id] = relevance

    return qrels

def train_grpo_reranker():
    """Train GRPO reranker using agents.md specification."""
    print("=== GRPO RERANKER TANÍTÁS ===")

    if not config.QRELS_FILE.exists():
        print(f"Hiba: Qrels fájl nem található: {config.QRELS_FILE}")
        return

    # Load qrels
    qrels = load_qrels(config.QRELS_FILE)

    if not qrels:
        print("Nincs qrels adat")
        return

    print(f"Qrels betöltve: {len(qrels)} query")

    # Initialize components
    reranker = GRPOReranker()
    retriever = HybridRetriever()

    # Prepare training data
    print("Training adat előkészítése...")
    training_data = reranker.prepare_training_data(retriever, qrels)

    if not training_data:
        print("Nincs training adat")
        return

    print(f"Training adat előkészítve: {len(training_data)} minta")

    # Train GRPO model
    reranker.train_grpo(training_data)

    # Final evaluation
    print("\n=== VÉGÉRTÉKELÉS ===")
    try:
        eval_results = reranker.evaluate(retriever, list(qrels.keys())[:20], qrels)
        if eval_results:
            print(f"Baseline nDCG@10: {eval_results['baseline_ndcg']:.4f}")
            print(f"Reranked nDCG@10: {eval_results['reranked_ndcg']:.4f}")
            print(f"Javulás: {eval_results['improvement']:.4f}")
    except Exception as e:
        print(f"Evaluation hiba: {e}")

    print("=== GRPO TANÍTÁS BEFEJEZVE ===")

if __name__ == '__main__':
    train_grpo_reranker()
