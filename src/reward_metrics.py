"""
Multi-Metric Reward Helper Functions

Ez a modul a GRPO training során használt metrika számításokat tartalmazza.
Célja: modularizált, tesztelhető, és gyors metrika számítás runtime közben.
"""

from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import ndcg_score


def compute_mrr(relevances: List[int], k: int = None) -> float:
    """
    Mean Reciprocal Rank számítása.
    
    MRR@k = 1 / rank_first_relevant (ha van releváns a top-k-ban)
           = 0 (ha nincs releváns a top-k-ban)
    
    Args:
        relevances: Relevancia értékek ranking sorrendben [rel_1, rel_2, ...]
        k: Top-k truncation (None = teljes lista)
    
    Returns:
        MRR érték [0, 1]
    """
    if k is not None:
        relevances = relevances[:k]
    
    for i, rel in enumerate(relevances, start=1):
        if rel > 0:
            return 1.0 / i
    
    return 0.0


def compute_recall(relevances: List[int], k: int, total_relevant: int = None) -> float:
    """
    Recall@k számítása.
    
    Recall@k = |{relevant docs in top-k}| / |{all relevant docs}|
    
    Args:
        relevances: Relevancia értékek ranking sorrendben [rel_1, rel_2, ...]
        k: Top-k truncation
        total_relevant: Összes releváns dokumentum száma (ha None, akkor a lista alapján)
    
    Returns:
        Recall érték [0, 1]
    """
    if not relevances:
        return 0.0
    
    # Relevantes doc-ok száma top-k-ban
    relevant_in_topk = sum(1 for rel in relevances[:k] if rel > 0)
    
    # Összes releváns (ha nincs megadva, akkor az egész listából számoljuk)
    if total_relevant is None:
        total_relevant = sum(1 for rel in relevances if rel > 0)
    
    if total_relevant == 0:
        return 0.0
    
    return relevant_in_topk / total_relevant


def compute_ndcg(relevances: List[int], k: int) -> float:
    """
    nDCG@k számítása sklearn használatával.
    
    Args:
        relevances: Relevancia értékek ranking sorrendben [rel_1, rel_2, ...]
        k: Top-k truncation
    
    Returns:
        nDCG@k érték [0, 1]
    """
    if not relevances or k <= 0:
        return 0.0
    
    # sklearn ndcg_score: [true_relevances], [predicted_scores]
    # predicted_scores: magasabb score = jobb ranking (fordított sorrend)
    relevances_truncated = relevances[:k]
    scores = list(range(len(relevances_truncated), 0, -1))  # [n, n-1, ..., 1]
    
    try:
        ndcg = ndcg_score(
            y_true=[relevances_truncated],
            y_score=[scores],
            k=k
        )
        return float(ndcg)
    except Exception:
        # Edge case: nincs releváns dokumentum
        return 0.0


def compute_multi_metric_reward(
    predicted_indices: List[int],
    slate: List[Dict],
    baseline_metrics: Dict[str, float],
    weights: Dict[str, float] = None,
    clip_range: Tuple[float, float] = (-1.0, 1.0),
    use_sigmoid: bool = True,
    mrr_tiebreak_bonus: float = 0.02,
    diversity_penalty_weight: float = 0.0
) -> Tuple[float, Dict[str, float]]:
    """
    Multi-metrika reward számítása sigmoid-alapú stabilizációval.
    
    Javítások:
    - Sigmoid transzformáció a delta-kra → stabilabb gradiens
    - MRR tie-break bonus: ha nDCG/Recall stagnál de MRR javul
    - Diverzitás büntetés: ugyanazon court/domain túlsúly a top-10-ben
    
    Args:
        predicted_indices: Model által prediktált ranking indexek
        slate: Slate adatok [{"relevance": int, "court": str, "domain": str, ...}, ...]
        baseline_metrics: Baseline metrikák {"ndcg@10": float, "mrr@5": float, ...}
        weights: Metrika súlyok {"ndcg10": 0.6, "mrr5": 0.3, "recall20": 0.1}
        clip_range: Reward clipping tartomány (min, max)
        use_sigmoid: Ha True, sigmoid(delta) transzformáció
        mrr_tiebreak_bonus: Bonus ha csak MRR javul (+ε)
        diversity_penalty_weight: Diverzitás büntetés súlya (0 = kikapcsolva)
    
    Returns:
        (total_reward, components_dict)
    """
    # Default weights (updated: 0.6/0.3/0.1)
    if weights is None:
        weights = {"ndcg10": 0.60, "mrr5": 0.30, "recall20": 0.10}
    
    # Relevancia array a predicted ranking szerint
    relevances = [slate[i]["relevance"] for i in predicted_indices]
    
    # Policy metrikák
    policy_ndcg10 = compute_ndcg(relevances, k=10)
    policy_mrr5 = compute_mrr(relevances, k=5)
    
    # Recall@20: total_relevant a teljes slate-ből (fixált candidate pool!)
    total_relevant = sum(1 for doc in slate if doc.get("relevance", 0) > 0)
    policy_recall20 = compute_recall(relevances, k=20, total_relevant=total_relevant)
    
    # Baseline metrikák
    baseline_ndcg10 = baseline_metrics.get("ndcg@10", 0.0)
    baseline_mrr5 = baseline_metrics.get("mrr@5", 0.0)
    baseline_recall20 = baseline_metrics.get("recall@20", 0.0)
    
    # Delta számítás
    delta_ndcg10 = policy_ndcg10 - baseline_ndcg10
    delta_mrr5 = policy_mrr5 - baseline_mrr5
    delta_recall20 = policy_recall20 - baseline_recall20
    
    # Sigmoid transzformáció (stabilabb reward signal)
    if use_sigmoid:
        # sigmoid(x) = 1 / (1 + exp(-k*x)), k=5 → érzékenység
        def sigmoid_transform(delta, k=5.0):
            return 2.0 / (1.0 + np.exp(-k * delta)) - 1.0  # range: [-1, 1]
        
        component_ndcg = weights["ndcg10"] * sigmoid_transform(delta_ndcg10)
        component_mrr = weights["mrr5"] * sigmoid_transform(delta_mrr5)
        component_recall = weights["recall20"] * sigmoid_transform(delta_recall20)
    else:
        # Lineáris (original)
        component_ndcg = weights["ndcg10"] * delta_ndcg10
        component_mrr = weights["mrr5"] * delta_mrr5
        component_recall = weights["recall20"] * delta_recall20
    
    # Total reward
    reward_raw = component_ndcg + component_mrr + component_recall
    
    # ====== TIE-BREAK BONUS: MRR javulás ======
    # Ha nDCG/Recall nem változik jelentősen, de MRR javul → felhasználói élmény++
    if abs(delta_ndcg10) < 0.01 and abs(delta_recall20) < 0.01 and delta_mrr5 > 0.05:
        reward_raw += mrr_tiebreak_bonus
    
    # ====== DIVERZITÁS BÜNTETÉS ======
    # Ha a top-10 túl homogén (ugyanaz a court/domain dominál) → -ε
    if diversity_penalty_weight > 0:
        top10_indices = predicted_indices[:10]
        top10_courts = [slate[i].get("court", "N/A") for i in top10_indices if i < len(slate)]
        top10_domains = [slate[i].get("domain", "N/A") for i in top10_indices if i < len(slate)]
        
        # Shannon entropy (magas = diverzitás, alacsony = homogén)
        def shannon_entropy(items):
            from collections import Counter
            counts = Counter(items)
            total = len(items)
            if total == 0:
                return 0.0
            probs = [c/total for c in counts.values()]
            return -sum(p * np.log(p + 1e-9) for p in probs)
        
        court_entropy = shannon_entropy(top10_courts)
        domain_entropy = shannon_entropy(top10_domains)
        
        # Normalizálás: max entropy = log(10) ≈ 2.3
        max_entropy = np.log(10)
        diversity_score = (court_entropy + domain_entropy) / (2 * max_entropy)
        
        # Büntetés ha alacsony diverzitás (< 0.5 → homogén)
        if diversity_score < 0.5:
            penalty = diversity_penalty_weight * (0.5 - diversity_score)
            reward_raw -= penalty
    
    # Clipping
    reward_clipped = np.clip(reward_raw, clip_range[0], clip_range[1])
    
    # Komponensek dictionary (részletes tracking)
    components = {
        "reward_total": float(reward_clipped),
        "reward_raw": float(reward_raw),
        "component_ndcg10": float(component_ndcg),
        "component_mrr5": float(component_mrr),
        "component_recall20": float(component_recall),
        "delta_ndcg10": float(delta_ndcg10),
        "delta_mrr5": float(delta_mrr5),
        "delta_recall20": float(delta_recall20),
        "policy_ndcg10": float(policy_ndcg10),
        "policy_mrr5": float(policy_mrr5),
        "policy_recall20": float(policy_recall20),
        "baseline_ndcg10": float(baseline_ndcg10),
        "baseline_mrr5": float(baseline_mrr5),
        "baseline_recall20": float(baseline_recall20),
    }
    
    return float(reward_clipped), components


def aggregate_reward_components(
    all_components: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Reward komponensek aggregálása (batch átlag).
    
    Args:
        all_components: Lista komponens dictionary-kről
    
    Returns:
        Aggregált statisztikák
    """
    if not all_components:
        return {}
    
    # Minden key-re számítsuk az átlagot
    keys = all_components[0].keys()
    aggregated = {}
    
    for key in keys:
        values = [comp[key] for comp in all_components if key in comp]
        if values:
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
            aggregated[f"{key}_min"] = np.min(values)
            aggregated[f"{key}_max"] = np.max(values)
    
    return aggregated


# Export public API
__all__ = [
    "compute_mrr",
    "compute_recall",
    "compute_ndcg",
    "compute_multi_metric_reward",
    "aggregate_reward_components",
]
