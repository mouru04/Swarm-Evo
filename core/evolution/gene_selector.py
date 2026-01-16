"""
Deterministic per-locus gene selection using
Quality–Diversity scoring (NO DPP).

Quality  = 0.3 * node_pheromone + 0.7 * gene_pheromone
Diversity = 1 - max_sim (against recent winners of the same locus)

Final score = quality + DIVERSITY_WEIGHT * diversity
Top-1 selected per locus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any
from collections import deque

import numpy as np
import faiss

from utils.logger_system import log_msg

from core.execution.journal import Journal, Node
from core.evolution.embedding_manager import CodeEmbeddingManager
from core.evolution.gene_registry import (
    GeneRegistry,
    compute_gene_id,
    normalize_gene_text,
)

# =========================
# Constants & Hyperparams
# =========================

LOCUS_TO_FIELD = {
    "DATA": "data_source",
    "MODEL": "model_source",
    "LOSS": "loss_source",
    "OPTIMIZER": "optimizer_source",
    "REGULARIZATION": "regularization_source",
    "INITIALIZATION": "initialization_source",
    "TRAINING_TRICKS": "tricks_source",
}

ALL_LOCI = list(LOCUS_TO_FIELD.keys())

DEFAULT_INIT_PHEROMONE = 0.1

# Quality blend 
QUALITY_BLEND = 0.3

# Diversity strength (λ)
DIVERSITY_WEIGHT = 0.3

# recent winner window per locus
RECENT_WINDOW = 5

# =========================
# Embedding manager (reuse)
# =========================

embedding_manager = CodeEmbeddingManager()

# =========================
# Recent selected buffer
# =========================

RECENT_SELECTED: Dict[str, deque[str]] = {
    locus: deque(maxlen=RECENT_WINDOW)
    for locus in ALL_LOCI
}

# =========================
# Data structure
# =========================

@dataclass
class GeneItem:
    locus: str
    gene_id: str
    content: str          # normalized (for embedding / similarity)
    raw_content: str      # original code (for merge)
    source_node_id: str
    gene_pheromone: float
    node_pheromone: float
    source_score: float
    created_step: int
    quality: float = 0.0


# =========================
# Public API
# =========================

def select_gene_plan(
    journal: Journal,
    gene_registry: GeneRegistry,
    current_step: int,
    # use_dpp: bool = False,   
    # dpp_k: int = 1,          
) -> Dict[str, Any]:
    """
    Select a merge-compatible gene plan using
    Quality–Diversity top-1 per locus.
    """
    pools = build_decision_gene_pools(journal, gene_registry)

    for locus, items in pools.items():
        log_msg(
            "INFO",
            f"[POOL] {locus}: size={len(items)} "
            f"genes={[i.gene_id[:6] for i in items]}"
        )

    winners: Dict[str, GeneItem] = {}
    for locus in ALL_LOCI:
        items = pools.get(locus, [])
        if not items:
            raise ValueError(f"Missing locus: {locus}")
        winner = _select_locus_winner(items)
        winners[locus] = winner

    gene_plan: Dict[str, Any] = {
        "reasoning": (
            "Per-locus selection: Top-1 by "
            "Quality–Diversity score "
            "(quality=0.7*node+0.3*gene, "
            "diversity=1-max_sim vs recent winners)."
        )
    }

    for locus, item in winners.items():
        field_name = LOCUS_TO_FIELD[locus]
        gene_plan[field_name] = {
            "locus": locus,
            "source_node_id": item.source_node_id,
            "gene_id": item.gene_id,
            "code": item.raw_content,
        }

    return gene_plan


# =========================
# Gene pool construction
# =========================

def build_decision_gene_pools(
    journal: Journal,
    gene_registry: GeneRegistry,
) -> Dict[str, List[GeneItem]]:
    pools: Dict[str, Dict[str, GeneItem]] = {locus: {} for locus in ALL_LOCI}

    for node in journal.nodes.values():
        if not _is_valid_node(node):
            continue

        for locus in ALL_LOCI:
            raw_content = (node.genes or {}).get(locus)
            if not raw_content:
                continue

            normalized = normalize_gene_text(raw_content)
            if not normalized:
                continue

            gene_id = compute_gene_id(normalized)
            gene_pheromone = gene_registry.get_gene_pheromone(
                locus, gene_id, DEFAULT_INIT_PHEROMONE
            )

            node_pheromone = 0.0
            if node.metadata:
                node_pheromone = float(node.metadata.get("pheromone_node", 0.0))

            item = GeneItem(
                locus=locus,
                gene_id=gene_id,
                content=normalized,
                raw_content=raw_content,
                source_node_id=node.id,
                gene_pheromone=float(gene_pheromone),
                node_pheromone=float(node_pheromone),
                source_score=float(node.score),
                created_step=int(node.step),
            )
            item.quality = _compute_quality(item)

            existing = pools[locus].get(gene_id)
            if existing is None or _is_better_item(item, existing):
                pools[locus][gene_id] = item

    return {locus: list(items.values()) for locus, items in pools.items()}


# =========================
# Core selection logic
# =========================

def _select_locus_winner(items: List[GeneItem]) -> GeneItem:
    """
    Select top-1 gene for a locus using:
    final_score = quality + DIVERSITY_WEIGHT * (1 - max_sim)
    """
    locus = items[0].locus

    # -----------------------------
    # Prepare recent reference texts
    # -----------------------------
    recent_gene_ids = list(RECENT_SELECTED[locus])
    recent_texts: List[str] = []

    if recent_gene_ids:
        content_map = {item.gene_id: item.content for item in items}
        for gid in recent_gene_ids:
            txt = content_map.get(gid)
            if txt:
                recent_texts.append(txt)

    # -----------------------------
    # Compute embeddings
    # -----------------------------
    candidate_texts = [item.content for item in items]
    candidate_vecs = embedding_manager.embed_texts(candidate_texts)

    if recent_texts:
        recent_vecs = embedding_manager.embed_texts(recent_texts)
    else:
        recent_vecs = None

    # -----------------------------
    # Compute max-sim per candidate
    # -----------------------------
    if recent_vecs is None or len(recent_vecs) == 0:
        max_sims = [0.0 for _ in items]
    else:
        index = faiss.IndexFlatIP(recent_vecs.shape[1])
        index.add(recent_vecs)

        sims, _ = index.search(candidate_vecs, 1)
        sims = sims.reshape(-1)
        max_sims = [float((s + 1.0) / 2.0) for s in sims]

    # -----------------------------
    # Score & select
    # -----------------------------
    scored = []
    for item, max_sim in zip(items, max_sims):
        diversity = 1.0 - max_sim
        final_score = item.quality + DIVERSITY_WEIGHT * diversity
        scored.append((final_score, diversity, item))

    scored.sort(
        key=lambda x: (
            -x[0],                # final score
            -x[1],                # diversity
            -x[2].quality,
            -x[2].node_pheromone,
            -x[2].source_score,
            x[2].source_node_id,
            x[2].gene_id,
        )
    )

    winner = scored[0][2]
    RECENT_SELECTED[locus].append(winner.gene_id)

    log_msg(
        "INFO",
        f"[WINNER] {locus}: "
        f"gene={winner.gene_id[:6]} "
        f"quality={winner.quality:.4f} "
        f"diversity={scored[0][1]:.4f} "
        f"final={scored[0][0]:.4f} "
        f"node={winner.source_node_id[:6]}"
    )

    return winner


# =========================
# Utilities
# =========================

def _is_valid_node(node: Node) -> bool:
    if node.score is None or node.is_buggy:
        return False
    if not node.code:
        return False
    if not node.genes:
        return False
    review_success = node.metadata.get("review_success") if node.metadata else None
    if review_success is not None and not review_success:
        return False
    return True


def _is_better_item(candidate: GeneItem, incumbent: GeneItem) -> bool:
    if candidate.gene_pheromone != incumbent.gene_pheromone:
        return candidate.gene_pheromone > incumbent.gene_pheromone
    if candidate.node_pheromone != incumbent.node_pheromone:
        return candidate.node_pheromone > incumbent.node_pheromone
    if candidate.source_score != incumbent.source_score:
        return candidate.source_score > incumbent.source_score
    return candidate.source_node_id < incumbent.source_node_id


def _compute_quality(item: GeneItem) -> float:
    return (
        QUALITY_BLEND * item.gene_pheromone
        + (1.0 - QUALITY_BLEND) * item.node_pheromone
    )

