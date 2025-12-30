"""Infrastructure for tracking gene-level pheromones."""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Any, Optional

from core.execution.journal import Node, Journal


_LOCUS_NAMES = [
    "DATA",
    "MODEL",
    "LOSS",
    "OPTIMIZER",
    "REGULARIZATION",
    "INITIALIZATION",
    "TRAINING_TRICKS",
]


def normalize_gene_text(text: str) -> str:
    """Normalize gene text for stable hashing."""
    if text is None:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = normalized.strip()
    # Collapse consecutive blank lines to a single blank line
    normalized = re.sub(r"\n\s*\n+", "\n\n", normalized)
    return normalized


def compute_gene_id(text: str) -> str:
    normalized = normalize_gene_text(text)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


class GeneRegistry:
    """Tracks pheromone statistics for individual genes per locus."""

    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, Dict[str, Any]]] = {
            locus: {} for locus in _LOCUS_NAMES
        }

    def update_from_reviewed_node(self, node: Node) -> None:
        """Update registry using a reviewed node's pheromone."""
        pheromone_value = getattr(node, "pheromone_node", None)
        if pheromone_value is None and node.metadata:
            pheromone_value = node.metadata.get("pheromone_node")
        if pheromone_value is None:
            return
        if node.score is None or node.is_buggy:
            return
        for locus in _LOCUS_NAMES:
            gene_content = node.genes.get(locus) if node.genes else None
            if not gene_content:
                continue
            normalized = normalize_gene_text(gene_content)
            if not normalized:
                continue
            gene_id = compute_gene_id(normalized)
            entry = self._registry[locus].setdefault(
                gene_id,
                {
                    "pheromone": 0.1,
                    "acc_sum": 0.0,
                    "count": 0,
                    "last_seen_step": -1,
                    "content": gene_content,
                    "source_node_id": node.id,
                },
            )
            entry["acc_sum"] += float(pheromone_value)
            entry["count"] += 1
            entry["pheromone"] = max(entry["acc_sum"] / entry["count"], 1e-9)
            entry["last_seen_step"] = node.step
            entry["content"] = gene_content
            entry["source_node_id"] = node.id

#gene_pool全集合构建
    def get_gene_pheromone(self, locus: str, gene_id: str, default_init: float = 0.1) -> float:
        locus_entries = self._registry.get(locus)
        if not locus_entries:
            return default_init
        entry = locus_entries.get(gene_id)
        if not entry:
            return default_init
        return float(entry.get("pheromone", default_init))

    def build_gene_pools(self, journal: Optional[Journal] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Return gene pools for all loci."""
        pools: Dict[str, List[Dict[str, Any]]] = {locus: [] for locus in _LOCUS_NAMES}
        for locus, entries in self._registry.items():
            for gene_id, record in entries.items():
                pools[locus].append(
                    {
                        "gene_id": gene_id,
                        "content": record.get("content", ""),
                        "pheromone": record.get("pheromone", 0.1),
                        "source_node_id": record.get("source_node_id"),
                        "last_seen_step": record.get("last_seen_step", -1),
                    }
                )
        return pools
