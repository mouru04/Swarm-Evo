"""
Journal module for managing the solution DAG (Directed Acyclic Graph).
Stores the history of all generated nodes (solutions) and their relationships.
"""

import uuid
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Set
from collections import deque

@dataclass
class Node:
    """Represents a single node in the solution search DAG."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_ids: List[str] = field(default_factory=list)
    
    # Code content
    code: str = ""
    
    # Evaluation results
    score: Optional[float] = None
    is_buggy: bool = False
    submission_created: bool = False
    
    # Execution context
    logs: str = ""
    summary: str = ""  # LLM evaluation summary
    
    # Metadata
    step: int = 0  # Global step number when this node was created
    action_type: str = "draft"  # draft, improve, debug, merge, etc.
    metadata: Dict[str, Any] = field(default_factory=dict) # Edge attributes or extra info
    
    children_ids: List[str] = field(default_factory=list)

    @property
    def parent_id(self) -> Optional[str]:
        """Backward compatibility for single parent access (returns primary parent)."""
        return self.parent_ids[0] if self.parent_ids else None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        # Handle legacy data with parent_id
        if 'parent_id' in data and 'parent_ids' not in data:
            pid = data.pop('parent_id')
            if pid:
                data['parent_ids'] = [pid]
            else:
                data['parent_ids'] = []
        return cls(**data)


class Journal:
    """Manages the DAG of Nodes."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.root_id: Optional[str] = None

    def add_node(self, node: Node) -> None:
        """Adds a new node to the journal and updates graph connections."""
        self.nodes[node.id] = node
        
        if node.parent_ids:
            for pid in node.parent_ids:
                parent = self.nodes.get(pid)
                if parent:
                    if node.id not in parent.children_ids:
                        parent.children_ids.append(node.id)
        else:
            # If no parent and no root, this is the root
            if self.root_id is None:
                self.root_id = node.id

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def get_best_node(self) -> Optional[Node]:
        """Returns the node with the highest score."""
        valid_nodes = [n for n in self.nodes.values() if n.score is not None and not n.is_buggy]
        if not valid_nodes:
            return None
        # Assuming higher score is better.
        return max(valid_nodes, key=lambda n: n.score)

    def get_leaf_nodes(self) -> List[Node]:
        """Returns all nodes that have no children."""
        return [n for n in self.nodes.values() if not n.children_ids]

    def get_trace(self, node_id: str) -> List[Node]:
        """
        Returns the primary lineage path from root to the specified node.
        
        This follows the first parent at each step, maintaining backward compatibility
        with tree-based assumptions.
        """
        trace = []
        current_id = node_id
        while current_id:
            node = self.nodes.get(current_id)
            if not node:
                break
            trace.append(node)
            # Follow primary lineage
            current_id = node.parent_id
        return list(reversed(trace))

    def get_ancestors(self, node_id: str) -> List[Node]:
        """
        Returns all ancestor nodes of the specified node, topologically sorted 
        (ancestors appear before descendants). Includes the node itself.
        """
        if node_id not in self.nodes:
            return []
            
        visited = set()
        sorted_nodes = []
        
        def visit(nid):
            if nid in visited:
                return
            visited.add(nid)
            node = self.nodes.get(nid)
            if node:
                for pid in node.parent_ids:
                    visit(pid)
                sorted_nodes.append(node)
        
        visit(node_id)
        return sorted_nodes

    def get_topological_sort(self) -> List[Node]:
        """Returns a topological sort of all nodes in the graph."""
        visited = set()
        stack = []
        
        def dfs(nid):
            visited.add(nid)
            node = self.nodes.get(nid)
            if not node: return
            for child_id in node.children_ids:
                if child_id not in visited:
                    dfs(child_id)
            stack.append(node)

        # Iterate over all nodes to ensure we cover disconnected components
        for nid in self.nodes:
            if nid not in visited:
                dfs(nid)
        
        return stack[::-1]

    def save_to_file(self, path: str) -> None:
        """Saves the entire journal to a JSON file."""
        data = {
            "root_id": self.root_id,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()}
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, path: str) -> 'Journal':
        """Loads the journal from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        journal = cls()
        journal.root_id = data.get("root_id")
        for nid, node_data in data.get("nodes", {}).items():
            journal.nodes[nid] = Node.from_dict(node_data)
        return journal

    def __len__(self) -> int:
        return len(self.nodes)
