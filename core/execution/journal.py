"""
Journal module for managing the solution DAG (Directed Acyclic Graph).
Stores the history of all generated nodes (solutions) and their relationships.
"""

import uuid
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Set
from collections import deque
import re
import ast
from utils.logger_system import log_msg

@dataclass
class Node:
    """Represents a single node in the solution search DAG."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_ids: List[str] = field(default_factory=list)
    
    # Code content
    code: str = ""
    genes: Dict[str, str] = field(default_factory=dict) # Stores parsed code components
    
    # Evaluation results
    score: Optional[float] = None
    is_buggy: bool = False
    submission_created: bool = False
    archive_path: Optional[str] = None # Path to the zipped solution/submission archive
    
    # Execution context
    logs: str = ""
    summary: str = ""  # LLM evaluation summary
    
    # Metadata
    step: int = 0  # Global step number when this node was created
    action_type: str = "draft"  # draft, improve, debug, merge, etc.
    metadata: Dict[str, Any] = field(default_factory=dict) # Edge attributes or extra info
    
    children_ids: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.code and not self.genes:
            self.genes = parse_solution_genes(self.code)

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
        self.score_min: Optional[float] = None
        self.score_max: Optional[float] = None

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
        """
        返回得分最高的节点。

        Returns:
            Optional[Node]: 得分最高的节点，若没有有效节点则返回None
        """
        valid_nodes = [n for n in self.nodes.values() if n.score is not None and not n.is_buggy]
        if not valid_nodes:
            return None
        return max(valid_nodes, key=lambda n: n.score if n.score is not None else 0.0)

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
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "score_min": self.score_min,
            "score_max": self.score_max,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, path: str) -> 'Journal':
        """
        从JSON文件加载Journal对象。

        Args:
            path: JSON文件路径

        Returns:
            Journal: 加载的Journal对象
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        journal = cls()
        journal.root_id = data.get("root_id")

        if "score_min" not in data:
            log_msg("ERROR", "加载的Journal文件缺少score_min字段，无法加载。请使用包含完整score信息的文件")
        if "score_max" not in data:
            log_msg("ERROR", "加载的Journal文件缺少score_max字段，无法加载。请使用包含完整score信息的文件")

        journal.score_min = data["score_min"]
        journal.score_max = data["score_max"]

        if journal.score_min is None:
            log_msg("ERROR", "score_min为None，无法加载。请确保Journal文件包含有效的score信息")
        if journal.score_max is None:
            log_msg("ERROR", "score_max为None，无法加载。请确保Journal文件包含有效的score信息")

        for nid, node_data in data.get("nodes", {}).items():
            journal.nodes[nid] = Node.from_dict(node_data)
        return journal

    def __len__(self) -> int:
        return len(self.nodes)


def parse_solution_genes(code: str) -> Dict[str, str]:
    """
    Parses the solution code into 7 distinct gene components based on 
    # [SECTION: NAME] delimiters.
    
    Args:
        code (str): The full solution.py source code.
        
    Returns:
        Dict[str, str]: A dictionary where keys are section names (DATA, MODEL, etc.)
                        and values are the corresponding code blocks.
    """
    genes = {}
    
    # Define the 7 expected sections
    expected_sections = [
        "DATA", "MODEL", "LOSS", "OPTIMIZER", 
        "REGULARIZATION", "INITIALIZATION", "TRAINING_TRICKS",
        "MAIN_LOOP" # Optional but good to have
    ]
    
    # Regex to find all sections
    # Matches: # [SECTION: NAME] ... content ... (until next # [SECTION: or End of String)
    pattern = re.compile(r"^#\s*\[SECTION:\s*(\w+)\]", re.MULTILINE)
    
    matches = list(pattern.finditer(code))
    
    for i, match in enumerate(matches):
        section_name = match.group(1)
        start_idx = match.end()
        
        # Determine end index (next match start or end of string)
        if i + 1 < len(matches):
            end_idx = matches[i+1].start()
        else:
            end_idx = len(code)
            
        content = code[start_idx:end_idx].strip()
        
        if section_name in expected_sections or True: # Allow capturing all valid sections
            genes[section_name] = content
            
    return genes
