"""JSON formatting and parsing utilities for Agent communication."""

import json
import re
from typing import Any, Dict, List, Optional, Union
from utils.logger_system import log_msg

def render_history_json(records: List[Dict[str, Any]]) -> str:
    """
    Render history records as a JSON array string.
    
    Args:
        records: List of history step dictionaries.
        
    Returns:
        A formatted JSON string representing the history.
    """
    if not records:
        return ""
    
    # We serialize the whole list to a pretty-printed JSON string
    return json.dumps(records, indent=2, ensure_ascii=False)


def normalize_json_output(text: str) -> str:
    """
    Cleaning and extracting JSON string from LLM output.
    1. Removes markdown code blocks (```json ... ```).
    2. Finds the first '{' and the last '}' to handle extra text.
    
    Args:
        text: Raw output from LLM.
        
    Returns:
        Cleaned JSON string candidate.
    """
    if not text:
        return ""
        
    cleaned = text.strip()
    
    # 1. Remove markdown code blocks
    if "```" in cleaned:
        # Patter to match ```json (content) ``` or just ``` (content) ```
        pattern = re.compile(r"```(?:\w+)?\s*(?P<content>.*?)\s*```", re.DOTALL)
        match = pattern.search(cleaned)
        if match:
            cleaned = match.group("content").strip()
            
    # 2. Extract strictly from first '{' to last '}'
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned = cleaned[start_idx : end_idx + 1]
        
    return cleaned

def parse_json_output(text: str) -> Dict[str, Any]:
    """
    Robustly parse JSON output from LLM.
    
    Args:
        text: Raw output from LLM.
        
    Returns:
        Parsed dictionary.
        
    Raises:
        ValueError: If JSON parsing fails.
    """
    normalized = normalize_json_output(text)
    try:
        return json.loads(normalized)
    except json.JSONDecodeError as e:
        log_msg("ERROR", f"Failed to parse JSON output: {e}\nNormalized text: {normalized}")
