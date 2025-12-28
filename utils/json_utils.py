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
    
    # 1. Try to find code blocks
    # Pattern to match ```(optional lang) content ```
    # We use findall to get ALL blocks, not just the first one
    code_block_pattern = re.compile(r"```(?P<lang>\w+)?\s*(?P<content>.*?)\s*```", re.DOTALL)
    matches = code_block_pattern.findall(cleaned)
    
    candidate_content = None

    # Strategy A: Explicit "json" tag
    for lang, content in matches:
        if lang and lang.lower() == "json":
            candidate_content = content
            break
            
    # Strategy B: If no explicit json tag, look for a block that looks like a JSON object
    if not candidate_content and matches:
        for _, content in matches:
            stripped = content.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                candidate_content = content
                break
                
    # If we found a block, use it. Otherwise continue with original text (cleaned)
    if candidate_content:
        cleaned = candidate_content
            
    # 2. Extract strictly from first '{' to last '}'
    # This handles surrounding whitespace or text even within the block
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned = cleaned[start_idx : end_idx + 1]
        
    return cleaned

def parse_json_output(text: str, suppress_error_log: bool = False) -> Dict[str, Any]:
    """
    Robustly parse JSON output from LLM.
    
    Args:
        text: Raw output from LLM.
        suppress_error_log: If True, do not log ERROR on failure.
        
    Returns:
        Parsed dictionary.
        
    Raises:
        ValueError: If JSON parsing fails.
    """
    normalized = normalize_json_output(text)
    try:
        return json.loads(normalized)
    except json.JSONDecodeError as e:
        if not suppress_error_log:
            with open("error.txt", "w") as f:
                f.write(text)
            log_msg("ERROR", f"Failed to parse JSON output: {e}\nNormalized text: {normalized}")
        raise e
