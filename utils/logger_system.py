import os
import json
import datetime
from typing import Dict, Any, List

class LoggerSystem:
    def __init__(self, log_dir: str):
        """
        Initialize the LoggerSystem.
        
        Args:
            log_dir: Directory path where logs will be saved.
        """
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.text_log_path = os.path.join(self.log_dir, "system.log")
        self.json_log_path = os.path.join(self.log_dir, "metrics.json")
        
        # Initialize JSON log list
        self.json_data: List[Dict[str, Any]] = []
        if os.path.exists(self.json_log_path):
            try:
                with open(self.json_log_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content:
                        self.json_data = json.loads(content)
            except json.JSONDecodeError:
                self.json_data = [] # Reset if corrupt
                
    def text_log(self, level: str, message: str) -> None:
        """
        Log a message to the text log file.
        
        Args:
            level: Log level/type (e.g., 'INFO', 'WARNING', 'ERROR')
            message: The message to log
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        with open(self.text_log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
            
    def json_log(self, data: Dict[str, Any]) -> None:
        """
        Log a dictionary to the JSON log file.
        
        Args:
            data: Dictionary containing data to log
        """
        self.json_data.append(data)
        
        with open(self.json_log_path, 'w', encoding='utf-8') as f:
            json.dump(self.json_data, f, indent=4, ensure_ascii=False)
