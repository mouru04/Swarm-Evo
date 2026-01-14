"""
优化模块

包含用于分析和改进系统性能的组件，特别是prompt反思器、生成器和版本管理器。
"""

from .reflector import PromptReflector, PerformanceMetrics
from .generator import PromptGenerator, GenerationResult
from .version_manager import (
    AgentVersionManager,
    AgentEvolutionRecord,
    PromptVersionRecord,
    NodeMetadata,
    TaskReviewRecord
)

__all__ = [
    'PromptReflector',
    'PerformanceMetrics',
    'PromptGenerator',
    'GenerationResult',
    'AgentVersionManager',
    'AgentEvolutionRecord',
    'PromptVersionRecord',
    'NodeMetadata',
    'TaskReviewRecord'
]
