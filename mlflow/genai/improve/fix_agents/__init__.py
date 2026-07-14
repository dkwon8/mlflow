"""
Built-in code agent implementations for the MLflow improve system.

Register agents on import so they're available via get_agent().
"""

from mlflow.genai.improve.fix_agents.claude_code_agent import ClaudeCodeAgent
from mlflow.genai.improve.fix_agent_registry import register_agent

register_agent("claude-code", ClaudeCodeAgent)
