"""
Built-in code agent implementations for the MLflow improve system.

Register agents on import so they're available via get_agent().
"""

from mlflow.genai.improve.agents.claude_code_agent import ClaudeCodeAgent
from mlflow.genai.improve.code_agent import register_agent

register_agent("claude-code", ClaudeCodeAgent)
