"""
Code agent interface for the MLflow improve system.

Defines the abstract interface that any code agent (Claude Code, OpenCode,
etc.) must implement to create fix PRs from detected issues.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class FixRequest:
    """A request to fix an issue in a repository."""
    issue_id: str
    issue_name: str
    issue_description: str
    root_causes: list[str]
    repo_url: str
    branch: str
    experiment_id: str


@dataclass
class FixResult:
    """The result of a fix attempt."""
    success: bool
    pr_url: str | None = None
    pr_number: int | None = None
    error: str | None = None
    changes_summary: str | None = None


class CodeAgent(ABC):
    """Interface for code agents that can analyze repos and create fix PRs.

    Implementations should:
    1. Clone or access the repository
    2. Analyze the issue against the codebase
    3. Create a branch with the fix
    4. Open a pull request
    5. Return the PR URL
    """

    @abstractmethod
    def create_fix(self, request: FixRequest) -> FixResult:
        """Analyze the issue against the repo and create a PR with a fix."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Return the agent name (e.g., 'claude-code', 'opencode')."""
        ...


_AGENT_REGISTRY: dict[str, type[CodeAgent]] = {}


def register_agent(name: str, agent_cls: type[CodeAgent]):
    """Register a code agent implementation."""
    _AGENT_REGISTRY[name] = agent_cls


def get_agent(name: str) -> CodeAgent:
    """Get a registered code agent by name."""
    if name not in _AGENT_REGISTRY:
        available = ", ".join(_AGENT_REGISTRY.keys()) or "none"
        raise ValueError(
            f"Unknown code agent '{name}'. Available: {available}. "
            f"Register one with mlflow.genai.improve.code_agent.register_agent()."
        )
    return _AGENT_REGISTRY[name]()
