"""
Suggestion engine for the MLflow improve system.

Takes findings from the analyzer and generates actionable improvement
suggestions. Each suggestion includes what to change, why, and a
confidence level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from .analyzer import Finding


@dataclass
class Suggestion:
    """An actionable improvement suggestion."""
    id: str
    type: str  # "model_upgrade", "prompt_fix", "config_change", "investigate"
    severity: str  # "low", "medium", "high"
    title: str
    description: str
    action: str
    confidence: float  # 0.0 to 1.0
    auto_applicable: bool  # safe to auto-apply without human review
    evidence: dict = field(default_factory=dict)


_SUGGESTION_COUNTER = 0


def _next_id() -> str:
    global _SUGGESTION_COUNTER
    _SUGGESTION_COUNTER += 1
    return f"s-{_SUGGESTION_COUNTER:04d}"


def generate_suggestions(findings: list[Finding]) -> list[Suggestion]:
    """Generate improvement suggestions from analyzer findings.

    Args:
        findings: List of Finding objects from analyze_traces()

    Returns:
        List of Suggestion objects, sorted by severity (high first).
    """
    suggestions = []
    for finding in findings:
        handler = _PATTERN_HANDLERS.get(finding.pattern)
        if handler:
            suggestion = handler(finding)
            if suggestion:
                suggestions.append(suggestion)

    severity_order = {"high": 0, "medium": 1, "low": 2}
    suggestions.sort(key=lambda s: severity_order.get(s.severity, 3))
    return suggestions


def _handle_context_bloat(finding: Finding) -> Suggestion:
    max_size = finding.evidence.get("max_size_bytes", 0)
    avg_size = finding.evidence.get("avg_size_bytes", 0)

    return Suggestion(
        id=_next_id(),
        type="model_upgrade",
        severity=finding.severity,
        title="Context window pressure detected",
        description=(
            f"Traces are averaging {avg_size / 1_000_000:.1f}MB "
            f"(max {max_size / 1_000_000:.1f}MB). "
            f"As resume count grows, the context window will fill up "
            f"and the agent will start dropping information or failing."
        ),
        action="Switch to a model with a larger context window (e.g., gpt-5.4-max with 1M tokens).",
        confidence=0.85 if max_size > 2_000_000 else 0.7,
        auto_applicable=True,
        evidence=finding.evidence,
    )


def _handle_context_growth(finding: Finding) -> Suggestion:
    ratio = finding.evidence.get("growth_ratio", 1)

    return Suggestion(
        id=_next_id(),
        type="investigate",
        severity=finding.severity,
        title="Trace sizes growing over time",
        description=(
            f"Recent traces are {ratio:.1f}x larger than older ones. "
            f"This could indicate conversation history accumulation, "
            f"larger resume batches, or unnecessary data in tool responses."
        ),
        action="Review conversation history management. Consider summarizing earlier turns instead of keeping full history.",
        confidence=0.6,
        auto_applicable=False,
        evidence=finding.evidence,
    )


def _handle_tool_redundancy(finding: Finding) -> Suggestion:
    dupes = finding.evidence.get("duplicate_counts", {})
    worst = max(dupes, key=dupes.get) if dupes else "unknown"
    rate = finding.evidence.get("rate", 0)

    return Suggestion(
        id=_next_id(),
        type="prompt_fix",
        severity=finding.severity,
        title=f"Redundant tool calls: {worst}",
        description=(
            f"The agent is calling {worst} multiple times in {rate:.0%} of runs. "
            f"This wastes API calls and increases latency."
        ),
        action=f"Add instruction to the system prompt: 'Do not call {worst} more than once per pipeline run unless processing different candidates.'",
        confidence=0.75,
        auto_applicable=False,
        evidence=finding.evidence,
    )


def _handle_score_degradation(finding: Finding) -> Suggestion:
    scorer = finding.evidence.get("scorer", "unknown")
    pass_rate = finding.evidence.get("pass_rate", 0)

    action_map = {
        "completeness": "Review if the agent is completing all requested steps. Check if the system prompt clearly lists all pipeline steps.",
        "tool_call_correctness": "Review recent tool call arguments. The agent may be passing incorrect parameters to MCP tools.",
        "tool_call_efficiency": "The agent is making unnecessary tool calls. Consider adding explicit instructions about which tools to use for each task.",
        "relevance_to_query": "The agent's responses are drifting off-topic. Review the system prompt's scope restrictions.",
    }

    return Suggestion(
        id=_next_id(),
        type="prompt_fix" if scorer in action_map else "investigate",
        severity=finding.severity,
        title=f"{scorer} score is low ({pass_rate:.0%})",
        description=(
            f"The {scorer} evaluation is passing only {pass_rate:.0%} of the time. "
            f"This indicates the agent's quality is below acceptable levels."
        ),
        action=action_map.get(scorer, f"Investigate why {scorer} is failing. Review recent traces for patterns."),
        confidence=0.8,
        auto_applicable=False,
        evidence=finding.evidence,
    )


def _handle_score_declining(finding: Finding) -> Suggestion:
    scorer = finding.evidence.get("scorer", "unknown")
    recent = finding.evidence.get("recent_rate", 0)
    older = finding.evidence.get("older_rate", 0)

    return Suggestion(
        id=_next_id(),
        type="investigate",
        severity=finding.severity,
        title=f"{scorer} quality declining",
        description=(
            f"{scorer} dropped from {older:.0%} to {recent:.0%} in recent runs. "
            f"Something changed that is affecting agent quality."
        ),
        action="Compare recent traces with older successful ones. Look for prompt changes, different input patterns, or infrastructure changes.",
        confidence=0.65,
        auto_applicable=False,
        evidence=finding.evidence,
    )


def _handle_slow_execution(finding: Finding) -> Suggestion:
    avg_ms = finding.evidence.get("avg_ms", 0)

    return Suggestion(
        id=_next_id(),
        type="config_change",
        severity=finding.severity,
        title=f"Slow pipeline ({avg_ms / 1000:.0f}s average)",
        description=(
            f"Pipeline runs are averaging {avg_ms / 1000:.0f} seconds. "
            f"This may be due to large batch sizes, many scoring passes, "
            f"or slow tool responses."
        ),
        action="Consider reducing scoring passes from 3 to 1 for initial filtering, or processing resumes in smaller batches.",
        confidence=0.6,
        auto_applicable=False,
        evidence=finding.evidence,
    )


def _handle_execution_slowdown(finding: Finding) -> Suggestion:
    ratio = finding.evidence.get("ratio", 1)

    return Suggestion(
        id=_next_id(),
        type="investigate",
        severity=finding.severity,
        title=f"Execution time increasing ({ratio:.1f}x slower)",
        description=(
            f"Recent runs are {ratio:.1f}x slower than older ones. "
            f"This could indicate growing input size, API rate limiting, "
            f"or context window pressure."
        ),
        action="Check if resume count has increased. Monitor API response times. Consider if the model needs upgrading.",
        confidence=0.6,
        auto_applicable=False,
        evidence=finding.evidence,
    )


def _handle_error_spike(finding: Finding) -> Suggestion:
    total = finding.evidence.get("total_errors", 0)
    rate = finding.evidence.get("error_rate", 0)

    return Suggestion(
        id=_next_id(),
        type="investigate",
        severity=finding.severity,
        title=f"Tool errors increasing ({rate:.0%} of runs affected)",
        description=(
            f"Tool calls are failing in {rate:.0%} of recent runs "
            f"({total} total errors). This could indicate MCP server issues, "
            f"API quota limits, or incorrect tool arguments."
        ),
        action="Check MCP server logs. Verify API keys and rate limits. Review tool call arguments in failing traces.",
        confidence=0.7,
        auto_applicable=False,
        evidence=finding.evidence,
    )


def _handle_incomplete_pipeline(finding: Finding) -> Suggestion:
    missing = finding.evidence.get("missing_tools", {})

    return Suggestion(
        id=_next_id(),
        type="prompt_fix",
        severity=finding.severity,
        title="Pipeline steps being skipped",
        description=(
            f"The agent is not completing all pipeline steps. "
            f"Missing: {', '.join(missing.keys())}."
        ),
        action="Review the system prompt's workflow section. Ensure all pipeline steps are clearly listed and the agent is instructed to complete them in order.",
        confidence=0.75,
        auto_applicable=False,
        evidence=finding.evidence,
    )


_PATTERN_HANDLERS = {
    "context_bloat": _handle_context_bloat,
    "context_growth": _handle_context_growth,
    "tool_redundancy": _handle_tool_redundancy,
    "score_degradation": _handle_score_degradation,
    "score_declining": _handle_score_declining,
    "slow_execution": _handle_slow_execution,
    "execution_slowdown": _handle_execution_slowdown,
    "error_spike": _handle_error_spike,
    "incomplete_pipeline": _handle_incomplete_pipeline,
}
