import logging

from mlflow.client import MlflowClient
from mlflow.entities.issue import IssueSeverity, IssueStatus

_logger = logging.getLogger(__name__)
from mlflow.entities.run_status import RunStatus
from mlflow.genai.improve import analyze
from mlflow.server.jobs import job
from mlflow.utils.mlflow_tags import (
    MLFLOW_IMPROVE_CODE_AGENT,
    MLFLOW_IMPROVE_GITHUB_BRANCH,
    MLFLOW_IMPROVE_GITHUB_REPO,
)


_SEVERITY_MAP = {
    "high": IssueSeverity.HIGH,
    "medium": IssueSeverity.MEDIUM,
    "low": IssueSeverity.LOW,
}


@job(name="invoke_improve_analysis", max_workers=2)
def invoke_improve_analysis_job(
    experiment_id: str,
    trace_count: int = 10,
    run_id: str | None = None,
):
    """
    Job function to run improve analysis on an experiment's traces.

    Analyzes recent traces for performance patterns (context bloat,
    tool redundancy, score degradation, etc.) and creates Issue
    entities for each finding.
    """
    from mlflow.tracing.client import TracingClient

    client = MlflowClient()
    try:
        experiment = client.get_experiment(experiment_id)
        result = analyze(
            experiment_name=experiment.name,
            trace_count=trace_count,
        )

        issues_created = []
        for suggestion in result["suggestions"]:
            description = f"{suggestion['description']}\n\nRecommended action: {suggestion['action']}"

            issue = TracingClient()._create_issue(
                experiment_id=experiment_id,
                name=suggestion["title"],
                description=description,
                severity=_SEVERITY_MAP.get(suggestion["severity"], IssueSeverity.MEDIUM),
                categories=[f"[improve_{suggestion['type']}]", "[improve_suggestion]"],
                root_causes=[
                    f"Confidence: {suggestion['confidence']:.0%}",
                    f"Auto-applicable: {suggestion['auto_applicable']}",
                    f"Pattern: {suggestion['id']}",
                ],
                source_run_id=run_id,
                created_by="mlflow.improve",
            )
            issues_created.append(issue.issue_id)

        if run_id:
            client.set_terminated(run_id, RunStatus.to_string(RunStatus.FINISHED))

        return {
            "summary": result["summary"],
            "findings_count": len(result["findings"]),
            "suggestions_count": len(result["suggestions"]),
            "issues_created": issues_created,
        }
    except Exception:
        if run_id:
            client.set_terminated(run_id, RunStatus.to_string(RunStatus.FAILED))
        raise


@job(name="invoke_improve_fix", max_workers=2)
def invoke_improve_fix_job(
    issue_id: str,
    experiment_id: str,
    run_id: str | None = None,
    source: str = "manual",
    suggestion_title: str | None = None,
    suggestion_description: str | None = None,
    suggestion_action: str | None = None,
):
    """
    Job function to create a fix PR for a detected issue.

    Reads the issue details and experiment's GitHub repo connection,
    then uses the configured code agent to clone the repo, analyze
    the issue, and create a pull request with a fix.

    When called from the scheduler with suggestion_* fields, skips
    the Issue entity lookup and uses the suggestion details directly.
    """
    from mlflow.genai.improve.fix_agent_registry import FixRequest, get_agent

    import mlflow.genai.improve.fix_agents  # noqa: F401

    client = MlflowClient()
    try:
        if suggestion_title:
            issue_name = suggestion_title
            issue_description = suggestion_description or ""
            root_causes = [suggestion_action] if suggestion_action else []
        else:
            from mlflow.tracing.client import TracingClient
            issue = TracingClient()._get_issue(issue_id)
            issue_name = issue.name
            issue_description = issue.description
            root_causes = issue.root_causes or []

        experiment = client.get_experiment(experiment_id)

        exp_tags = experiment.tags or {}
        repo_url = exp_tags.get(MLFLOW_IMPROVE_GITHUB_REPO)
        if not repo_url:
            raise ValueError(
                f"No GitHub repo configured for experiment {experiment_id}. "
                f"Set the '{MLFLOW_IMPROVE_GITHUB_REPO}' experiment tag first."
            )
        branch = exp_tags.get(MLFLOW_IMPROVE_GITHUB_BRANCH, "main")
        agent_name = exp_tags.get(MLFLOW_IMPROVE_CODE_AGENT, "claude-code")

        agent = get_agent(agent_name)
        request = FixRequest(
            issue_id=issue_id,
            issue_name=issue_name,
            issue_description=issue_description,
            root_causes=root_causes,
            repo_url=repo_url,
            branch=branch,
            experiment_id=experiment_id,
        )

        fix_result = agent.create_fix(request)

        if fix_result.success and fix_result.pr_url:
            from mlflow.tracing.client import TracingClient
            store = TracingClient().store
            resolved_description = f"{issue_description}\n\n**Fix PR:** {fix_result.pr_url}"
            try:
                store.update_issue(
                    issue_id=issue_id,
                    status=IssueStatus.RESOLVED,
                    description=resolved_description,
                )
            except Exception:
                try:
                    store.create_issue(
                        experiment_id=experiment_id,
                        name=issue_name,
                        description=resolved_description,
                        status=IssueStatus.RESOLVED,
                        categories=["[improve_fix]"],
                        created_by="mlflow.improve.fix_agent",
                    )
                except Exception:
                    _logger.debug("Failed to record fix as Issue", exc_info=True)

        if run_id:
            status = RunStatus.FINISHED if fix_result.success else RunStatus.FAILED
            client.set_terminated(run_id, RunStatus.to_string(status))

        return {
            "success": fix_result.success,
            "pr_url": fix_result.pr_url,
            "pr_number": fix_result.pr_number,
            "error": fix_result.error,
            "changes_summary": fix_result.changes_summary,
        }
    except Exception:
        if run_id:
            client.set_terminated(run_id, RunStatus.to_string(RunStatus.FAILED))
        raise
