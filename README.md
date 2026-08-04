# MLflow Fork — Self-Optimization for AI Agents

This is a fork of [MLflow](https://github.com/mlflow/mlflow) that adds **Improve** — a continuous diagnostics system that detects problems in any MLflow-traced agent and suggests fixes, including automated GitHub PRs.

Any agent framework that logs traces to MLflow (OpenAI Agents SDK, LangChain, LlamaIndex, custom) gets automatic anomaly detection, code analysis, and self-healing — with zero agent-specific configuration.

## What Improve Does

1. **Detects anomalies in traces** using 6 z-score statistical detectors:
   - Context bloat and growth (traces getting larger over time)
   - Tool redundancy (same tool called multiple times)
   - Score degradation (evaluation quality dropping)
   - Execution slowdown (latency increasing)
   - Error spikes (failure rate rising)

2. **Analyzes your agent's source code** by fetching files from the connected GitHub repo and running LLM-based code review (GPT-5.4-mini), looking for anti-patterns, prompt quality issues, missing error handling, security risks, and performance problems.

3. **Creates fix PRs automatically** — when you click "Fix It" on a suggestion, a Claude Code agent clones your repo, applies the fix, and opens a GitHub pull request.

## Three Trigger Modes

| Mode | How it works |
|------|-------------|
| **Manual** | Click "Analyze" in the Improve tab |
| **Event-driven** | Automatically triggered when an ERROR trace arrives (with 120s cooldown) |
| **Scheduled** | Cron job runs every 10 minutes on experiments with monitoring enabled |

## Improve Tab in MLflow UI

The fork adds an **Improve** tab to each experiment page with three views:

- **Fix** — Error alerts from traces with root cause analysis and "Fix It" buttons
- **Improve** — Optimization suggestions with severity, confidence, and actionable recommendations
- **Resolved** — Previously fixed issues with links to the generated PRs

Plus stat cards showing traces analyzed, healthy count, and average latency.

## Quick Start

### 1. Run the forked MLflow server

```bash
git clone https://github.com/dkwon8/mlflow.git
cd mlflow
git checkout improve/dynamic-code-analyzer

pip install -e ".[genai]"
mlflow server --port 5001
```

### 2. Log traces from your agent

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("my-agent")

# If using OpenAI Agents SDK:
mlflow.openai.autolog()

# Run your agent — traces are logged automatically
```

### 3. Connect a GitHub repo

In the MLflow UI, go to your experiment's **Improve** tab and enter your GitHub repo URL (e.g., `owner/repo`). The system also auto-detects repos from git metadata in traces.

### 4. Analyze

Click **Analyze** to run the full diagnostic pipeline, or enable monitoring for automatic scheduled analysis.

### Dependencies for Fix Agent

The "Fix It" feature requires:
- `claude` CLI ([Claude Code](https://docs.anthropic.com/en/docs/claude-code))
- `gh` CLI ([GitHub CLI](https://cli.github.com/))
- `GITHUB_TOKEN` environment variable or `gh auth login`

## Architecture

The Improve module lives at `mlflow/genai/improve/`. For detailed documentation:

- [WALKTHROUGH.md](mlflow/genai/improve/WALKTHROUGH.md) — Technical deep-dive: data flow, z-score math, file selection scoring, experiment tags, dependencies
- [WORKFLOW.md](mlflow/genai/improve/WORKFLOW.md) — Architecture diagrams: trigger paths, analysis flow, fix flow, error cases

### Key Files

```
mlflow/genai/improve/
├── __init__.py            # Entry point: analyze(), compare(), snapshot()
├── trace_analyzer.py      # 6 z-score statistical detectors
├── code_analyzer.py       # LLM-powered code review
├── github_fetcher.py      # GitHub REST API file fetching
├── suggestions.py         # Finding-to-suggestion mapping
├── summary.py             # Health stats and error alerts
├── scheduler.py           # Cron periodic monitoring
├── background_work.py     # Huey async job wrappers
├── fix_agent_registry.py  # Abstract fix agent interface
├── fix_agents/            # Claude Code agent for PR creation
├── _error_hook.py         # Event-driven error analysis trigger
└── utils.py               # URL normalization

mlflow/server/handlers.py  # 4 HTTP endpoints (/improve/invoke, /fix, /feedback, /pr-status)

mlflow/server/js/src/experiment-tracking/pages/experiment-improve/
├── ExperimentImprovePage.tsx   # Page wrapper
└── ExperimentImproveView.tsx   # Full React UI (tabs, stat cards, suggestion cards)
```

## Related

The [HR Recruitment Agent](https://github.com/dkwon8/hr_agent) serves as the demo and test case for this feature — an OpenAI Agents SDK agent with 5 MCP tool servers that logs traces to this forked MLflow.

## Upstream MLflow

For standard MLflow documentation, see the [upstream repository](https://github.com/mlflow/mlflow).
