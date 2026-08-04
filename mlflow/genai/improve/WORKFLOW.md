# MLflow Improve — Complete Workflow

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User's Agent                                 │
│  (OpenAI, LangChain, LlamaIndex, PydanticAI, any framework)       │
│                                                                     │
│  mlflow.openai.autolog()  ←── one line to enable tracing           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ traces flow automatically
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MLflow Tracking Server                          │
│                     (localhost:5001)                                 │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │ Experiment A │  │ Experiment B │  │ Experiment C │                │
│  │ "nps-agent"  │  │ "hr-agent"   │  │ "test-agent" │                │
│  │              │  │              │  │              │                │
│  │ Tags:        │  │ Tags:        │  │ Tags:        │                │
│  │ github_repo= │  │ github_repo= │  │ github_repo= │                │
│  │ Nehanth/     │  │ dkwon8/      │  │ dkwon8/      │                │
│  │ nps_agent    │  │ hr_agent     │  │ test-agent   │                │
│  │              │  │              │  │              │                │
│  │ Traces: 17   │  │ Traces: 200+ │  │ Traces: 25  │                │
│  └─────────────┘  └─────────────┘  └─────────────┘                │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Improve Module                             │   │
│  │                    mlflow/genai/improve/                      │   │
│  │                                                              │   │
│  │  Triggers:                                                   │   │
│  │  1. User clicks "Analyze"     → handlers.py → analyze()     │   │
│  │  2. Error trace arrives       → _error_hook.py → analyze()  │   │
│  │  3. Scheduler fires (10 min)  → scheduler.py → analyze()    │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Trigger 1: User Clicks "Analyze"

```
User clicks "Analyze" in MLflow UI
         │
         ▼
┌─ handlers.py ──────────────────────────────────────────────────┐
│  POST /ajax-api/3.0/mlflow/improve/invoke                      │
│                                                                 │
│  1. Read experiment_id from request                            │
│  2. Save repo_url as experiment tag:                           │
│     mlflow.improve.github_repo = "Nehanth/nps_agent"           │
│  3. Call analyze(experiment_name, repo_url, branch)            │
│  4. Create Issue entities for each suggestion                  │
│  5. Return results to UI                                       │
└────────────────────┬───────────────────────────────────────────┘
                     │
                     ▼
            analyze() — see "Analysis Flow" below
```

## Trigger 2: Error Trace Arrives

```
Agent throws error → ERROR trace logged to MLflow
         │
         ▼
┌─ _error_hook.py ───────────────────────────────────────────────┐
│  maybe_submit_error_analysis(experiment_id)                     │
│                                                                 │
│  Gate 1: Has 2 minutes passed since last analysis?             │
│          NO → skip (prevent spam)                              │
│          YES ↓                                                  │
│                                                                 │
│  Gate 2: Does this experiment have a GitHub repo connected?    │
│          NO → skip (can't analyze code without repo)           │
│          YES ↓                                                  │
│                                                                 │
│  Submit background job: invoke_improve_analysis_job()           │
│  (trace_count=10 — smaller window for fast reaction)           │
└────────────────────┬───────────────────────────────────────────┘
                     │
                     ▼
            analyze() — see "Analysis Flow" below
```

## Trigger 3: Scheduler (Every 10 Minutes)

```
Huey periodic task fires every 10 minutes
         │
         ▼
┌─ scheduler.py ─────────────────────────────────────────────────┐
│  run_improve_monitoring_scheduler()                             │
│                                                                 │
│  1. Find all experiments with github_repo tag set              │
│                                                                 │
│  For each experiment:                                          │
│  Gate 1: Is active_monitor == "true"?                          │
│          NO → skip                                             │
│                                                                 │
│  Gate 2: Has 10 min passed since last_monitor_time?            │
│          NO → skip                                             │
│          YES ↓                                                  │
│                                                                 │
│  2. Stamp last_monitor_time = now                              │
│  3. Call analyze(experiment_name)                              │
│  4. Compare current patterns vs last_patterns tag              │
│  5. Create Issues only for NEW patterns (dedup)                │
│  6. Save current patterns to last_patterns tag                 │
└────────────────────┬───────────────────────────────────────────┘
                     │
                     ▼
            analyze() — see "Analysis Flow" below
```

---

## Analysis Flow (the core)

All three triggers converge here:

```
analyze(experiment_name, repo_url, branch)
│
│  ┌─ Step 1: Find sibling experiments ──────────────────────┐
│  │                                                          │
│  │  Search all experiments where                           │
│  │  mlflow.improve.github_repo == repo_url                 │
│  │                                                          │
│  │  Example: nps-agent-dev AND nps-agent-prod both         │
│  │  point to Nehanth/nps_agent → pool their traces         │
│  └─────────────────────────────────────────────────────────┘
│
│  ┌─ Step 2: Fetch traces ──────────────────────────────────┐
│  │                                                          │
│  │  mlflow.search_traces(                                  │
│  │      experiment_ids=all_exp_ids,   ← pooled             │
│  │      max_results=20                                     │
│  │  )                                                       │
│  │                                                          │
│  │  Minimum 10 traces required, otherwise returns error    │
│  └──────────────────────────┬──────────────────────────────┘
│                              │
│         ┌────────────────────┼────────────────────┐
│         │                    │                    │
│         ▼                    ▼                    ▼
│  ┌─ Step 3A ─────┐   ┌─ Step 3B ──────┐   ┌─ Step 3C ─────────┐
│  │ Trace Analysis │   │ Fetch Code     │   │ Extract Hints      │
│  │                │   │                │   │                    │
│  │ trace_analyzer │   │ github_fetcher │   │ Tool names from    │
│  │ .py            │   │ .py            │   │ traces used to     │
│  │                │   │                │   │ prioritize files   │
│  │ 6 detectors:   │   │ GitHub API:    │   │                    │
│  │ • error_spike  │   │ 1 tree call    │   │ e.g. search_parks  │
│  │ • context_bloat│   │ +8-12 file     │   │ → nps_mcp_server   │
│  │ • tool_redund  │   │ fetches        │   │ gets +15 priority  │
│  │ • score_degrad │   │                │   │                    │
│  │ • slowdown     │   │ No git clone   │   └────────┬───────────┘
│  │ • incomplete   │   │ No temp files  │            │
│  │                │   │                │            │
│  │ Pure math:     │   └────────┬───────┘            │
│  │ z-scores,      │            │                    │
│  │ baselines      │            ▼                    │
│  │                │   ┌─ Step 3D ─────────────────────────────┐
│  │ No LLM cost    │   │ Code Analysis                         │
│  │                │   │                                       │
│  └────────┬───────┘   │ code_analyzer.py                     │
│           │           │                                       │
│           │           │ Sends files + trace findings to       │
│           │           │ GPT-5.4-mini with structured output   │
│           │           │                                       │
│           │           │ System prompt: "You are an expert     │
│           │           │ code reviewer. Find: anti_patterns,   │
│           │           │ config_issues, security, performance, │
│           │           │ missing_error_handling..."             │
│           │           │                                       │
│           │           │ Returns: list of CodeFinding objects   │
│           │           │ (file, problem, why, fix, confidence) │
│           │           └────────┬──────────────────────────────┘
│           │                    │
│           ▼                    ▼
│  ┌─ Step 4: Merge findings ─────────────────────────────────┐
│  │                                                           │
│  │  Trace findings   +   Code findings                      │
│  │  (statistical)        (LLM-generated)                    │
│  │                                                           │
│  │  Each categorized as:                                    │
│  │  • "heal" — errors, crashes, security (Fix tab)          │
│  │  • "improve" — optimizations, performance (Improve tab)  │
│  └──────────────────────────┬────────────────────────────────┘
│                              │
│                              ▼
│  ┌─ Step 5: Generate suggestions ────────────────────────────┐
│  │  suggestions.py                                            │
│  │                                                            │
│  │  Each finding → handler → Suggestion card                 │
│  │                                                            │
│  │  Trace findings: hardcoded human-readable text            │
│  │  Code findings: LLM's own analysis passed through         │
│  │                                                            │
│  │  Sorted by severity (high → medium → low)                 │
│  │  Stable IDs via hash (numbers stripped for dedup)          │
│  └──────────────────────────┬────────────────────────────────┘
│                              │
│                              ▼
│  ┌─ Step 6: Compute alerts ──────────────────────────────────┐
│  │  summary.py                                                │
│  │                                                            │
│  │  1. Get errors from last 5 traces                         │
│  │  2. Extract search terms from error messages:             │
│  │     "model 'gpt-4o-mnii' does not exist"                  │
│  │     → extracts "gpt-4o-mnii"                              │
│  │  3. Search for that term in fetched code files            │
│  │  4. If term ABSENT from code → error resolved, hide it   │
│  │  5. Deduplicate: one alert per unique error signature     │
│  └──────────────────────────┬────────────────────────────────┘
│                              │
│                              ▼
│  Return to UI:
│  {
│    findings: [...],         ← trace-based
│    code_findings: [...],    ← LLM-based
│    suggestions: [...],      ← UI cards
│    alerts: [...],           ← error alerts
│    summary: {...}           ← stats
│  }
```

---

## Fix Flow (User Clicks "Fix It")

```
User clicks "Fix it" on a suggestion card
         │
         ▼
┌─ handlers.py ──────────────────────────────────────────────────┐
│  POST /ajax-api/3.0/mlflow/improve/fix                         │
│                                                                 │
│  1. Read suggestion details from request                       │
│  2. Read repo_url from experiment tag                          │
│  3. Submit background job: invoke_improve_fix_job()            │
└────────────────────┬───────────────────────────────────────────┘
                     │
                     ▼
┌─ background_work.py ───────────────────────────────────────────┐
│  invoke_improve_fix_job()                                       │
│                                                                 │
│  1. Get suggestion details (from request or Issue entity)      │
│  2. Read experiment tags:                                      │
│     • repo_url (which GitHub repo)                             │
│     • branch (default "main")                                  │
│     • code_agent (default "claude-code")                       │
│  3. Build FixRequest with all context                          │
│  4. Call agent.create_fix(request)                             │
└────────────────────┬───────────────────────────────────────────┘
                     │
                     ▼
┌─ claude_code_agent.py ─────────────────────────────────────────┐
│  ClaudeCodeAgent.create_fix(request)                            │
│                                                                 │
│  Gate 1: Is 'claude' CLI installed?                            │
│          NO → return error "Install Claude Code CLI"           │
│                                                                 │
│  Gate 2: Is 'gh' CLI installed?                                │
│          NO → return error "Install GitHub CLI"                │
│                                                                 │
│  Step 1: Clone repo                                            │
│    clone_or_fetch_repo("Nehanth/nps_agent", "main")            │
│    Copy to temp dir (keep cache clean)                         │
│    git checkout -b improve/fix-s-24eb76c1                      │
│                                                                 │
│  Step 2: Build prompt                                          │
│    "Fix the following issue detected by MLflow's improve       │
│     system.                                                     │
│     Issue: HTTP requests do not set explicit timeouts...        │
│     Root causes: ...                                            │
│     Code analysis findings: ...                                 │
│     Analyze the codebase, find the source of this issue,       │
│     and fix it. Make minimal, targeted changes."               │
│                                                                 │
│  Step 3: Run Claude Code                                       │
│    Option A (SDK):                                             │
│      ClaudeSDKClient(cwd=repo_dir)                             │
│      → client.query(prompt)                                    │
│      → Claude reads files, edits, commits autonomously         │
│                                                                 │
│    Option B (CLI fallback):                                    │
│      subprocess: claude -p "prompt" --dangerously-skip-perms   │
│      → Same result, via command line                           │
│                                                                 │
│  Step 4: Push and create PR                                    │
│    Check: did Claude make any changes?                         │
│      NO → return error "No changes made"                       │
│      YES ↓                                                      │
│    git add -A (if uncommitted changes)                         │
│    git commit -m "fix: {issue_name}"                           │
│    git push origin improve/fix-s-24eb76c1                      │
│    gh pr create --title "[MLflow Improve] Fix: ..."            │
│                                                                 │
│  Return: FixResult(success=True, pr_url="https://...")         │
└────────────────────┬───────────────────────────────────────────┘
                     │
                     ▼
┌─ background_work.py (continued) ───────────────────────────────┐
│                                                                 │
│  If PR was created successfully:                               │
│    Update Issue entity → status = RESOLVED                     │
│    Append PR URL to Issue description                          │
│                                                                 │
│  Return result to UI:                                          │
│  { success: true, pr_url: "https://github.com/.../pull/14" }  │
│                                                                 │
│  UI removes suggestion from active list                        │
│  PR appears in Resolved tab                                    │
└────────────────────────────────────────────────────────────────┘
```

---

## Error Cases

| What goes wrong | Where it fails | What user sees |
|---|---|---|
| No GitHub repo entered | `analyze()` line 77 | "Connect a GitHub repository to use the improve feature" |
| Fewer than 10 traces | `analyze()` line 97 | "Need at least 10 traces (currently N)" |
| GitHub API rate limited | `github_fetcher.py` | "Dynamic code analysis failed" (falls back to trace-only) |
| Private repo, no GITHUB_TOKEN | `github_fetcher.py` | 404 error, analysis fails |
| Claude CLI not installed | `claude_code_agent.py` line 43 | "Claude Code CLI is required. Install: npm install -g ..." |
| gh CLI not installed | `claude_code_agent.py` line 48 | "GitHub CLI (gh) is required for PR creation" |
| No push access to repo | `claude_code_agent.py` line 207 | "Fix failed: git push returned non-zero exit status 128" |
| Claude makes no changes | `claude_code_agent.py` line 197 | "Agent ran but no changes were made to create a PR" |
| LLM code analysis fails | `code_analyzer.py` line 267 | Silently skipped, trace findings still shown |

---

## State Tracking (Experiment Tags)

All state is stored as experiment tags — no new database tables:

```
Experiment: "nps-agent" (id: 1)
├── mlflow.improve.github_repo        = "Nehanth/nps_agent"
├── mlflow.improve.github_branch      = "main"
├── mlflow.improve.code_agent         = "claude-code"
├── mlflow.improve.active_monitor     = "true"
├── mlflow.improve.last_monitor_time  = "2026-07-22T14:30:00+00:00"
├── mlflow.improve.last_patterns      = '["error_spike","context_bloat"]'
└── mlflow.improve.last_snapshot      = '{"traces_analyzed":17,...}'
```

---

## File Map

```
mlflow/genai/improve/
├── __init__.py              ← analyze(), compare(), snapshot()
├── trace_analyzer.py        ← 6 statistical detectors (z-scores)
├── code_analyzer.py         ← LLM code review (GPT-5.4-mini)
├── github_fetcher.py        ← GitHub REST API file fetching
├── suggestions.py           ← Finding → UI card conversion
├── summary.py               ← Error alerts + stat computation
├── scheduler.py             ← 10-min cron monitoring
├── background_work.py       ← Job queue (Analyze + Fix It)
├── _error_hook.py           ← Event-driven error detection
├── fix_agent_registry.py    ← Plugin interface for fix agents
├── utils.py                 ← URL normalization
├── fix_agents/
│   ├── __init__.py          ← Registers claude-code agent
│   └── claude_code_agent.py ← Clone → Claude edits → PR
├── WALKTHROUGH.md           ← Technical documentation
└── WORKFLOW.md              ← This file
```
