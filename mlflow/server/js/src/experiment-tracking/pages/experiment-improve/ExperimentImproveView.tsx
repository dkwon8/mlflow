import { useState, useCallback, useEffect } from 'react';
import { Button, Card, Input, SparkleIcon, Tabs, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { TagColors } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { getAjaxUrl } from '../../../common/utils/FetchUtils';

interface Suggestion {
  id: string;
  type: string;
  severity: string;
  category: string;
  title: string;
  description: string;
  action: string;
  confidence: number;
  auto_applicable: boolean;
}

interface Finding {
  pattern: string;
  severity: string;
  category: string;
  description: string;
}

interface CodeFinding {
  pattern: string;
  severity: string;
  description: string;
  file_path: string | null;
  root_cause: string | null;
  suggested_fix: string | null;
  confidence: number;
}

interface Alert {
  trace_id: string;
  error_message: string;
  user_query: string;
  failing_span: string;
  severity: string;
  timestamp?: string;
}

interface ResolvedFix {
  issue_id: string;
  title: string;
  pr_url: string;
  pr_number?: number;
  repo_url?: string;
  status?: 'merged' | 'open' | 'closed' | 'unknown';
  branch?: string;
  source?: 'auto' | 'manual';
}

interface AnalysisResult {
  findings: Finding[];
  code_findings?: CodeFinding[];
  suggestions: Suggestion[];
  alerts: Alert[];
  resolved_fixes?: ResolvedFix[];
  summary: {
    status: string;
    total_traces?: number;
    healthy_count?: number;
    error_count?: number;
    avg_latency_ms?: number;
    traces_analyzed?: number;
    traces_available?: number;
    traces_required?: number;
    findings_count?: number;
    code_findings_count?: number;
    suggestions_count?: number;
    avg_tool_calls?: number;
    high_severity?: number;
    medium_severity?: number;
    repo_analyzed?: boolean;
  };
}

const MIN_TRACES = 10;

const SEVERITY_COLORS: Record<string, TagColors> = {
  high: 'coral',
  medium: 'lemon',
  low: 'charcoal',
};

const TYPE_LABELS: Record<string, string> = {
  model_upgrade: 'Model',
  prompt_fix: 'Prompt',
  config_change: 'Config',
  investigate: 'Investigate',
};

export const ExperimentImproveView = ({ experimentId }: { experimentId: string }) => {
  const { theme } = useDesignSystemTheme();
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isFixing, setIsFixing] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [githubRepo, setGithubRepo] = useState('');
  const [repoSaved, setRepoSaved] = useState(false);
  const [repoSource, setRepoSource] = useState<'auto' | 'manual' | null>(null);
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  const [traceCount, setTraceCount] = useState<number | null>(null);

  const [lastFixBranch, setLastFixBranch] = useState<string | null>(null);
  const [lastFixPrUrl, setLastFixPrUrl] = useState<string | null>(null);
  const [feedback, setFeedback] = useState('');
  const [isSendingFeedback, setIsSendingFeedback] = useState(false);
  const [prStatusFixes, setPrStatusFixes] = useState<ResolvedFix[]>([]);
  const [isLoadingPrStatus, setIsLoadingPrStatus] = useState(false);

  const hasEnoughTraces = traceCount !== null && traceCount >= MIN_TRACES;
  const canAnalyze = repoSaved && hasEnoughTraces;

  useEffect(() => {
    const loadExperimentTags = async () => {
      try {
        const response = await fetch(
          getAjaxUrl(`ajax-api/2.0/mlflow/experiments/get?experiment_id=${experimentId}`),
        );
        if (response.ok) {
          const data = await response.json();
          const tags = data.experiment?.tags || [];
          const repoTag = tags.find((t: any) => t.key === 'mlflow.improve.github_repo');
          const sourceTag = tags.find((t: any) => t.key === 'mlflow.improve.github_repo_source');
          if (repoTag?.value) {
            setGithubRepo(repoTag.value);
            setRepoSaved(true);
            setRepoSource((sourceTag?.value as 'auto' | 'manual') || 'manual');
          }
        }
      } catch {
        // Non-critical
      }
    };

    const loadTraceCount = async () => {
      try {
        const response = await fetch(
          getAjaxUrl(`ajax-api/2.0/mlflow/traces?experiment_ids=${experimentId}&max_results=100`),
        );
        if (response.ok) {
          const data = await response.json();
          setTraceCount(data.traces?.length ?? 0);
        }
      } catch {
        setTraceCount(0);
      }
    };

    const loadPrStatus = async () => {
      setIsLoadingPrStatus(true);
      try {
        const response = await fetch(getAjaxUrl('ajax-api/3.0/mlflow/improve/pr-status'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ experiment_id: experimentId }),
        });
        if (response.ok) {
          const data = await response.json();
          setPrStatusFixes(data.resolved_fixes || []);
        }
      } catch {
        // Non-critical
      } finally {
        setIsLoadingPrStatus(false);
      }
    };

    loadExperimentTags();
    loadTraceCount();
    loadPrStatus();
  }, [experimentId]);

  const runAnalysis = useCallback(async () => {
    setIsAnalyzing(true);
    setError(null);
    setLastFixBranch(null);
    setLastFixPrUrl(null);
    try {
      const response = await fetch(getAjaxUrl('ajax-api/3.0/mlflow/improve/invoke'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          experiment_id: experimentId,
          trace_count: 20,
          mode: 'traces_only',
        }),
      });
      if (!response.ok) throw new Error(`Analysis failed: ${response.statusText}`);
      const result = await response.json();
      setAnalysisResult(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Analysis failed');
    } finally {
      setIsAnalyzing(false);
    }
  }, [experimentId]);

  const triggerFix = useCallback(
    async (suggestion: Suggestion | null, alert: Alert | null) => {
      const fixId = suggestion?.id || alert?.trace_id || '';
      setIsFixing(fixId);
      setError(null);
      try {
        const response = await fetch(getAjaxUrl('ajax-api/3.0/mlflow/improve/fix'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            issue_id: fixId,
            experiment_id: experimentId,
            suggestion: suggestion
              ? { title: suggestion.title, description: suggestion.description, action: suggestion.action }
              : undefined,
            code_findings: analysisResult?.code_findings || [],
            trace_id: alert?.trace_id,
            failing_span: alert?.failing_span,
            error_message: alert?.error_message,
          }),
        });
        if (!response.ok) throw new Error(`Fix failed: ${response.statusText}`);
        const result = await response.json();
        if (result.success && result.pr_url) {
          window.open(result.pr_url, '_blank');
          setLastFixPrUrl(result.pr_url);
          const branchMatch = result.pr_url.match(/improve\/[^/]+/);
          if (branchMatch) setLastFixBranch(branchMatch[0]);
          setAnalysisResult((prev) => {
            if (!prev) return prev;
            const newResolved: ResolvedFix = {
              issue_id: fixId,
              title: suggestion?.title || (alert ? `Runtime error in ${alert.failing_span}` : fixId),
              pr_url: result.pr_url,
              repo_url: githubRepo || undefined,
            };
            return {
              ...prev,
              resolved_fixes: [...(prev.resolved_fixes || []), newResolved],
              suggestions: suggestion ? prev.suggestions.filter((s) => s.id !== suggestion.id) : prev.suggestions,
              alerts: alert ? prev.alerts.filter((a) => a.trace_id !== alert.trace_id) : prev.alerts,
            };
          });
        } else if (result.error) {
          setError(`Fix failed: ${result.error}`);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Fix failed');
      } finally {
        setIsFixing(null);
      }
    },
    [experimentId, analysisResult, githubRepo],
  );

  const sendFeedback = useCallback(async () => {
    if (!feedback.trim() || !lastFixBranch) return;
    setIsSendingFeedback(true);
    setError(null);
    try {
      const response = await fetch(getAjaxUrl('ajax-api/3.0/mlflow/improve/feedback'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          experiment_id: experimentId,
          branch_name: lastFixBranch,
          feedback: feedback.trim(),
        }),
      });
      if (!response.ok) throw new Error(`Feedback failed: ${response.statusText}`);
      const result = await response.json();
      if (result.success) {
        setFeedback('');
      } else if (result.error) {
        setError(`Feedback failed: ${result.error}`);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Feedback failed');
    } finally {
      setIsSendingFeedback(false);
    }
  }, [experimentId, lastFixBranch, feedback]);

  const saveGithubRepo = useCallback(async () => {
    try {
      await fetch(getAjaxUrl('ajax-api/2.0/mlflow/experiments/set-experiment-tag'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ experiment_id: experimentId, key: 'mlflow.improve.github_repo', value: githubRepo }),
      });
      await fetch(getAjaxUrl('ajax-api/2.0/mlflow/experiments/set-experiment-tag'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ experiment_id: experimentId, key: 'mlflow.improve.github_repo_source', value: 'manual' }),
      });
      setRepoSaved(true);
      setRepoSource('manual');
      setAnalysisResult(null);
    } catch (e) {
      setError('Failed to save GitHub repo');
    }
  }, [experimentId, githubRepo]);


  const summary = analysisResult?.summary;
  const alerts = analysisResult?.alerts || [];
  const tagResolvedFixes = analysisResult?.resolved_fixes || [];
  const allResolvedFixes = (() => {
    const seen = new Set<string>();
    const merged: ResolvedFix[] = [];
    for (const fix of prStatusFixes) {
      const key = fix.pr_url || fix.issue_id;
      if (!seen.has(key)) { seen.add(key); merged.push(fix); }
    }
    for (const fix of tagResolvedFixes) {
      const key = fix.pr_url || fix.issue_id;
      if (!seen.has(key)) { seen.add(key); merged.push({ ...fix, status: fix.status || 'merged' }); }
    }
    return merged;
  })();
  const resolvedFixes = allResolvedFixes.filter((r) => !r.repo_url || r.repo_url === githubRepo);
  const resolvedTitles = new Set(resolvedFixes.map((r) => r.title));

  const activeSuggestions = analysisResult?.suggestions.filter((s) => !resolvedTitles.has(s.title)) || [];
  const uniqueSuggestions = [...new Map(activeSuggestions.map((s) => [s.title.replace(/\s*\(.*$/, ''), s])).values()];
  const healSuggestions = uniqueSuggestions.filter((s) => s.category === 'heal');
  const improveSuggestions = uniqueSuggestions.filter((s) => s.category === 'improve');

  const healCount = healSuggestions.length + alerts.length;
  const improveCount = improveSuggestions.length;
  const autoFixOpenPRs = resolvedFixes.filter((r) => r.source === 'auto' && r.status === 'open');

  return (
    <div css={{ padding: theme.spacing.lg, overflowY: 'auto', height: '100%' }}>
      {/* Header */}
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: theme.spacing.md }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <SparkleIcon />
          <Typography.Title level={3} css={{ margin: 0 }}>
            <FormattedMessage defaultMessage="Improve" description="Title for the improve page" />
          </Typography.Title>
        </div>
        {canAnalyze && (
          <Button componentId="mlflow.improve.run-analysis" type="primary" loading={isAnalyzing} onClick={runAnalysis}>
            <FormattedMessage defaultMessage="Analyze" description="Button to run improve analysis" />
          </Button>
        )}
      </div>

      {/* Error */}
      {error && (
        <Card componentId="mlflow.improve.error-card" css={{ marginBottom: theme.spacing.md, borderLeft: `3px solid ${theme.colors.textValidationDanger}` }}>
          <Typography.Text color="error">{error}</Typography.Text>
        </Card>
      )}

      {/* GitHub Connection */}
      <Card componentId="mlflow.improve.github-card" css={{ marginBottom: theme.spacing.md }}>
        <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm, fontSize: theme.typography.fontSizeSm }}>
          <FormattedMessage defaultMessage="GitHub Repository" description="GitHub connection label" />
        </Typography.Text>
        <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center', flexWrap: 'wrap' }}>
          <Input
            componentId="mlflow.improve.github-input"
            placeholder="owner/repo-name"
            value={githubRepo}
            onChange={(e) => { setGithubRepo(e.target.value); setRepoSaved(false); setAnalysisResult(null); }}
            css={{ width: 280, flexShrink: 0 }}
          />
          <Button componentId="mlflow.improve.connect-repo" onClick={saveGithubRepo} disabled={!githubRepo || repoSaved}>
            {repoSaved ? 'Saved' : 'Connect'}
          </Button>
          {repoSaved && (
            <Button
              componentId="mlflow.improve.view-repo"
              onClick={() => window.open(`https://github.com/${githubRepo}`, '_blank')}
            >
              View Repo
            </Button>
          )}
          {repoSaved && repoSource === 'auto' && (
            <Tag componentId="mlflow.improve.auto-detected" color="teal">Auto-detected</Tag>
          )}
          {repoSaved && traceCount !== null && (
            <Tag componentId="mlflow.improve.trace-count" color="charcoal">
              {traceCount} trace{traceCount !== 1 ? 's' : ''}
              {!hasEnoughTraces && ` · Need ${MIN_TRACES}+`}
            </Tag>
          )}
        </div>
      </Card>

      {/* Pre-analysis state */}
      {!analysisResult && !isAnalyzing && (
        <Card componentId="mlflow.improve.empty-state" css={{ width: '100%' }}>
          <div css={{ textAlign: 'center', padding: `${theme.spacing.xl}px ${theme.spacing.lg}px` }}>
            <SparkleIcon css={{ fontSize: 28, marginBottom: theme.spacing.sm, color: theme.colors.actionDisabledText }} />
            <Typography.Text color="secondary" css={{ display: 'block' }}>
              {!repoSaved
                ? 'Connect a GitHub repository above to get started.'
                : !hasEnoughTraces
                  ? `This experiment needs at least ${MIN_TRACES} traces before analysis can run. Currently has ${traceCount ?? 0}.`
                  : "Click 'Analyze' to scan this experiment's traces for issues and optimization opportunities."}
            </Typography.Text>
          </div>
        </Card>
      )}

      {/* Insufficient traces from backend */}
      {analysisResult?.summary?.status === 'insufficient_traces' && (
        <Card componentId="mlflow.improve.insufficient-traces" css={{ marginBottom: theme.spacing.md, width: '100%' }}>
          <div css={{ textAlign: 'center', padding: theme.spacing.lg }}>
            <Typography.Text color="warning">
              Not enough traces for analysis. Have {analysisResult.summary.traces_available}, need at least {analysisResult.summary.traces_required}. Run your agent more to generate traces.
            </Typography.Text>
          </div>
        </Card>
      )}

      {/* Auto-fix banner */}
      {autoFixOpenPRs.length > 0 && (
        <Card componentId="mlflow.improve.auto-fix-banner" css={{ marginBottom: theme.spacing.md, borderLeft: `3px solid ${theme.colors.textValidationSuccess}`, backgroundColor: 'rgba(34, 197, 94, 0.06)' }}>
          <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <Typography.Text bold css={{ display: 'block' }}>
                Self-Healing auto-created {autoFixOpenPRs.length} fix PR{autoFixOpenPRs.length > 1 ? 's' : ''}
              </Typography.Text>
              <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                Review and merge to apply.
              </Typography.Text>
            </div>
            <div css={{ display: 'flex', gap: theme.spacing.xs }}>
              {autoFixOpenPRs.map((pr) => (
                <Button key={pr.pr_url} componentId="mlflow.improve.auto-fix-pr" onClick={() => window.open(pr.pr_url, '_blank')}>
                  View PR
                </Button>
              ))}
            </div>
          </div>
        </Card>
      )}

      {/* Health Dashboard */}
      {summary && summary.status === 'ok' && (
        <>
          <div css={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: theme.spacing.md, marginBottom: theme.spacing.md }}>
            {[
              { label: 'Total Traces', value: summary.total_traces ?? summary.traces_analyzed ?? 0, color: theme.colors.textPrimary, bg: 'rgba(130, 140, 160, 0.08)' },
              { label: 'Healthy', value: summary.healthy_count ?? 0, color: theme.colors.textValidationSuccess, bg: 'rgba(34, 197, 94, 0.10)' },
              { label: 'With Issues', value: healCount + improveCount, color: theme.colors.textValidationWarning, bg: 'rgba(234, 179, 8, 0.10)' },
              { label: 'Errors', value: healCount, color: theme.colors.textValidationDanger, bg: 'rgba(239, 68, 68, 0.10)' },
              { label: 'Avg Latency', value: summary.avg_latency_ms ? `${(summary.avg_latency_ms / 1000).toFixed(1)}s` : '—', color: theme.colors.textPrimary, bg: 'rgba(99, 140, 210, 0.08)' },
            ].map((card) => (
              <Card componentId="mlflow.improve.health-card" key={card.label} css={{ textAlign: 'center', padding: theme.spacing.md, minWidth: 0, overflow: 'hidden', backgroundColor: card.bg, border: `1px solid ${card.bg}` }}>
                <Typography.Text color="secondary" css={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.5, display: 'block' }}>
                  {card.label}
                </Typography.Text>
                <Typography.Title level={2} css={{ marginTop: theme.spacing.xs, marginBottom: 0, color: card.color }}>
                  {card.value}
                </Typography.Title>
              </Card>
            ))}
          </div>

          <Tabs.Root componentId="mlflow.improve.sections" defaultValue="healing">
            <Tabs.List>
              <Tabs.Trigger value="healing">
                Self-Healing
                <Tag componentId="mlflow.improve.heal-count" color={healCount > 0 ? 'coral' : 'charcoal'} css={{ marginLeft: theme.spacing.xs }}>
                  {healCount}
                </Tag>
              </Tabs.Trigger>
              <Tabs.Trigger value="improvement">
                Self-Improvement
                <Tag componentId="mlflow.improve.improve-count" color={improveCount > 0 ? 'lemon' : 'charcoal'} css={{ marginLeft: theme.spacing.xs }}>
                  {improveCount}
                </Tag>
              </Tabs.Trigger>
              {(resolvedFixes.length > 0 || isLoadingPrStatus) && (
                <Tabs.Trigger value="resolved">
                  Resolved
                  <Tag componentId="mlflow.improve.resolved-count" color="teal" css={{ marginLeft: theme.spacing.xs }}>
                    {resolvedFixes.length}
                  </Tag>
                </Tabs.Trigger>
              )}
            </Tabs.List>

            {/* ── Self-Healing Tab ── */}
            <Tabs.Content value="healing">
              <div css={{ paddingTop: theme.spacing.md }}>
                <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm, fontSize: theme.typography.fontSizeSm }}>
                  Errors, failures, and broken tool calls that need fixing.
                </Typography.Text>

                {healCount > 0 ? (
                  <div css={{ display: 'grid', gridTemplateColumns: selectedAlert ? '1fr 1fr' : '1fr', gap: theme.spacing.md }}>
                    <div>
                      {alerts.map((alert, i) => (
                        <Card
                          componentId="mlflow.improve.alert-card"
                          key={`alert-${i}`}
                          css={{
                            marginBottom: theme.spacing.sm,
                            borderLeft: `3px solid ${theme.colors.textValidationDanger}`,
                            padding: theme.spacing.md,
                            cursor: 'pointer',
                            backgroundColor: selectedAlert?.trace_id === alert.trace_id ? theme.colors.actionTertiaryBackgroundPress : undefined,
                          }}
                          onClick={() => setSelectedAlert(selectedAlert?.trace_id === alert.trace_id ? null : alert)}
                        >
                          <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: theme.spacing.xs }}>
                            <div css={{ display: 'flex', gap: theme.spacing.xs, alignItems: 'center' }}>
                              <Tag componentId="mlflow.improve.alert-sev" color="coral">Error</Tag>
                              <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                                {alert.timestamp ? new Date(alert.timestamp).toLocaleString() : alert.trace_id.substring(0, 20) + '...'}
                              </Typography.Text>
                            </div>
                            <Button componentId="mlflow.improve.fix-alert-inline" type="primary" danger loading={isFixing === alert.trace_id} onClick={(e: React.MouseEvent) => { e.stopPropagation(); triggerFix(null, alert); }} css={{ flexShrink: 0 }}>
                              Fix it
                            </Button>
                          </div>
                          <Typography.Text bold css={{ display: 'block' }}>{alert.failing_span}</Typography.Text>
                          <Typography.Text color="secondary" css={{ display: 'block', marginTop: 2 }}>
                            {alert.error_message.length > 120 ? alert.error_message.substring(0, 120) + '...' : alert.error_message}
                          </Typography.Text>
                        </Card>
                      ))}

                      {healSuggestions.map((s) => (
                        <Card componentId="mlflow.improve.heal-suggestion" key={s.id} css={{ marginBottom: theme.spacing.sm, borderLeft: `3px solid ${theme.colors.textValidationDanger}`, padding: theme.spacing.md }}>
                          <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: theme.spacing.xs }}>
                            <div css={{ display: 'flex', gap: theme.spacing.xs, alignItems: 'center' }}>
                              <Tag componentId="mlflow.improve.sev-tag" color={SEVERITY_COLORS[s.severity] || 'charcoal'}>{s.severity}</Tag>
                              <Tag componentId="mlflow.improve.type-tag" color="charcoal">{TYPE_LABELS[s.type] || s.type}</Tag>
                              <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                                {Math.round(s.confidence * 100)}% confidence
                              </Typography.Text>
                            </div>
                            <Button componentId="mlflow.improve.fix-heal" type="primary" danger loading={isFixing === s.id} onClick={() => triggerFix(s, null)} css={{ flexShrink: 0 }}>
                              Fix it
                            </Button>
                          </div>
                          <Typography.Title level={4} css={{ marginTop: theme.spacing.xs }}>{s.title}</Typography.Title>
                          <Typography.Text color="secondary" css={{ display: 'block', marginTop: 2 }}>{s.description}</Typography.Text>
                          <div css={{ marginTop: theme.spacing.sm, padding: theme.spacing.sm, backgroundColor: theme.colors.backgroundSecondary, borderRadius: 4 }}>
                            <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm, textTransform: 'uppercase', display: 'block' }}>Recommended action</Typography.Text>
                            <Typography.Text css={{ display: 'block', marginTop: 4 }}>{s.action}</Typography.Text>
                          </div>
                        </Card>
                      ))}
                    </div>

                    {selectedAlert && (
                      <div>
                        <Typography.Title level={4} css={{ marginBottom: theme.spacing.sm }}>Error Detail</Typography.Title>
                        <Card componentId="mlflow.improve.alert-detail" css={{ marginBottom: theme.spacing.sm }}>
                          <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm, textTransform: 'uppercase', display: 'block', marginBottom: theme.spacing.xs }}>Error Message</Typography.Text>
                          <div css={{ backgroundColor: theme.colors.backgroundSecondary, padding: theme.spacing.sm, borderRadius: 4, fontFamily: 'monospace', fontSize: 12, overflowX: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                            {selectedAlert.error_message}
                          </div>
                        </Card>
                        {selectedAlert.user_query && (
                          <Card componentId="mlflow.improve.alert-query" css={{ marginBottom: theme.spacing.sm }}>
                            <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm, textTransform: 'uppercase', display: 'block', marginBottom: theme.spacing.xs }}>User Query</Typography.Text>
                            <Typography.Text>{selectedAlert.user_query}</Typography.Text>
                          </Card>
                        )}
                        <Card componentId="mlflow.improve.trace-ref" css={{ marginBottom: theme.spacing.sm }}>
                          <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm, textTransform: 'uppercase', display: 'block', marginBottom: theme.spacing.xs }}>Trace Reference</Typography.Text>
                          <Typography.Text css={{ display: 'block' }}>Trace ID: <code>{selectedAlert.trace_id}</code></Typography.Text>
                          <Typography.Text css={{ display: 'block', marginTop: 2 }}>Failing span: <code>{selectedAlert.failing_span}</code></Typography.Text>
                        </Card>
                        <Button componentId="mlflow.improve.fix-alert" type="primary" danger loading={isFixing === selectedAlert.trace_id} onClick={() => triggerFix(null, selectedAlert)} css={{ width: '100%' }}>
                          Fix It
                        </Button>
                      </div>
                    )}
                  </div>
                ) : (
                  <Card componentId="mlflow.improve.no-heal">
                    <div css={{ textAlign: 'center', padding: theme.spacing.md }}>
                      <Typography.Text color="secondary">No errors or failures detected.</Typography.Text>
                    </div>
                  </Card>
                )}
              </div>
            </Tabs.Content>

            {/* ── Self-Improvement Tab ── */}
            <Tabs.Content value="improvement">
              <div css={{ paddingTop: theme.spacing.md }}>
                <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm, fontSize: theme.typography.fontSizeSm }}>
                  Optimization opportunities: redundant calls, slow execution, declining scores.
                </Typography.Text>

                {improveCount > 0 ? (
                  <div>
                    {improveSuggestions.map((s) => (
                      <Card componentId="mlflow.improve.suggestion-card" key={s.id} css={{ marginBottom: theme.spacing.sm, padding: theme.spacing.md }}>
                        <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: theme.spacing.xs }}>
                          <div css={{ display: 'flex', gap: theme.spacing.xs, alignItems: 'center' }}>
                            <Tag componentId="mlflow.improve.sev-tag" color={SEVERITY_COLORS[s.severity] || 'charcoal'}>{s.severity}</Tag>
                            <Tag componentId="mlflow.improve.type-tag" color="charcoal">{TYPE_LABELS[s.type] || s.type}</Tag>
                            <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                              {Math.round(s.confidence * 100)}% confidence
                            </Typography.Text>
                          </div>
                          <Button componentId="mlflow.improve.fix-suggestion" type="primary" loading={isFixing === s.id} onClick={() => triggerFix(s, null)} css={{ flexShrink: 0 }}>
                            Fix it
                          </Button>
                        </div>
                        <Typography.Title level={4} css={{ marginTop: theme.spacing.xs }}>{s.title}</Typography.Title>
                        <Typography.Text color="secondary" css={{ display: 'block', marginTop: 2 }}>{s.description}</Typography.Text>
                        <div css={{ marginTop: theme.spacing.sm, padding: theme.spacing.sm, backgroundColor: theme.colors.backgroundSecondary, borderRadius: 4 }}>
                          <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm, textTransform: 'uppercase', display: 'block' }}>Recommended action</Typography.Text>
                          <Typography.Text css={{ display: 'block', marginTop: 4 }}>{s.action}</Typography.Text>
                        </div>
                      </Card>
                    ))}
                  </div>
                ) : (
                  <Card componentId="mlflow.improve.no-improve">
                    <div css={{ textAlign: 'center', padding: theme.spacing.md }}>
                      <Typography.Text color="secondary">No optimization opportunities detected. Your agent is performing well.</Typography.Text>
                    </div>
                  </Card>
                )}
              </div>
            </Tabs.Content>

            {/* ── Resolved Tab ── */}
            {(resolvedFixes.length > 0 || isLoadingPrStatus) && (
              <Tabs.Content value="resolved">
                <div css={{ paddingTop: theme.spacing.md }}>
                  {isLoadingPrStatus && resolvedFixes.length === 0 && (
                    <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm, fontSize: theme.typography.fontSizeSm }}>
                      Checking PR status...
                    </Typography.Text>
                  )}
                  {resolvedFixes.map((r, i) => {
                    const statusColor: TagColors =
                      r.status === 'merged' ? 'teal' :
                      r.status === 'open' ? 'lemon' :
                      r.status === 'closed' ? 'charcoal' :
                      'teal';
                    const statusLabel =
                      r.status === 'merged' ? 'Merged' :
                      r.status === 'open' ? 'Open' :
                      r.status === 'closed' ? 'Closed' :
                      'Fixed';
                    return (
                      <Card componentId="mlflow.improve.resolved-card" key={r.pr_url || i} css={{ marginBottom: theme.spacing.sm }}>
                        <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <div css={{ display: 'flex', gap: theme.spacing.xs, alignItems: 'center' }}>
                            <Tag componentId="mlflow.improve.resolved-tag" color={statusColor}>{statusLabel}</Tag>
                            <Typography.Text>{r.title}</Typography.Text>
                          </div>
                          {r.pr_url && (
                            <Button componentId="mlflow.improve.view-pr" onClick={() => window.open(r.pr_url, '_blank')}>
                              View PR
                            </Button>
                          )}
                        </div>
                      </Card>
                    );
                  })}
                </div>
              </Tabs.Content>
            )}
          </Tabs.Root>
        </>
      )}

      {/* PR result + Feedback */}
      {lastFixPrUrl && (
        <Card componentId="mlflow.improve.pr-result" css={{ marginTop: theme.spacing.lg }}>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, marginBottom: theme.spacing.sm }}>
            <Tag componentId="mlflow.improve.pr-tag" color="teal">PR Created</Tag>
            <Typography.Text>
              <a href={lastFixPrUrl} target="_blank" rel="noopener noreferrer">{lastFixPrUrl}</a>
            </Typography.Text>
          </div>
          <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm, fontSize: theme.typography.fontSizeSm }}>
            Not satisfied? Describe what to adjust and the agent will make additional commits on the same branch.
          </Typography.Text>
          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
            <Input
              componentId="mlflow.improve.feedback-input"
              placeholder="e.g., don't change the model, just fix the prompt"
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              onKeyDown={(e: React.KeyboardEvent) => { if (e.key === 'Enter') sendFeedback(); }}
              css={{ flex: 1 }}
              disabled={isSendingFeedback}
            />
            <Button componentId="mlflow.improve.send-feedback" type="primary" loading={isSendingFeedback} onClick={sendFeedback} disabled={!feedback.trim()}>
              Send Feedback
            </Button>
          </div>
        </Card>
      )}
    </div>
  );
};
