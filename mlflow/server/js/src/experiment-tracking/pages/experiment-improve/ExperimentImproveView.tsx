import { useState, useCallback, useEffect } from 'react';
import { Button, Card, Input, SparkleIcon, Tabs, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { TagColors } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { getAjaxUrl } from '../../../common/utils/FetchUtils';

interface Suggestion {
  id: string;
  type: string;
  severity: string;
  title: string;
  description: string;
  action: string;
  confidence: number;
  auto_applicable: boolean;
}

interface Finding {
  pattern: string;
  severity: string;
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
  repo_url?: string;
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
    findings_count?: number;
    code_findings_count?: number;
    suggestions_count?: number;
    avg_tool_calls?: number;
    high_severity?: number;
    medium_severity?: number;
    repo_analyzed?: boolean;
  };
}

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
        // Non-critical — tags will just start empty
      }
    };
    loadExperimentTags();
  }, [experimentId]);

  const runAnalysis = useCallback(async () => {
    setIsAnalyzing(true);
    setError(null);
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
  const allResolvedFixes = analysisResult?.resolved_fixes || [];
  const resolvedFixes = allResolvedFixes.filter((r) => !r.repo_url || r.repo_url === githubRepo);
  const resolvedTitles = new Set(resolvedFixes.map((r) => r.title));

  const activeSuggestions = analysisResult?.suggestions.filter((s) => !resolvedTitles.has(s.title)) || [];

  return (
    <div css={{ padding: theme.spacing.lg, overflowY: 'auto', height: '100%' }}>
      {/* Header */}
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: theme.spacing.lg }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <SparkleIcon />
          <Typography.Title level={3} css={{ margin: 0 }}>
            <FormattedMessage defaultMessage="Improve" description="Title for the improve page" />
          </Typography.Title>
        </div>
        <Button componentId="mlflow.improve.run-analysis" type="primary" loading={isAnalyzing} onClick={runAnalysis}>
          <FormattedMessage defaultMessage="Refresh" description="Button to refresh improve analysis" />
        </Button>
      </div>

      {/* Error */}
      {error && (
        <Card componentId="mlflow.improve.error-card" css={{ marginBottom: theme.spacing.md, borderLeft: `3px solid ${theme.colors.textValidationDanger}` }}>
          <Typography.Text color="error">{error}</Typography.Text>
        </Card>
      )}

      {/* GitHub Connection */}
      <Card componentId="mlflow.improve.github-card" css={{ marginBottom: theme.spacing.lg }}>
        <Typography.Title level={4} css={{ marginBottom: theme.spacing.sm }}>
          <FormattedMessage defaultMessage="GitHub Repository" description="GitHub connection section title" />
        </Typography.Title>
        <Typography.Text color="secondary" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
          <FormattedMessage
            defaultMessage="Connect a GitHub repository to enable code analysis and automatic fix PRs."
            description="GitHub connection description"
          />
        </Typography.Text>
        <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
          <Input
            componentId="mlflow.improve.github-input"
            placeholder="owner/repo-name"
            value={githubRepo}
            onChange={(e) => { setGithubRepo(e.target.value); setRepoSaved(false); setAnalysisResult(null); }}
            css={{ flex: 1 }}
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
            <Tag componentId="mlflow.improve.auto-detected" color="teal">
              Auto-detected from traces
            </Tag>
          )}
        </div>
      </Card>

      {/* Empty state */}
      {!analysisResult && !isAnalyzing && (
        <Card componentId="mlflow.improve.empty-state">
          <div css={{ textAlign: 'center', padding: theme.spacing.lg }}>
            <SparkleIcon css={{ fontSize: 32, marginBottom: theme.spacing.sm, color: theme.colors.actionDisabledText }} />
            <Typography.Text color="secondary" css={{ display: 'block' }}>
              {repoSaved ? (
                "Click 'Run Analysis' to analyze your repository code and traces. Use 'Code Only' mode to analyze code without traces."
              ) : (
                "Connect a GitHub repository above, then click 'Run Analysis' to find issues in your code and traces."
              )}
            </Typography.Text>
          </div>
        </Card>
      )}

      {/* Health Dashboard */}
      {summary && (
        <>
          <div css={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: theme.spacing.sm, marginBottom: theme.spacing.sm }}>
            {[
              { label: 'Total Traces', value: summary.total_traces ?? summary.traces_analyzed ?? 0, color: theme.colors.textPrimary },
              { label: 'Healthy', value: summary.healthy_count ?? 0, color: theme.colors.textValidationSuccess },
              { label: 'With Issues', value: summary.findings_count ?? 0, color: theme.colors.textValidationWarning },
              { label: 'Errors', value: summary.error_count ?? 0, color: theme.colors.textValidationDanger },
              { label: 'Avg Latency', value: summary.avg_latency_ms ? `${(summary.avg_latency_ms / 1000).toFixed(1)}s` : '—', color: theme.colors.textPrimary },
            ].map((card) => (
              <Card componentId="mlflow.improve.health-card" key={card.label} css={{ minHeight: 80 }}>
                <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm, textTransform: 'uppercase', display: 'block' }}>
                  {card.label}
                </Typography.Text>
                <Typography.Title level={3} css={{ marginTop: 4, marginBottom: 0, color: card.color }}>
                  {card.value}
                </Typography.Title>
              </Card>
            ))}
          </div>
        </>
      )}

      {/* Tabs: Issues | Resolved */}
      {analysisResult && (
        <Tabs.Root componentId="mlflow.improve.tabs" defaultValue="issues" valueHasNoPii>
          <Tabs.List>
            <Tabs.Trigger value="issues">
              Issues ({activeSuggestions.length + alerts.length})
            </Tabs.Trigger>
            <Tabs.Trigger value="resolved">
              Resolved ({resolvedFixes.length})
            </Tabs.Trigger>
          </Tabs.List>

          {/* Issues Tab — errors + optimization suggestions unified */}
          <Tabs.Content value="issues">
            <div css={{ paddingTop: theme.spacing.md }}>
              {(activeSuggestions.length + alerts.length) > 0 ? (
                <div css={{ display: 'grid', gridTemplateColumns: selectedAlert ? '1fr 1fr' : '1fr', gap: theme.spacing.md }}>
                  <div>
                    {/* Error alerts */}
                    {alerts.map((alert, i) => (
                      <Card
                        componentId="mlflow.improve.alert-card"
                        key={`alert-${i}`}
                        css={{
                          marginBottom: theme.spacing.sm,
                          borderLeft: `3px solid ${theme.colors.textValidationDanger}`,
                          cursor: 'pointer',
                          backgroundColor: selectedAlert?.trace_id === alert.trace_id ? theme.colors.actionTertiaryBackgroundPress : undefined,
                        }}
                        onClick={() => setSelectedAlert(selectedAlert?.trace_id === alert.trace_id ? null : alert)}
                      >
                        <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                          <div css={{ flex: 1 }}>
                            <div css={{ display: 'flex', gap: theme.spacing.xs, alignItems: 'center', marginBottom: theme.spacing.xs }}>
                              <Tag componentId="mlflow.improve.alert-sev" color="coral">Error</Tag>
                              <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                                {alert.timestamp ? new Date(alert.timestamp).toLocaleString() : alert.trace_id.substring(0, 20) + '...'}
                              </Typography.Text>
                            </div>
                            <Typography.Text bold css={{ display: 'block' }}>{alert.failing_span}</Typography.Text>
                            <Typography.Text color="secondary" css={{ display: 'block', marginTop: 2 }}>
                              {alert.error_message.length > 120 ? alert.error_message.substring(0, 120) + '...' : alert.error_message}
                            </Typography.Text>
                            {alert.user_query && (
                              <Typography.Text color="secondary" css={{ display: 'block', marginTop: theme.spacing.xs, fontSize: theme.typography.fontSizeSm }}>
                                Query: {alert.user_query.length > 80 ? alert.user_query.substring(0, 80) + '...' : alert.user_query}
                              </Typography.Text>
                            )}
                          </div>
                        </div>
                      </Card>
                    ))}

                    {/* Optimization suggestions */}
                    {activeSuggestions.map((s) => (
                      <Card componentId="mlflow.improve.suggestion-card" key={s.id} css={{ marginBottom: theme.spacing.sm }}>
                        <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: theme.spacing.xs }}>
                          <div css={{ display: 'flex', gap: theme.spacing.xs, alignItems: 'center' }}>
                            <Tag componentId="mlflow.improve.sev-tag" color={SEVERITY_COLORS[s.severity] || 'charcoal'}>{s.severity}</Tag>
                            <Tag componentId="mlflow.improve.type-tag" color="charcoal">{TYPE_LABELS[s.type] || s.type}</Tag>
                            <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                              {Math.round(s.confidence * 100)}% confidence
                            </Typography.Text>
                          </div>
                          {repoSaved && (
                            <Button
                              componentId="mlflow.improve.fix-suggestion"
                              type="primary"
                              loading={isFixing === s.id}
                              onClick={() => triggerFix(s, null)}
                              css={{ flexShrink: 0 }}
                            >
                              Fix it
                            </Button>
                          )}
                        </div>
                        <Typography.Title level={4} css={{ marginTop: theme.spacing.xs }}>{s.title}</Typography.Title>
                        <Typography.Text color="secondary" css={{ display: 'block', marginTop: 2 }}>{s.description}</Typography.Text>
                        <div css={{ marginTop: theme.spacing.sm, padding: theme.spacing.sm, backgroundColor: theme.colors.backgroundSecondary, borderRadius: 4 }}>
                          <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm, textTransform: 'uppercase', display: 'block' }}>
                            Recommended action
                          </Typography.Text>
                          <Typography.Text css={{ display: 'block', marginTop: 4 }}>{s.action}</Typography.Text>
                        </div>
                      </Card>
                    ))}
                  </div>

                  {/* Alert detail panel */}
                  {selectedAlert && (
                    <div>
                      <Typography.Title level={4} css={{ marginBottom: theme.spacing.sm }}>Error Detail</Typography.Title>

                      <Card componentId="mlflow.improve.alert-detail" css={{ marginBottom: theme.spacing.sm }}>
                        <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm, textTransform: 'uppercase', display: 'block', marginBottom: theme.spacing.xs }}>
                          Error Message
                        </Typography.Text>
                        <div css={{
                          backgroundColor: theme.colors.backgroundSecondary,
                          padding: theme.spacing.sm,
                          borderRadius: 4,
                          fontFamily: 'monospace',
                          fontSize: 12,
                          overflowX: 'auto',
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                        }}>
                          {selectedAlert.error_message}
                        </div>
                      </Card>

                      {selectedAlert.user_query && (
                        <Card componentId="mlflow.improve.alert-query" css={{ marginBottom: theme.spacing.sm }}>
                          <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm, textTransform: 'uppercase', display: 'block', marginBottom: theme.spacing.xs }}>
                            User Query
                          </Typography.Text>
                          <Typography.Text>{selectedAlert.user_query}</Typography.Text>
                        </Card>
                      )}

                      {repoSaved && (
                        <Card componentId="mlflow.improve.linked-agent" css={{ marginBottom: theme.spacing.sm }}>
                          <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm, textTransform: 'uppercase', display: 'block', marginBottom: theme.spacing.xs }}>
                            Linked Agent
                          </Typography.Text>
                          <Typography.Text bold css={{ display: 'block' }}>{githubRepo}</Typography.Text>
                          <Button
                            componentId="mlflow.improve.view-repo-detail"
                            css={{ marginTop: theme.spacing.sm }}
                            onClick={() => window.open(`https://github.com/${githubRepo}`, '_blank')}
                          >
                            View Repo
                          </Button>
                        </Card>
                      )}

                      <Card componentId="mlflow.improve.trace-ref" css={{ marginBottom: theme.spacing.sm }}>
                        <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm, textTransform: 'uppercase', display: 'block', marginBottom: theme.spacing.xs }}>
                          Trace Reference
                        </Typography.Text>
                        <Typography.Text css={{ display: 'block' }}>Trace ID: <code>{selectedAlert.trace_id}</code></Typography.Text>
                        <Typography.Text css={{ display: 'block', marginTop: 2 }}>Failing span: <code>{selectedAlert.failing_span}</code></Typography.Text>
                      </Card>

                      {repoSaved && (
                        <Button
                          componentId="mlflow.improve.fix-alert"
                          type="primary"
                          danger
                          loading={isFixing === selectedAlert.trace_id}
                          onClick={() => triggerFix(null, selectedAlert)}
                          css={{ width: '100%' }}
                        >
                          Fix It
                        </Button>
                      )}
                    </div>
                  )}
                </div>
              ) : (
                <Card componentId="mlflow.improve.no-issues">
                  <div css={{ textAlign: 'center', padding: theme.spacing.lg }}>
                    <Typography.Text color="secondary">
                      No issues detected. Your agent is performing well.
                    </Typography.Text>
                  </div>
                </Card>
              )}
            </div>
          </Tabs.Content>

          {/* Resolved Tab */}
          <Tabs.Content value="resolved">
            <div css={{ paddingTop: theme.spacing.md }}>
              {resolvedFixes.length > 0 ? (
                <>
                  {resolvedFixes.map((r, i) => (
                    <Card componentId="mlflow.improve.resolved-card" key={i} css={{ marginBottom: theme.spacing.sm }}>
                      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div css={{ display: 'flex', gap: theme.spacing.xs, alignItems: 'center' }}>
                          <Tag componentId="mlflow.improve.resolved-tag" color="teal">Fixed</Tag>
                          <Typography.Text>{r.title}</Typography.Text>
                        </div>
                        {r.pr_url && (
                          <Button
                            componentId="mlflow.improve.view-pr"
                            onClick={() => window.open(r.pr_url, '_blank')}
                          >
                            View PR
                          </Button>
                        )}
                      </div>
                    </Card>
                  ))}
                </>
              ) : (
                <Card componentId="mlflow.improve.no-resolved">
                  <div css={{ textAlign: 'center', padding: theme.spacing.lg }}>
                    <Typography.Text color="secondary">
                      No resolved issues yet. Use "Fix it" on suggestions to create fix PRs.
                    </Typography.Text>
                  </div>
                </Card>
              )}
            </div>
          </Tabs.Content>

        </Tabs.Root>
      )}
    </div>
  );
};
