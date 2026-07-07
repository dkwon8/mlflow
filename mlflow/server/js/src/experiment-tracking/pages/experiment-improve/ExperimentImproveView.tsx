import { useState, useCallback } from 'react';
import { Button, Card, Input, SparkleIcon, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
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

interface AnalysisResult {
  findings: Finding[];
  suggestions: Suggestion[];
  summary: {
    status: string;
    traces_analyzed?: number;
    findings_count?: number;
    suggestions_count?: number;
    avg_tool_calls?: number;
    high_severity?: number;
    medium_severity?: number;
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

  const runAnalysis = useCallback(async () => {
    setIsAnalyzing(true);
    setError(null);
    try {
      const response = await fetch(getAjaxUrl('ajax-api/3.0/mlflow/improve/invoke'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ experiment_id: experimentId, trace_count: 20 }),
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
    async (issueId: string) => {
      setIsFixing(issueId);
      try {
        const response = await fetch(getAjaxUrl('ajax-api/3.0/mlflow/improve/fix'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ issue_id: issueId, experiment_id: experimentId }),
        });
        if (!response.ok) throw new Error(`Fix failed: ${response.statusText}`);
        const result = await response.json();
        if (result.pr_url) window.open(result.pr_url, '_blank');
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Fix failed');
      } finally {
        setIsFixing(null);
      }
    },
    [experimentId],
  );

  const saveGithubRepo = useCallback(async () => {
    try {
      await fetch(getAjaxUrl('ajax-api/2.0/mlflow/experiments/set-experiment-tag'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ experiment_id: experimentId, key: 'mlflow.improve.github_repo', value: githubRepo }),
      });
      setRepoSaved(true);
    } catch (e) {
      setError('Failed to save GitHub repo');
    }
  }, [experimentId, githubRepo]);

  const summary = analysisResult?.summary;
  const healthStatus =
    (summary?.high_severity ?? 0) > 0 ? 'critical' : (summary?.medium_severity ?? 0) > 0 ? 'warning' : 'healthy';

  return (
    <div css={{ padding: theme.spacing.lg, maxWidth: 900 }}>
      {/* Header */}
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: theme.spacing.lg }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <SparkleIcon />
          <Typography.Title level={3} css={{ margin: 0 }}>
            <FormattedMessage defaultMessage="Improve" description="Title for the improve page" />
          </Typography.Title>
        </div>
        <Button componentId="mlflow.improve.run-analysis" type="primary" loading={isAnalyzing} onClick={runAnalysis}>
          <FormattedMessage defaultMessage="Run Analysis" description="Button to run improve analysis" />
        </Button>
      </div>

      {/* Error */}
      {error && (
        <Card componentId="mlflow.improve.error-card" css={{ marginBottom: theme.spacing.md }}>
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
            defaultMessage="Connect a GitHub repository to enable automatic fix PRs."
            description="GitHub connection description"
          />
        </Typography.Text>
        <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
          <Input
            componentId="mlflow.improve.github-input"
            placeholder="owner/repo-name"
            value={githubRepo}
            onChange={(e) => {
              setGithubRepo(e.target.value);
              setRepoSaved(false);
            }}
            css={{ flex: 1 }}
          />
          <Button componentId="mlflow.improve.connect-repo" onClick={saveGithubRepo} disabled={!githubRepo || repoSaved}>
            {repoSaved ? 'Saved' : 'Connect'}
          </Button>
        </div>
      </Card>

      {/* Empty state */}
      {!analysisResult && !isAnalyzing && (
        <Card componentId="mlflow.improve.empty-state">
          <div css={{ textAlign: 'center', padding: theme.spacing.lg }}>
            <SparkleIcon css={{ fontSize: 32, marginBottom: theme.spacing.sm, color: theme.colors.actionDisabledText }} />
            <Typography.Text color="secondary" css={{ display: 'block' }}>
              <FormattedMessage
                defaultMessage="Click 'Run Analysis' to analyze your recent traces for performance issues, quality degradation, and optimization opportunities."
                description="Empty state message"
              />
            </Typography.Text>
          </div>
        </Card>
      )}

      {/* Health Summary */}
      {summary && (
        <div css={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: theme.spacing.sm, marginBottom: theme.spacing.lg }}>
          {[
            { label: 'Status', value: healthStatus === 'critical' ? 'Needs attention' : healthStatus === 'warning' ? 'Some issues' : 'Healthy' },
            { label: 'Traces Analyzed', value: summary.traces_analyzed ?? 0 },
            { label: 'Findings', value: summary.findings_count ?? 0 },
            { label: 'Avg Tool Calls', value: summary.avg_tool_calls ?? '—' },
          ].map((card) => (
            <Card componentId="mlflow.improve.summary-card" key={card.label}>
              <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm, textTransform: 'uppercase' }}>
                {card.label}
              </Typography.Text>
              <Typography.Title level={4} css={{ marginTop: 4 }}>
                {card.value}
              </Typography.Title>
            </Card>
          ))}
        </div>
      )}

      {/* Suggestions */}
      {analysisResult && analysisResult.suggestions.length > 0 && (
        <div css={{ marginBottom: theme.spacing.lg }}>
          <Typography.Title level={4} css={{ marginBottom: theme.spacing.sm }}>
            Suggestions ({analysisResult.suggestions.length})
          </Typography.Title>
          {analysisResult.suggestions.map((s) => (
            <Card componentId="mlflow.improve.suggestion-card" key={s.id} css={{ marginBottom: theme.spacing.sm }}>
              <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div css={{ flex: 1 }}>
                  <div css={{ display: 'flex', gap: theme.spacing.xs, marginBottom: theme.spacing.xs, alignItems: 'center' }}>
                    <Tag componentId="mlflow.improve.severity-tag" color={SEVERITY_COLORS[s.severity] || 'charcoal'}>
                      {s.severity}
                    </Tag>
                    <Tag componentId="mlflow.improve.type-tag" color="charcoal">
                      {TYPE_LABELS[s.type] || s.type}
                    </Tag>
                    <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                      {Math.round(s.confidence * 100)}% confidence
                    </Typography.Text>
                  </div>
                  <Typography.Title level={4} css={{ marginTop: theme.spacing.xs }}>
                    {s.title}
                  </Typography.Title>
                  <Typography.Text color="secondary">{s.description}</Typography.Text>
                  <Card componentId="mlflow.improve.action-card" css={{ marginTop: theme.spacing.sm }}>
                    <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm, textTransform: 'uppercase' }}>
                      Recommended action
                    </Typography.Text>
                    <Typography.Text css={{ display: 'block', marginTop: 4 }}>{s.action}</Typography.Text>
                  </Card>
                </div>
                {repoSaved && (
                  <Button
                    componentId="mlflow.improve.fix-button"
                    type="primary"
                    loading={isFixing === s.id}
                    onClick={() => triggerFix(s.id)}
                    css={{ marginLeft: theme.spacing.md }}
                  >
                    Fix it
                  </Button>
                )}
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Findings */}
      {analysisResult && analysisResult.findings.length > 0 && (
        <div>
          <Typography.Title level={4} css={{ marginBottom: theme.spacing.sm }}>
            Findings ({analysisResult.findings.length})
          </Typography.Title>
          {analysisResult.findings.map((f, i) => (
            <Card componentId="mlflow.improve.finding-card" key={i} css={{ marginBottom: theme.spacing.xs }}>
              <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'flex-start' }}>
                <Tag componentId="mlflow.improve.finding-severity" color={SEVERITY_COLORS[f.severity] || 'charcoal'}>
                  {f.severity}
                </Tag>
                <div>
                  <Typography.Text bold>{f.pattern.replace(/_/g, ' ')}</Typography.Text>
                  <Typography.Text color="secondary" css={{ display: 'block', marginTop: 2 }}>
                    {f.description}
                  </Typography.Text>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* No issues found */}
      {analysisResult && analysisResult.suggestions.length === 0 && analysisResult.findings.length === 0 && (
        <Card componentId="mlflow.improve.no-issues">
          <div css={{ textAlign: 'center', padding: theme.spacing.lg }}>
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="No issues detected. Your agent is performing within expected parameters."
                description="No issues found"
              />
            </Typography.Text>
          </div>
        </Card>
      )}
    </div>
  );
};
