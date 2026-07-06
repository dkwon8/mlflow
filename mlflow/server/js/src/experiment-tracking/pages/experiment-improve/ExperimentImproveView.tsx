import { useState, useCallback } from 'react';
import {
  Button,
  Card,
  Input,
  SparkleIcon,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { TagColors } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { getAjaxUrl } from '../../../common/utils/FetchUtils';

const { Title, Text, Paragraph } = Typography;

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

interface Summary {
  status: string;
  traces_analyzed?: number;
  findings_count?: number;
  suggestions_count?: number;
  avg_tool_calls?: number;
  high_severity?: number;
  medium_severity?: number;
}

interface AnalysisResult {
  findings: Finding[];
  suggestions: Suggestion[];
  summary: Summary;
}

const SEVERITY_COLORS: Record<string, TagColors> = {
  high: 'red',
  medium: 'lemon',
  low: 'turquoise',
};

const TYPE_LABELS: Record<string, string> = {
  model_upgrade: 'Model',
  prompt_fix: 'Prompt',
  config_change: 'Config',
  investigate: 'Investigate',
};

interface ExperimentImproveViewProps {
  experimentId: string;
}

export const ExperimentImproveView = ({ experimentId }: ExperimentImproveViewProps) => {
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
      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }
      const jobResult = await response.json();

      // Poll for job completion, then fetch results directly
      // For now, run analysis synchronously via the Python API
      const directResponse = await fetch(getAjaxUrl('ajax-api/3.0/mlflow/improve/invoke'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ experiment_id: experimentId, trace_count: 20 }),
      });
      const result = await directResponse.json();
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
        if (!response.ok) {
          throw new Error(`Fix failed: ${response.statusText}`);
        }
        const result = await response.json();
        if (result.pr_url) {
          window.open(result.pr_url, '_blank');
        }
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
        body: JSON.stringify({
          experiment_id: experimentId,
          key: 'mlflow.improve.github_repo',
          value: githubRepo,
        }),
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
    <div style={{ padding: theme.spacing.lg, maxWidth: 900 }}>
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: theme.spacing.lg,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <SparkleIcon />
          <Title level={3}>
            <FormattedMessage defaultMessage="Improve" description="Title for the improve page" />
          </Title>
        </div>
        <Button type="primary" loading={isAnalyzing} onClick={runAnalysis}>
          <FormattedMessage defaultMessage="Run Analysis" description="Button to run improve analysis" />
        </Button>
      </div>

      {/* Error message */}
      {error && (
        <Card style={{ marginBottom: theme.spacing.md, borderColor: theme.colors.red }}>
          <Text style={{ color: theme.colors.red }}>{error}</Text>
        </Card>
      )}

      {/* GitHub Connection */}
      <Card style={{ marginBottom: theme.spacing.lg }}>
        <Title level={4} style={{ marginBottom: theme.spacing.sm }}>
          <FormattedMessage defaultMessage="GitHub Repository" description="GitHub connection section title" />
        </Title>
        <Text type="secondary" style={{ display: 'block', marginBottom: theme.spacing.sm }}>
          <FormattedMessage
            defaultMessage="Connect a GitHub repository to enable automatic fix PRs."
            description="GitHub connection description"
          />
        </Text>
        <div style={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
          <Input
            placeholder="owner/repo-name"
            value={githubRepo}
            onChange={(e) => {
              setGithubRepo(e.target.value);
              setRepoSaved(false);
            }}
            style={{ flex: 1 }}
          />
          <Button onClick={saveGithubRepo} disabled={!githubRepo || repoSaved}>
            {repoSaved ? (
              <FormattedMessage defaultMessage="Saved" description="Repo saved confirmation" />
            ) : (
              <FormattedMessage defaultMessage="Connect" description="Connect repo button" />
            )}
          </Button>
        </div>
      </Card>

      {/* No analysis yet */}
      {!analysisResult && !isAnalyzing && (
        <Card>
          <div style={{ textAlign: 'center', padding: theme.spacing.lg }}>
            <SparkleIcon style={{ fontSize: 32, marginBottom: theme.spacing.sm, color: theme.colors.grey500 }} />
            <Paragraph type="secondary">
              <FormattedMessage
                defaultMessage="Click 'Run Analysis' to analyze your recent traces for performance issues, quality degradation, and optimization opportunities."
                description="Empty state message for improve page"
              />
            </Paragraph>
          </div>
        </Card>
      )}

      {/* Health Summary */}
      {summary && (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: theme.spacing.sm,
            marginBottom: theme.spacing.lg,
          }}
        >
          <Card>
            <Text type="secondary" style={{ fontSize: 11, textTransform: 'uppercase' }}>
              Status
            </Text>
            <Title
              level={4}
              style={{
                color:
                  healthStatus === 'critical'
                    ? theme.colors.red
                    : healthStatus === 'warning'
                      ? theme.colors.yellow
                      : theme.colors.green,
                marginTop: 4,
              }}
            >
              {healthStatus === 'critical' ? 'Needs attention' : healthStatus === 'warning' ? 'Some issues' : 'Healthy'}
            </Title>
          </Card>
          <Card>
            <Text type="secondary" style={{ fontSize: 11, textTransform: 'uppercase' }}>
              Traces Analyzed
            </Text>
            <Title level={4} style={{ marginTop: 4 }}>
              {summary.traces_analyzed ?? 0}
            </Title>
          </Card>
          <Card>
            <Text type="secondary" style={{ fontSize: 11, textTransform: 'uppercase' }}>
              Findings
            </Text>
            <Title level={4} style={{ marginTop: 4 }}>
              {summary.findings_count ?? 0}
            </Title>
          </Card>
          <Card>
            <Text type="secondary" style={{ fontSize: 11, textTransform: 'uppercase' }}>
              Avg Tool Calls
            </Text>
            <Title level={4} style={{ marginTop: 4 }}>
              {summary.avg_tool_calls ?? '—'}
            </Title>
          </Card>
        </div>
      )}

      {/* Suggestions */}
      {analysisResult && analysisResult.suggestions.length > 0 && (
        <div style={{ marginBottom: theme.spacing.lg }}>
          <Title level={4} style={{ marginBottom: theme.spacing.sm }}>
            <FormattedMessage
              defaultMessage="Suggestions ({count})"
              description="Suggestions section title"
              values={{ count: analysisResult.suggestions.length }}
            />
          </Title>
          {analysisResult.suggestions.map((s) => (
            <Card key={s.id} style={{ marginBottom: theme.spacing.sm }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', gap: theme.spacing.xs, marginBottom: theme.spacing.xs }}>
                    <Tag color={SEVERITY_COLORS[s.severity] || 'charcoal'}>{s.severity}</Tag>
                    <Tag color="charcoal">{TYPE_LABELS[s.type] || s.type}</Tag>
                    <Text type="secondary" style={{ fontSize: 11 }}>
                      {Math.round(s.confidence * 100)}% confidence
                    </Text>
                  </div>
                  <Title level={4} style={{ marginTop: theme.spacing.xs }}>
                    {s.title}
                  </Title>
                  <Paragraph type="secondary">{s.description}</Paragraph>
                  <Card
                    style={{
                      marginTop: theme.spacing.sm,
                      backgroundColor: theme.colors.grey100,
                    }}
                  >
                    <Text type="secondary" style={{ fontSize: 11, textTransform: 'uppercase' }}>
                      Recommended action
                    </Text>
                    <Paragraph style={{ marginTop: 4 }}>{s.action}</Paragraph>
                  </Card>
                </div>
                {repoSaved && (
                  <Button
                    type="primary"
                    loading={isFixing === s.id}
                    onClick={() => triggerFix(s.id)}
                    style={{ marginLeft: theme.spacing.md }}
                  >
                    <FormattedMessage defaultMessage="Fix it" description="Button to trigger auto-fix" />
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
          <Title level={4} style={{ marginBottom: theme.spacing.sm }}>
            <FormattedMessage
              defaultMessage="Findings ({count})"
              description="Findings section title"
              values={{ count: analysisResult.findings.length }}
            />
          </Title>
          {analysisResult.findings.map((f, i) => (
            <Card key={i} style={{ marginBottom: theme.spacing.xs }}>
              <div style={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'flex-start' }}>
                <Tag color={SEVERITY_COLORS[f.severity] || 'charcoal'}>{f.severity}</Tag>
                <div>
                  <Text strong>{f.pattern.replace(/_/g, ' ')}</Text>
                  <Paragraph type="secondary" style={{ marginTop: 2 }}>
                    {f.description}
                  </Paragraph>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Empty state after analysis */}
      {analysisResult && analysisResult.suggestions.length === 0 && analysisResult.findings.length === 0 && (
        <Card>
          <div style={{ textAlign: 'center', padding: theme.spacing.lg }}>
            <Paragraph type="secondary">
              <FormattedMessage
                defaultMessage="No issues detected. Your agent is performing within expected parameters."
                description="No issues found message"
              />
            </Paragraph>
          </div>
        </Card>
      )}
    </div>
  );
};
