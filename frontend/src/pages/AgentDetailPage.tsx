import { useParams, Link, useNavigate } from "react-router-dom"
import { Navigation } from "@/components/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Label } from "@/components/ui/label"
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine } from "recharts"
import { ArrowLeft, Cpu, Database, Activity } from "lucide-react"
import { useAgents } from "@/hooks/useAgents"

// Mock data for cost comparison (per agent)
const getMockCostData = () => [
  { batch: "Batch 1", llm: 450, slm: 180 },
  { batch: "Batch 2", llm: 445, slm: 165 },
  { batch: "Batch 3", llm: 460, slm: 150 },
  { batch: "Batch 4", llm: 455, slm: 140 },
  { batch: "Batch 5", llm: 450, slm: 135 },
  { batch: "Batch 6", llm: 445, slm: 120 },
  { batch: "Batch 7", llm: 460, slm: 110 },
  { batch: "Batch 8", llm: 455, slm: 95 },
  { batch: "Batch 9", llm: 450, slm: 85 },
  { batch: "Batch 10", llm: 445, slm: 75 },
  { batch: "Batch 11", llm: 460, slm: 65 },
  { batch: "Batch 12", llm: 455, slm: 55 },
  { batch: "Batch 13", llm: 450, slm: 50 },
  { batch: "Batch 14", llm: 445, slm: 48 },
  { batch: "Batch 15", llm: 460, slm: 45 },
]

// Mock data for accuracy comparison (per agent)
const getMockAccuracyData = () => [
  { batch: "Batch 1", llm: 100, slm: 72 },
  { batch: "Batch 2", llm: 100, slm: 75 },
  { batch: "Batch 3", llm: 100, slm: 78 },
  { batch: "Batch 4", llm: 100, slm: 80 },
  { batch: "Batch 5", llm: 100, slm: 82 },
  { batch: "Batch 6", llm: 100, slm: 84 },
  { batch: "Batch 7", llm: 100, slm: 86 },
  { batch: "Batch 8", llm: 100, slm: 87 },
  { batch: "Batch 9", llm: 100, slm: 88 },
  { batch: "Batch 10", llm: 100, slm: 89 },
  { batch: "Batch 11", llm: 100, slm: 90 },
  { batch: "Batch 12", llm: 100, slm: 91 },
  { batch: "Batch 13", llm: 100, slm: 92 },
  { batch: "Batch 14", llm: 100, slm: 93 },
  { batch: "Batch 15", llm: 100, slm: 94 },
]

export default function AgentDetailPage() {
  const { agentId } = useParams<{ agentId: string }>()
  const navigate = useNavigate()
  const { data: agentsData, isLoading, error } = useAgents()

  const agent = agentsData?.find(a => a.id === agentId)

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Navigation />
        <div className="w-full h-full px-4 sm:px-6 lg:px-8 py-8 flex items-center justify-center">
          <p className="text-muted-foreground">Loading agent details...</p>
        </div>
      </div>
    )
  }

  if (error || !agent) {
    return (
      <div className="min-h-screen bg-background">
        <Navigation />
        <div className="w-full h-full px-4 sm:px-6 lg:px-8 py-8">
          <Link to="/" className="inline-flex items-center gap-2 text-foreground hover:text-primary transition-colors mb-4">
            <ArrowLeft className="size-4" />
            <span>Back to Dashboard</span>
          </Link>
          <p className="text-destructive">Agent not found</p>
        </div>
      </div>
    )
  }

  const costData = getMockCostData()
  const accuracyData = getMockAccuracyData()
  const currentAccuracy = agent.accuracy ?? null

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <div className="w-full h-full px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Header with back button */}
        <div>
          <Link to="/" className="inline-flex items-center gap-2 text-foreground hover:text-primary transition-colors mb-4">
            <ArrowLeft className="size-4" />
            <span>Back to Dashboard</span>
          </Link>
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold tracking-tight text-foreground">{agent.name}</h1>
              <p className="text-muted-foreground mt-1">Performance metrics & Finetuning Config</p>
            </div>
            <Button
              onClick={() => navigate(`/playground?llm=${encodeURIComponent(agent.original_llm)}&slm=${encodeURIComponent(agent.slm_model)}`)}
              variant="default"
            >
              Try Out
            </Button>
          </div>
        </div>

        {/* 3 Cards Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Current Model Configuration */}
          <Card className="border-border bg-card">
            <CardHeader>
              <div className="flex items-start justify-between">
                <div>
                  <CardTitle className="text-foreground flex items-center gap-2">
                    <Cpu className="size-5 text-primary" />
                    Model
                  </CardTitle>
                  <CardDescription className="text-muted-foreground">
                    Active model configuration
                  </CardDescription>
                </div>
                <Badge variant="secondary" className="bg-accent/20 text-accent border-accent/30">
                  Active
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label className="text-sm text-muted-foreground">SLM Model</Label>
                <p className="text-lg font-semibold text-foreground mt-1">{agent.slm_model}</p>
              </div>
              <div>
                <Label className="text-sm text-muted-foreground">Original LLM</Label>
                <p className="text-foreground mt-1">{agent.original_llm}</p>
              </div>
              <div>
                <Label className="text-sm text-muted-foreground">Task Description</Label>
                <p className="text-foreground mt-1">{agent.task_description}</p>
              </div>
            </CardContent>
          </Card>

          {/* Training Status */}
          <Card className="border-border bg-card">
            <CardHeader>
              <CardTitle className="text-foreground flex items-center gap-2">
                <Activity className="size-5 text-primary" />
                Training Status
              </CardTitle>
              <CardDescription className="text-muted-foreground">
                Currently Training...
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label className="text-sm text-muted-foreground">Status</Label>
                <div className="flex items-center gap-2 mt-1">
                  <div className="size-2 rounded-full bg-accent animate-pulse" />
                  <p className="text-foreground">Active fine-tuning in progress</p>
                </div>
              </div>
              <div>
                <Label className="text-sm text-muted-foreground">Iterations Completed</Label>
                <p className="text-lg font-semibold text-foreground mt-1">2,847</p>
              </div>
              <div>
                <Label className="text-sm text-muted-foreground">Training Data Size</Label>
                <p className="text-lg font-semibold text-foreground mt-1">18,542 samples</p>
              </div>
            </CardContent>
          </Card>

          {/* Model Metrics */}
          <Card className="border-border bg-card">
            <CardHeader>
              <CardTitle className="text-foreground flex items-center gap-2">
                <Database className="size-5 text-primary" />
                Performance Metrics
              </CardTitle>
              <CardDescription className="text-muted-foreground">
                Cost savings and accuracy
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label className="text-sm text-muted-foreground">Total Cost Savings</Label>
                <p className="text-2xl font-bold text-foreground mt-1">
                  {agent.model_costs_saved !== null ? `$${agent.model_costs_saved.toFixed(2)}` : 'N/A'}
                </p>
              </div>
              <div>
                <Label className="text-sm text-muted-foreground">Current Accuracy</Label>
                <p className="text-2xl font-bold text-foreground mt-1">{currentAccuracy !== null ? `${currentAccuracy}%` : 'N/A'}</p>
              </div>
              <div>
                <Label className="text-sm text-muted-foreground">Last Updated</Label>
                <p className="text-sm text-foreground mt-1">
                  {new Date(agent.last_updated_at).toLocaleDateString()}
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Performance Charts */}
        <div className="space-y-8">
          {/* Inference Cost Comparison */}
          <div className="space-y-4">
            <div>
              <h3 className="text-xl font-bold text-foreground">Inference Cost Comparison</h3>
              <p className="text-muted-foreground mt-1">
                Cost per batch on 20% test dataset split (in USD)
              </p>
            </div>
            <Card className="border-border bg-card">
              <CardContent className="p-6">
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={costData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="batch" stroke="var(--muted-foreground)" style={{ fontSize: "12px" }} />
                    <YAxis
                      stroke="var(--muted-foreground)"
                      style={{ fontSize: "12px" }}
                      label={{ value: "Cost ($)", angle: -90, position: "insideLeft", fill: "var(--muted-foreground)" }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "var(--card)",
                        border: "1px solid var(--border)",
                        borderRadius: "8px",
                        color: "var(--foreground)",
                      }}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="llm"
                      stroke="var(--primary)"
                      strokeWidth={2}
                      name="LLM"
                      dot={{ fill: "var(--primary)" }}
                    />
                    <Line
                      type="monotone"
                      dataKey="slm"
                      stroke="var(--accent)"
                      strokeWidth={2}
                      name="Fine-tuned SLM"
                      dot={{ fill: "var(--accent)" }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Accuracy Comparison */}
          <div className="space-y-4">
            <div>
              <h3 className="text-xl font-bold text-foreground">Accuracy Comparison</h3>
              <p className="text-muted-foreground mt-1">
                Model performance on evaluation dataset
              </p>
            </div>
            <Card className="border-border bg-card">
              <CardContent className="p-6">
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={accuracyData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="batch" stroke="var(--muted-foreground)" style={{ fontSize: "12px" }} />
                    <YAxis
                      stroke="var(--muted-foreground)"
                      style={{ fontSize: "12px" }}
                      domain={[70, 100]}
                      label={{ value: "Accuracy (%)", angle: -90, position: "insideLeft", fill: "var(--muted-foreground)" }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "var(--card)",
                        border: "1px solid var(--border)",
                        borderRadius: "8px",
                        color: "var(--foreground)",
                      }}
                    />
                    <Legend />
                    <ReferenceLine
                      y={90}
                      stroke="var(--destructive)"
                      strokeDasharray="5 5"
                      label={{ value: "Target: 90%", position: "right", fill: "var(--destructive)" }}
                    />
                    <Line
                      type="monotone"
                      dataKey="llm"
                      stroke="var(--primary)"
                      strokeWidth={2}
                      name="LLM Baseline"
                      dot={{ fill: "var(--primary)" }}
                    />
                    <Line
                      type="monotone"
                      dataKey="slm"
                      stroke="var(--accent)"
                      strokeWidth={2}
                      name="Fine-tuned SLM"
                      dot={{ fill: "var(--accent)" }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
