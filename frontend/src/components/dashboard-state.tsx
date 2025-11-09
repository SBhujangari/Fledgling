import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine } from "recharts"
import { TrendingDown, Activity, ArrowLeft } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { useState } from "react"

// Mock agents data
const mockAgents = [
  { id: "agent-1", name: "Research Agent", costSavings: 405, costSavingsPercent: 90, currentAccuracy: 94 },
  { id: "agent-2", name: "Analysis Agent", costSavings: 320, costSavingsPercent: 85, currentAccuracy: 91 },
  { id: "agent-3", name: "Synthesis Agent", costSavings: 280, costSavingsPercent: 88, currentAccuracy: 89 },
  { id: "agent-4", name: "Validation Agent", costSavings: 195, costSavingsPercent: 78, currentAccuracy: 87 },
]

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

export function DashboardState() {
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null)
  const [agents] = useState(mockAgents)
  
  // Calculate total cost savings
  const totalCostSavings = agents.reduce((sum, agent) => sum + agent.costSavings, 0)
  const totalCostSavingsPercent = Math.round((totalCostSavings / (agents.length * 450)) * 100)
  const avgAccuracy = Math.round(agents.reduce((sum, agent) => sum + agent.currentAccuracy, 0) / agents.length)

  const selectedAgentData = selectedAgent ? agents.find(a => a.id === selectedAgent) : null
  const costData = getMockCostData()
  const accuracyData = getMockAccuracyData()

  return (
    <div className="w-full h-full px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">Model Performance Dashboard</h1>
          <p className="text-muted-foreground mt-1">Real-time comparison of LLM vs fine-tuned SLM performance</p>
        </div>
        <Badge variant="outline" className="text-sm px-4 py-2 border-primary/30 text-primary">
          <Activity className="mr-2 size-3" />
          Training Active
        </Badge>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card className="border-border bg-card">
          <CardHeader className="pb-3">
            <CardDescription className="text-muted-foreground">Total Cost Savings</CardDescription>
            <CardTitle className="text-3xl font-bold text-foreground">${totalCostSavings.toFixed(2)}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center text-sm text-accent">
              <TrendingDown className="mr-1 size-4" />
              <span>{totalCostSavingsPercent}% reduction</span>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border bg-card">
          <CardHeader className="pb-3">
            <CardDescription className="text-muted-foreground">Average Accuracy</CardDescription>
            <CardTitle className="text-3xl font-bold text-foreground">{avgAccuracy}%</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center text-sm text-muted-foreground">
              <span>vs LLM baseline (100%)</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Agents Table or Graphs View */}
      <div className="relative overflow-hidden">
        {/* Agents Table View */}
        <div
          className={`space-y-4 transition-all duration-500 ease-in-out ${
            !selectedAgent
              ? "opacity-100 scale-100"
              : "opacity-0 scale-95 absolute inset-0 pointer-events-none"
          }`}
        >
          <div>
            <h2 className="text-2xl font-bold text-foreground">Agents</h2>
            <p className="text-muted-foreground mt-1">
              Click on an agent to view detailed performance metrics
            </p>
          </div>
          <Card className="border-border bg-card">
            <CardContent className="p-6">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-3 px-4 text-sm font-semibold text-foreground">Agent Name</th>
                      <th className="text-right py-3 px-4 text-sm font-semibold text-foreground">Cost Savings</th>
                      <th className="text-right py-3 px-4 text-sm font-semibold text-foreground">Accuracy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {agents.map((agent) => (
                      <tr
                        key={agent.id}
                        onClick={() => setSelectedAgent(agent.id)}
                        className="border-b border-border hover:bg-muted/50 cursor-pointer transition-colors"
                      >
                        <td className="py-4 px-4 text-foreground font-medium">{agent.name}</td>
                        <td className="py-4 px-4 text-right">
                          <span className="text-foreground font-semibold">${agent.costSavings.toFixed(2)}</span>
                          <span className="text-muted-foreground text-sm ml-2">({agent.costSavingsPercent}%)</span>
                        </td>
                        <td className="py-4 px-4 text-right text-foreground font-semibold">{agent.currentAccuracy}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Agent Graphs View */}
        <div
          className={`space-y-8 transition-all duration-500 ease-in-out ${
            selectedAgent
              ? "opacity-100 scale-100"
              : "opacity-0 scale-95 absolute inset-0 pointer-events-none"
          }`}
        >
            <div className="space-y-4">
              <button
                onClick={() => setSelectedAgent(null)}
                className="flex items-center gap-2 text-foreground hover:text-primary transition-colors"
              >
                <ArrowLeft className="size-4" />
                <span>Back to Agents</span>
              </button>
              <div>
                <h2 className="text-2xl font-bold text-foreground">{selectedAgentData?.name}</h2>
                <p className="text-muted-foreground mt-1">
                  Performance metrics for {selectedAgentData?.name}
                </p>
              </div>
            </div>

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

