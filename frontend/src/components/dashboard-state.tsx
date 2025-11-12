import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { TrendingDown, Activity } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { useNavigate } from "react-router-dom"
import { useAgents } from "@/hooks/useAgents"

export function DashboardState() {
  const navigate = useNavigate()
  const { data: agentsData, isLoading, error } = useAgents()

  // Transform API data to UI format
  const agents = (agentsData || []).map((agent) => {
    const costSavings = agent.model_costs_saved ?? null
    const currentAccuracy = agent.accuracy ?? null
    // Calculate percentage saved based on baseline cost of $450
    const baselineCost = 450
    const costSavingsPercent = costSavings !== null && costSavings > 0
      ? Math.round((costSavings / baselineCost) * 100)
      : null

    return {
      id: agent.id,
      name: agent.name || agent.id.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' '),
      costSavings,
      costSavingsPercent,
      currentAccuracy,
    }
  })

  // Calculate total cost savings (only include agents with available data)
  const agentsWithCostData = agents.filter(agent => agent.costSavings !== null)
  const totalCostSavings = agentsWithCostData.reduce((sum, agent) => sum + (agent.costSavings ?? 0), 0)
  const totalCostSavingsPercent = agentsWithCostData.length > 0
    ? Math.round((totalCostSavings / (agentsWithCostData.length * 450)) * 100)
    : null
  
  const agentsWithAccuracyData = agents.filter(agent => agent.currentAccuracy !== null)
  const avgAccuracy = agentsWithAccuracyData.length > 0
    ? Math.round(agentsWithAccuracyData.reduce((sum, agent) => sum + (agent.currentAccuracy ?? 0), 0) / agentsWithAccuracyData.length)
    : null

  if (isLoading) {
    return (
      <div className="w-full h-full px-4 sm:px-6 lg:px-8 py-8 flex items-center justify-center">
        <p className="text-muted-foreground">Loading agents...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="w-full h-full px-4 sm:px-6 lg:px-8 py-8 flex items-center justify-center">
        <p className="text-destructive">Error loading agents: {error.message}</p>
      </div>
    )
  }

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
            <CardTitle className="text-3xl font-bold text-foreground">
              {totalCostSavings !== null ? `$${totalCostSavings.toFixed(2)}` : 'N/A'}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center text-sm text-accent">
              <TrendingDown className="mr-1 size-4" />
              <span>{totalCostSavingsPercent !== null ? `${totalCostSavingsPercent}% reduction` : 'N/A'}</span>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border bg-card">
          <CardHeader className="pb-3">
            <CardDescription className="text-muted-foreground">Average Accuracy</CardDescription>
            <CardTitle className="text-3xl font-bold text-foreground">{avgAccuracy !== null ? `${avgAccuracy}%` : 'N/A'}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center text-sm text-muted-foreground">
              <span>vs LLM baseline (100%)</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Agents Table */}
      <div className="space-y-4">
          <div>
            <h2 className="text-2xl font-bold text-foreground">Agents</h2>
            <p className="text-muted-foreground mt-1">
              Click on an agent to view detailed performance metrics
            </p>
          </div>
          <Card className="border-border bg-card">
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-2 px-6 text-sm font-semibold text-foreground">Agent Name</th>
                      <th className="text-right py-2 px-6 text-sm font-semibold text-foreground">Cost Savings</th>
                      <th className="text-right py-2 px-6 text-sm font-semibold text-foreground">Accuracy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {agents.map((agent) => (
                      <tr
                        key={agent.id}
                        onClick={() => navigate(`/agent/${agent.id}`)}
                        className="border-b border-border last:border-0 hover:bg-muted/50 cursor-pointer transition-colors"
                      >
                        <td className="py-5 px-6 text-foreground font-medium">{agent.name}</td>
                        <td className="py-5 px-6 text-right">
                          {agent.costSavings !== null ? (
                            <>
                              <span className="text-foreground font-semibold text-base">${agent.costSavings.toFixed(2)}</span>
                              {agent.costSavingsPercent !== null && (
                                <span className="text-muted-foreground text-sm font-normal ml-2">({agent.costSavingsPercent}%)</span>
                              )}
                            </>
                          ) : (
                            <span className="text-muted-foreground font-semibold text-base">N/A</span>
                          )}
                        </td>
                        <td className="py-5 px-6 text-right">
                          {agent.currentAccuracy !== null ? (
                            <span className="text-foreground font-semibold text-base">{agent.currentAccuracy}%</span>
                          ) : (
                            <span className="text-muted-foreground font-semibold text-base">N/A</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
      </div>
    </div>
  )
}

