import { useParams, Link, useNavigate } from "react-router-dom"
import { Navigation } from "@/components/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, Plus } from "lucide-react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { api } from "@/lib/api"
import { useAgents } from "@/hooks/useAgents"

function getStatusBadgeVariant(status: string) {
  switch (status) {
    case 'completed':
      return 'default'
    case 'running':
      return 'secondary'
    case 'failed':
      return 'destructive'
    case 'queued':
      return 'outline'
    default:
      return 'outline'
  }
}

export default function RunsPage() {
  const { agentId } = useParams<{ agentId: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { data: agentsData } = useAgents()
  
  const agent = agentsData?.find(a => a.id === agentId)

  const { data: runs = [], isLoading } = useQuery({
    queryKey: ["runs", agentId],
    queryFn: () => api.getRuns(agentId),
    enabled: !!agentId,
  })

  const createRunMutation = useMutation({
    mutationFn: (name: string) => api.createRun({ agentId: agentId!, name }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["runs", agentId] })
      queryClient.invalidateQueries({ queryKey: ["agents"] })
    },
  })

  const handleCreateRun = () => {
    if (!agent) return
    const runName = `Run ${new Date().toLocaleDateString()}`
    createRunMutation.mutate(runName)
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Navigation />
        <div className="w-full h-full px-4 sm:px-6 lg:px-8 py-8 flex items-center justify-center">
          <p className="text-muted-foreground">Loading runs...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <div className="w-full h-full px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Header */}
        <div>
          <Link 
            to={`/agent/${agentId}`} 
            className="inline-flex items-center gap-2 text-foreground hover:text-primary transition-colors mb-4"
          >
            <ArrowLeft className="size-4" />
            <span>Back to Agent</span>
          </Link>
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold tracking-tight text-foreground">
                Fine-tuning Runs
              </h1>
              <p className="text-muted-foreground mt-1">
                {agent ? `Runs for ${agent.name}` : "Agent fine-tuning runs"}
              </p>
            </div>
            <Button
              onClick={handleCreateRun}
              disabled={!agent || agent.is_training || createRunMutation.isPending}
              variant="default"
            >
              <Plus className="size-4 mr-2" />
              New Run
            </Button>
          </div>
        </div>

        {/* Runs Table */}
        <Card className="border-border bg-card">
          <CardHeader>
            <CardTitle className="text-foreground">Runs</CardTitle>
            <CardDescription className="text-muted-foreground">
              Track fine-tuning runs for this agent
            </CardDescription>
          </CardHeader>
          <CardContent>
            {runs.length === 0 ? (
              <div className="text-center py-12">
                <p className="text-muted-foreground">No runs yet. Create a new run to get started.</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-3 px-6 text-sm font-semibold text-foreground">ID</th>
                      <th className="text-left py-3 px-6 text-sm font-semibold text-foreground">Name</th>
                      <th className="text-left py-3 px-6 text-sm font-semibold text-foreground">Status</th>
                      <th className="text-left py-3 px-6 text-sm font-semibold text-foreground">Date Queued</th>
                      <th className="text-left py-3 px-6 text-sm font-semibold text-foreground">Date Completed</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.map((run) => (
                      <tr
                        key={run.id}
                        className="border-b border-border last:border-0 hover:bg-muted/50 transition-colors"
                      >
                        <td className="py-5 px-6 text-foreground font-mono text-sm">{run.id}</td>
                        <td className="py-5 px-6 text-foreground font-medium">{run.name}</td>
                        <td className="py-5 px-6">
                          <Badge variant={getStatusBadgeVariant(run.status)}>
                            {run.status}
                          </Badge>
                        </td>
                        <td className="py-5 px-6 text-foreground">
                          {new Date(run.queuedAt).toLocaleString()}
                        </td>
                        <td className="py-5 px-6 text-foreground">
                          {run.completedAt ? new Date(run.completedAt).toLocaleString() : 'N/A'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

