import { useEffect, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Navigation } from "@/components/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Play, Pause, RefreshCw, CheckCircle2, AlertCircle, Clock, Cpu, Zap } from "lucide-react"

interface TrainingStatus {
  status: string
  message?: string
  currentStep?: number
  totalSteps?: number
  percentComplete?: number
  elapsedSeconds?: number
  remainingSeconds?: number
  remainingDisplay?: string
  avgStepSeconds?: number
  stepTimeDisplay?: string
  stepsPerMinute?: number
  eta?: string | null
  hardware?: {
    summary?: string
    gpuCount?: number
    cudaVisibleDevices?: string
    gpus?: Array<{
      name?: string
      memoryTotalGb?: number | null
    }>
  }
  updatedAt?: string
}

interface JudgeResult {
  winner: 'A' | 'B' | 'tie'
  overall_score_a: number
  overall_score_b: number
  confidence: number
  reasoning_summary: string
}

export default function TrainingDemoPage() {
  const [isTraining, setIsTraining] = useState(false)
  const [judgeResult, setJudgeResult] = useState<JudgeResult | null>(null)
  const [judgeLoading, setJudgeLoading] = useState(false)

  const { data: trainingStatus, refetch } = useQuery<TrainingStatus>({
    queryKey: ["training-status"],
    queryFn: async () => {
      const response = await fetch("http://localhost:4000/api/training/status")
      if (!response.ok) throw new Error("Failed to fetch training status")
      return response.json()
    },
    refetchInterval: 5000, // Refresh every 5 seconds
  })

  useEffect(() => {
    if (trainingStatus?.status === "running") {
      setIsTraining(true)
    } else if (trainingStatus?.status === "completed") {
      setIsTraining(false)
    }
  }, [trainingStatus])

  const startTraining = async () => {
    setIsTraining(true)
    // Trigger training endpoint
    try {
      const response = await fetch("http://localhost:4000/api/training", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          agentId: "demo-agent",
          llmModel: "gpt-4o-mini",
          slmModel: "qwen2.5-7b-instruct"
        })
      })
      if (!response.ok) throw new Error("Failed to start training")
      refetch()
    } catch (error) {
      console.error("Training error:", error)
      setIsTraining(false)
    }
  }

  const runJudgeEval = async () => {
    setJudgeLoading(true)
    try {
      const response = await fetch("http://localhost:4000/api/judge/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: "Explain quantum computing in simple terms",
          modelAOutput: "Quantum computing uses quantum mechanics principles like superposition and entanglement to process information in ways classical computers cannot.",
          modelBOutput: "Quantum computers are really fast computers that use quantum physics to do calculations.",
          criteria: ["accuracy", "helpfulness", "conciseness"]
        })
      })
      if (!response.ok) throw new Error("Failed to run judge evaluation")
      const result = await response.json()
      setJudgeResult(result.result)
    } catch (error) {
      console.error("Judge error:", error)
    } finally {
      setJudgeLoading(false)
    }
  }

  const getStatusIcon = () => {
    if (!trainingStatus) return <Clock className="h-5 w-5 text-muted-foreground" />
    switch (trainingStatus.status) {
      case "running":
        return <RefreshCw className="h-5 w-5 text-primary animate-spin" />
      case "completed":
        return <CheckCircle2 className="h-5 w-5 text-green-500" />
      case "error":
        return <AlertCircle className="h-5 w-5 text-destructive" />
      default:
        return <Clock className="h-5 w-5 text-muted-foreground" />
    }
  }

  const getStatusColor = () => {
    if (!trainingStatus) return "default"
    switch (trainingStatus.status) {
      case "running":
        return "default"
      case "completed":
        return "default"
      case "error":
        return "destructive"
      default:
        return "secondary"
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navigation />
      <main className="w-full px-4 py-8 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl space-y-8">
          {/* Header */}
          <header className="space-y-2 border-b border-border pb-6">
            <h1 className="text-3xl font-semibold tracking-tight">Auto-SFT Training Monitor</h1>
            <p className="text-muted-foreground">
              Real-time visualization of supervised fine-tuning progress and LLM-as-a-judge evaluation
            </p>
          </header>

          <Tabs defaultValue="training" className="w-full">
            <TabsList className="bg-muted">
              <TabsTrigger value="training">Training Status</TabsTrigger>
              <TabsTrigger value="judge">LLM Judge</TabsTrigger>
            </TabsList>

            {/* Training Tab */}
            <TabsContent value="training" className="space-y-6 mt-6">
              {/* Control Card */}
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      {getStatusIcon()}
                      <div>
                        <CardTitle>Training Status</CardTitle>
                        <CardDescription>
                          {trainingStatus?.message || "No active training session"}
                        </CardDescription>
                      </div>
                    </div>
                    <Badge variant={getStatusColor()}>
                      {trainingStatus?.status || "unknown"}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  {trainingStatus?.currentStep && trainingStatus?.totalSteps && (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">
                          Step {trainingStatus.currentStep} / {trainingStatus.totalSteps}
                        </span>
                        <span className="font-semibold">
                          {trainingStatus.percentComplete?.toFixed(1)}%
                        </span>
                      </div>
                      <Progress value={trainingStatus.percentComplete || 0} />
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>{trainingStatus.stepTimeDisplay || "..."}</span>
                        <span>ETA: {trainingStatus.remainingDisplay || trainingStatus.eta || "..."}</span>
                      </div>
                    </div>
                  )}

                  <div className="flex items-center gap-3 pt-2">
                    <Button
                      onClick={startTraining}
                      disabled={isTraining}
                      className="gap-2"
                    >
                      {isTraining ? (
                        <>
                          <RefreshCw className="h-4 w-4 animate-spin" />
                          Training...
                        </>
                      ) : (
                        <>
                          <Play className="h-4 w-4" />
                          Start Training
                        </>
                      )}
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => refetch()}
                      className="gap-2"
                    >
                      <RefreshCw className="h-4 w-4" />
                      Refresh
                    </Button>
                  </div>
                </CardContent>
              </Card>

              {/* Hardware Info */}
              {trainingStatus?.hardware && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Cpu className="h-5 w-5" />
                      Hardware Configuration
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {trainingStatus.hardware.summary && (
                      <p className="text-sm text-muted-foreground">
                        {trainingStatus.hardware.summary}
                      </p>
                    )}
                    <div className="grid gap-4 md:grid-cols-2">
                      {trainingStatus.hardware.gpus?.map((gpu, idx) => (
                        <div key={idx} className="rounded-lg border border-border p-3 space-y-1">
                          <p className="font-semibold text-sm">{gpu.name || `GPU ${idx}`}</p>
                          {gpu.memoryTotalGb && (
                            <p className="text-xs text-muted-foreground">
                              {gpu.memoryTotalGb.toFixed(1)} GB VRAM
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Metrics */}
              <div className="grid gap-6 md:grid-cols-3">
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Steps/Min
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-2xl font-bold">
                      {trainingStatus?.stepsPerMinute?.toFixed(1) || "—"}
                    </p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Elapsed Time
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-2xl font-bold">
                      {trainingStatus?.elapsedSeconds
                        ? `${Math.floor(trainingStatus.elapsedSeconds / 60)}m ${Math.floor(trainingStatus.elapsedSeconds % 60)}s`
                        : "—"}
                    </p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Remaining
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-2xl font-bold">
                      {trainingStatus?.remainingDisplay || "—"}
                    </p>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            {/* Judge Tab */}
            <TabsContent value="judge" className="space-y-6 mt-6">
              <Card>
                <CardHeader>
                  <CardTitle>LLM-as-a-Judge Evaluation</CardTitle>
                  <CardDescription>
                    Use a powerful LLM to evaluate and compare model outputs
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Button
                    onClick={runJudgeEval}
                    disabled={judgeLoading}
                    className="gap-2"
                  >
                    {judgeLoading ? (
                      <>
                        <RefreshCw className="h-4 w-4 animate-spin" />
                        Evaluating...
                      </>
                    ) : (
                      <>
                        <Zap className="h-4 w-4" />
                        Run Demo Evaluation
                      </>
                    )}
                  </Button>

                  {judgeResult && (
                    <div className="space-y-4 pt-4 border-t">
                      <div className="flex items-center justify-between">
                        <h3 className="text-lg font-semibold">Evaluation Results</h3>
                        <Badge variant="outline" className="text-lg px-3 py-1">
                          Winner: Model {judgeResult.winner.toUpperCase()}
                        </Badge>
                      </div>

                      <div className="grid gap-4 md:grid-cols-2">
                        <Card className="border-2">
                          <CardHeader className="pb-3">
                            <CardTitle className="text-sm">Model A</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <p className="text-3xl font-bold">
                              {judgeResult.overall_score_a.toFixed(1)}
                            </p>
                            <p className="text-sm text-muted-foreground">Overall Score</p>
                          </CardContent>
                        </Card>

                        <Card className="border-2">
                          <CardHeader className="pb-3">
                            <CardTitle className="text-sm">Model B</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <p className="text-3xl font-bold">
                              {judgeResult.overall_score_b.toFixed(1)}
                            </p>
                            <p className="text-sm text-muted-foreground">Overall Score</p>
                          </CardContent>
                        </Card>
                      </div>

                      <div className="rounded-lg bg-muted/50 p-4">
                        <p className="text-sm font-medium mb-2">Judge's Reasoning:</p>
                        <p className="text-sm text-muted-foreground">{judgeResult.reasoning_summary}</p>
                        <p className="text-xs text-muted-foreground mt-2">
                          Confidence: {(judgeResult.confidence * 100).toFixed(0)}%
                        </p>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  )
}
