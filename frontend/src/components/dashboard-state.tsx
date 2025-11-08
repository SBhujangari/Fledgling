import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts"
import { TrendingDown, Target, Activity } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { useEffect, useState } from "react"
import axios from "axios"

export function DashboardState() {
  const [costData, setCostData] = useState([])
  const [accuracyData, setAccuracyData] = useState([])
  const [summary, setSummary] = useState({ targetAccuracy: 90, currentAccuracy: 94, costSavings: 90 })

  useEffect(() => {
    axios
      .get("/api/metrics/cost")
      .then((response) => setCostData(response.data))
      .catch((error) => console.error("Error fetching cost data:", error))

    axios
      .get("/api/metrics/accuracy")
      .then((response) => setAccuracyData(response.data))
      .catch((error) => console.error("Error fetching accuracy data:", error))

    axios
      .get("/api/metrics/summary")
      .then((response) => setSummary(response.data))
      .catch((error) => console.error("Error fetching summary data:", error))
  }, [])

  const { targetAccuracy, currentAccuracy, costSavings } = summary

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

      <div className="grid gap-6 md:grid-cols-3">
        <Card className="border-border bg-card">
          <CardHeader className="pb-3">
            <CardDescription className="text-muted-foreground">Cost Savings</CardDescription>
            <CardTitle className="text-3xl font-bold text-foreground">{costSavings}%</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center text-sm text-accent">
              <TrendingDown className="mr-1 size-4" />
              <span>${(450 - 45).toFixed(2)} saved per batch</span>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border bg-card">
          <CardHeader className="pb-3">
            <CardDescription className="text-muted-foreground">Current Accuracy</CardDescription>
            <CardTitle className="text-3xl font-bold text-foreground">{currentAccuracy}%</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center text-sm text-muted-foreground">
              <span>vs LLM baseline (100%)</span>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border bg-card">
          <CardHeader className="pb-3">
            <CardDescription className="text-muted-foreground">Target Goal</CardDescription>
            <CardTitle className="text-3xl font-bold text-foreground">{targetAccuracy}%</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center text-sm text-accent">
              <Target className="mr-1 size-4" />
              <span>Goal achieved!</span>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="border-border bg-card">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-foreground">Inference Cost Comparison</CardTitle>
              <CardDescription className="text-muted-foreground">
                Cost per batch on 20% test dataset split (in USD)
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={costData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.25 0 0)" />
              <XAxis dataKey="batch" stroke="oklch(0.6 0 0)" style={{ fontSize: "12px" }} />
              <YAxis
                stroke="oklch(0.6 0 0)"
                style={{ fontSize: "12px" }}
                label={{ value: "Cost ($)", angle: -90, position: "insideLeft", fill: "oklch(0.6 0 0)" }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "oklch(0.16 0 0)",
                  border: "1px solid oklch(0.25 0 0)",
                  borderRadius: "8px",
                  color: "oklch(0.98 0 0)",
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="llm"
                stroke="oklch(0.75 0.15 250)"
                strokeWidth={2}
                name="LLM"
                dot={{ fill: "oklch(0.75 0.15 250)" }}
              />
              <Line
                type="monotone"
                dataKey="slm"
                stroke="oklch(0.7 0.18 190)"
                strokeWidth={2}
                name="Fine-tuned SLM"
                dot={{ fill: "oklch(0.7 0.18 190)" }}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card className="border-border bg-card">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-foreground">Accuracy Comparison</CardTitle>
              <CardDescription className="text-muted-foreground">
                Model performance on evaluation dataset
              </CardDescription>
            </div>
            <Badge variant="secondary" className="bg-secondary text-secondary-foreground">
              Target: {targetAccuracy}%
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={accuracyData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.25 0 0)" />
              <XAxis dataKey="batch" stroke="oklch(0.6 0 0)" style={{ fontSize: "12px" }} />
              <YAxis
                stroke="oklch(0.6 0 0)"
                style={{ fontSize: "12px" }}
                domain={[70, 100]}
                label={{ value: "Accuracy (%)", angle: -90, position: "insideLeft", fill: "oklch(0.6 0 0)" }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "oklch(0.16 0 0)",
                  border: "1px solid oklch(0.25 0 0)",
                  borderRadius: "8px",
                  color: "oklch(0.98 0 0)",
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="llm"
                stroke="oklch(0.75 0.15 250)"
                strokeWidth={2}
                name="LLM Baseline"
                dot={{ fill: "oklch(0.75 0.15 250)" }}
              />
              <Line
                type="monotone"
                dataKey="slm"
                stroke="oklch(0.7 0.18 190)"
                strokeWidth={2}
                name="Fine-tuned SLM"
                dot={{ fill: "oklch(0.7 0.18 190)" }}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  )
}

