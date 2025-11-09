import { Navigation } from "@/components/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Settings, Cpu, Database, Activity, AlertCircle, Target, Download } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { useState } from "react"

export default function TuningPage() {
  const [customModel, setCustomModel] = useState(false)
  const [selectedModel, setSelectedModel] = useState("default")
  const [iterationCycle, setIterationCycle] = useState("500")
  const [autoExport, setAutoExport] = useState(true)
  const [accuracyThreshold, setAccuracyThreshold] = useState("90")
  const [settingsOpen, setSettingsOpen] = useState(false)

  // START: Hard-coded value - replace with backend API call
  // API endpoint: GET /api/tuning/status
  // Expected response: { currentAccuracy: number, modelConfig: object, trainingStatus: object }
  const currentAccuracy = 94
  // END: Hard-coded value

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <div className="w-full h-full px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-foreground">Fine-Tuning Configuration</h1>
            <p className="text-muted-foreground mt-1">Monitor and configure your SLM training parameters</p>
          </div>
          <div className="flex items-center gap-4">
            <Badge variant="outline" className="text-sm px-4 py-2 border-primary/30 text-primary">
              <Activity className="mr-2 size-3" />
              {currentAccuracy}% Accuracy
            </Badge>
            <Dialog open={settingsOpen} onOpenChange={setSettingsOpen}>
              <DialogTrigger asChild>
                <Button variant="outline">
                  <Settings className="mr-2 size-4" />
                  Model Settings
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl">
                <DialogHeader>
                  <DialogTitle>Model & Export Settings</DialogTitle>
                  <DialogDescription>
                    Configure model selection and export preferences
                  </DialogDescription>
                </DialogHeader>
                
                <div className="space-y-6 mt-4">
                  {/* Model Selection */}
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-foreground">Model Selection</h3>
                    <div className="flex items-center justify-between p-4 bg-secondary/50 rounded-lg border border-border">
                      <div className="space-y-1">
                        <Label htmlFor="custom-model" className="text-foreground">
                          Use Custom Model
                        </Label>
                        <p className="text-sm text-muted-foreground">
                          Override recommended model (may lead to suboptimal results)
                        </p>
                      </div>
                      <Switch id="custom-model" checked={customModel} onCheckedChange={setCustomModel} />
                    </div>

                    {customModel && (
                      <>
                        <Alert className="border-destructive/50 bg-destructive/10">
                          <AlertCircle className="size-4 text-destructive" />
                          <AlertDescription className="text-destructive">
                            Using a custom model may result in worse performance. Our recommended model is optimized for your specific use case.
                          </AlertDescription>
                        </Alert>

                        <div className="space-y-2">
                          <Label htmlFor="model-select" className="text-foreground">
                            Select Model
                          </Label>
                          {/* START: Hard-coded model list - replace with backend API call */}
                          {/* API endpoint: GET /api/models/available */}
                          {/* Expected response: Array<{ id, name, description, recommended: boolean }> */}
                          <Select value={selectedModel} onValueChange={setSelectedModel}>
                            <SelectTrigger id="model-select" className="bg-input border-border text-foreground">
                              <SelectValue placeholder="Choose a model" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="default">Llama-3.2-1B (Recommended)</SelectItem>
                              <SelectItem value="llama-1b">Llama-3.2-1B-Base</SelectItem>
                              <SelectItem value="llama-3b">Llama-3.2-3B-Base</SelectItem>
                              <SelectItem value="phi-2">Phi-2-2.7B</SelectItem>
                              <SelectItem value="gemma-2b">Gemma-2B</SelectItem>
                              <SelectItem value="mistral-7b">Mistral-7B-Base</SelectItem>
                            </SelectContent>
                          </Select>
                          {/* END: Hard-coded model list */}
                        </div>
                      </>
                    )}
                  </div>

                  {/* Export Settings */}
                  <div className="space-y-4 pt-4 border-t border-border">
                    <h3 className="text-lg font-semibold text-foreground">Export Settings</h3>
                    <div className="flex items-center justify-between p-4 bg-secondary/50 rounded-lg border border-border">
                      <div className="space-y-1">
                        <Label htmlFor="auto-export" className="text-foreground font-semibold">
                          Auto-Export on Accuracy Threshold
                        </Label>
                        <p className="text-sm text-muted-foreground">
                          Automatically export model when accuracy goal is reached
                        </p>
                      </div>
                      <Switch id="auto-export" checked={autoExport} onCheckedChange={setAutoExport} />
                    </div>

                    {autoExport && (
                      <div className="space-y-2 pl-4 border-l-2 border-accent/50">
                        <Label htmlFor="accuracy-threshold" className="text-foreground">
                          Accuracy Threshold for Auto-Export
                        </Label>
                        <div className="flex items-center gap-4">
                          <Input
                            id="accuracy-threshold"
                            type="number"
                            value={accuracyThreshold}
                            onChange={(e) => setAccuracyThreshold(e.target.value)}
                            min="0"
                            max="100"
                            className="max-w-[200px] bg-input border-border text-foreground"
                          />
                          <span className="text-muted-foreground">%</span>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          Model will automatically export when accuracy reaches {accuracyThreshold}%
                        </p>
                        {currentAccuracy >= Number.parseInt(accuracyThreshold || "0") && (
                          <Alert className="border-accent/50 bg-accent/10">
                            <Target className="size-4 text-accent" />
                            <AlertDescription className="text-accent">
                              Threshold reached! Current accuracy ({currentAccuracy}%) meets the export requirement.
                            </AlertDescription>
                          </Alert>
                        )}
                      </div>
                    )}

                    <div className="pt-4 border-t border-border">
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <h4 className="text-base font-semibold text-foreground">Manual Export</h4>
                          <p className="text-sm text-muted-foreground mt-1">
                            Export the model at any time if you're satisfied with the current performance
                          </p>
                        </div>
                        <Button className="bg-primary text-primary-foreground hover:bg-primary/90">
                          <Download className="mr-2 size-4" />
                          Export Now
                        </Button>
                      </div>

                      <div className="p-4 bg-muted/30 rounded-lg space-y-2">
                        {/* START: Hard-coded export info - replace with backend API call */}
                        {/* API endpoint: GET /api/export/info */}
                        {/* Expected response: { accuracy, size, format } */}
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">Current Accuracy</span>
                          <span className="font-semibold text-foreground">{currentAccuracy}%</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">Model Size</span>
                          <span className="font-semibold text-foreground">~2.1 GB</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">Export Format</span>
                          <span className="font-semibold text-foreground">GGUF / SafeTensors</span>
                        </div>
                        {/* END: Hard-coded export info */}
                      </div>
                    </div>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Current Model Configuration */}
          <Card className="border-border bg-card">
            <CardHeader>
              <div className="flex items-start justify-between">
                <div>
                  <CardTitle className="text-foreground flex items-center gap-2">
                    <Cpu className="size-5 text-primary" />
                    Current Model
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
              {/* START: Hard-coded model info - replace with backend API call */}
              {/* API endpoint: GET /api/models/current */}
              {/* Expected response: { name, description, parameters, contextWindow, topic } */}
              <div>
                <Label className="text-sm text-muted-foreground">Model Arch</Label>
                <p className="text-lg font-semibold text-foreground mt-1">Llama-3.2-1B</p>
              </div>
              <div>
                <Label className="text-sm text-muted-foreground">Task Specification</Label>
                <p className="text-foreground mt-1">
                  Solves the user's workflow for classifying financial documents into investor reports, company reports, and tax docs.
                </p>
              </div>
              <div>
                <Label className="text-sm text-muted-foreground">Parameters</Label>
                <div className="flex items-center gap-4 mt-1">
                  <div>
                    <p className="text-2xl font-bold text-foreground">1B</p>
                    <p className="text-xs text-muted-foreground">Total params</p>
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-foreground">4K</p>
                    <p className="text-xs text-muted-foreground">Context window</p>
                  </div>
                </div>
              </div>
              {/* END: Hard-coded model info */}
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
              {/* START: Hard-coded training status - replace with backend API call */}
              {/* API endpoint: GET /api/training/status */}
              {/* Expected response: { status, iterations, dataSize } */}
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
              {/* END: Hard-coded training status */}
            </CardContent>
          </Card>

          {/* Fine-Tuning Cycle Configuration */}
          <Card className="border-border bg-card">
            <CardHeader>
              <CardTitle className="text-foreground flex items-center gap-2">
                <Database className="size-5 text-primary" />
                Cycle Settings
              </CardTitle>
              <CardDescription className="text-muted-foreground">
                Configure fine-tuning cycle intervals
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="iteration-cycle" className="text-foreground">
                  Iterations Before Next Cycle
                </Label>
                <div className="flex items-center gap-4">
                  <Input
                    id="iteration-cycle"
                    type="number"
                    value={iterationCycle}
                    onChange={(e) => setIterationCycle(e.target.value)}
                    className="w-full bg-input border-border text-foreground"
                  />
                </div>
                <p className="text-sm text-muted-foreground">
                  Next cycle after {iterationCycle} more iterations
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
