import { useEffect, useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Sparkles, Zap, SendHorizontal } from "lucide-react"
import { ChatMessage } from "@/components/chat-message"
import { TraceMessage } from "@/components/trace-message"
import type { HistoryItem } from "@/types"

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "")

interface AgentOption {
  id: string
  name: string
  slm_model: string
  original_llm: string
}

interface ModelOption {
  id: string
  label: string
  description?: string
}

interface CompareRun {
  label: string
  model: string
  provider: string
  fallback?: boolean
  result: {
    output?: string
    error?: string
  }
}

interface TraceSampleStep {
  type: string
  toolName?: string
  input?: unknown
  output?: unknown
}

interface CompareResponse {
  prompt: string
  runs: CompareRun[]
  traceSample?: { steps?: TraceSampleStep[] }
}

export function Playground() {
  const [prompt, setPrompt] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [liveComparison, setLiveComparison] = useState<HistoryItem | null>(null)
  const [agents, setAgents] = useState<AgentOption[]>([])
  const [llmModels, setLlmModels] = useState<ModelOption[]>([])
  const [selectedAgentId, setSelectedAgentId] = useState<string>("")
  const [selectedLlmModel, setSelectedLlmModel] = useState<string>("")
  const [fallbackNotice, setFallbackNotice] = useState<string | null>(null)

  useEffect(() => {
    async function loadMetadata() {
      try {
        const [agentsRes, modelsRes] = await Promise.all([
          fetch(`${API_BASE_URL}/api/agents`),
          fetch(`${API_BASE_URL}/api/models/llm`),
        ])

        const agentsData = (await agentsRes.json()) as AgentOption[]
        const modelsData = (await modelsRes.json()) as ModelOption[]

        setAgents(agentsData)
        setLlmModels(modelsData)
        if (agentsData.length > 0) {
          setSelectedAgentId((prev) => prev || agentsData[0].id)
        }
        if (modelsData.length > 0) {
          setSelectedLlmModel((prev) => prev || modelsData[0].id)
        }
      } catch (err) {
        console.error("Failed to load metadata", err)
        setError("Failed to load agents/models. Ensure backend is running.")
      }
    }

    loadMetadata()
  }, [])

  const llmLabel = useMemo(() => {
    return llmModels.find((model) => model.id === selectedLlmModel)?.label || selectedLlmModel
  }, [llmModels, selectedLlmModel])

  const slmLabel = useMemo(() => {
    return agents.find((agent) => agent.id === selectedAgentId)?.name || "SLM"
  }, [agents, selectedAgentId])

  const originalModel = useMemo(() => {
    return agents.find((agent) => agent.id === selectedAgentId)?.original_llm || ''
  }, [agents, selectedAgentId])

  const handleSubmit = async () => {
    setError(null)
    if (!prompt.trim() || !selectedAgentId) return

    setIsLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/api/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          agentId: selectedAgentId,
          prompt,
          llmModel: selectedLlmModel,
        }),
      })

      if (!response.ok) {
        const body = await response.text()
        throw new Error(body)
      }

      const data: CompareResponse = await response.json()
      const [llmRun, slmRun] = data.runs ?? []

      const slmToolCalls = Array.isArray(data.traceSample?.steps)
        ? data.traceSample.steps
            .filter((step: { type?: string }) => step.type === "tool_call")
            .map((step: any) => ({
              name: step.toolName ?? "tool",
              input: typeof step.input === "object" && step.input !== null ? step.input : { value: step.input },
              result: step.output ?? null,
            }))
        : undefined

      if (slmRun?.fallback) {
        const fallbackModelName = typeof slmRun.model === 'string'
          ? slmRun.model
          : slmRun.label || 'fallback model';
        setFallbackNotice(
          `SLM provider unavailable; using ${fallbackModelName} for this comparison.`
        )
      } else {
        setFallbackNotice(null)
      }

      setLiveComparison({
        id: `compare-${Date.now()}`,
        timestamp: new Date().toLocaleString(),
        iteration: "",
        prompt: data.prompt,
        llmResponse: {
          content: llmRun?.result?.output || `Error: ${llmRun?.result?.error ?? "No response"}`,
        },
        slmResponse: {
          content: slmRun?.result?.output || `Error: ${slmRun?.result?.error ?? "No response"}`,
          toolCalls: slmToolCalls,
        },
        slmFallback: Boolean(slmRun?.fallback),
      })
    } catch (err) {
      console.error("Comparison failed", err)
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="w-full h-full px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-foreground mb-2">Playground</h1>
        <p className="text-muted-foreground">Compare LLM and SLM outputs side-by-side</p>
      </div>

      {/* Unified Testing Component */}
      <Card className="p-6 mb-8 bg-card border-border">
        <h2 className="text-xl font-semibold text-foreground mb-6">Playground</h2>

        {error && (
          <p className="text-sm text-destructive mb-4">{error}</p>
        )}
        
        {/* LLM and SLM Output Panels */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div>
            <div className="flex items-center gap-2 mb-4 flex-wrap">
              <Sparkles className="size-5 text-primary" />
              <Select value={selectedLlmModel} onValueChange={setSelectedLlmModel}>
                <SelectTrigger className="bg-transparent border-none shadow-none p-0 h-auto text-lg font-semibold text-foreground hover:bg-transparent focus:ring-0 cursor-pointer w-auto">
                  <SelectValue>
                    {llmLabel || "Baseline LLM"}
                  </SelectValue>
                </SelectTrigger>
                <SelectContent className="bg-card border-border text-foreground">
                  {llmModels.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {originalModel && (
                <span className="text-xs text-muted-foreground ml-2">Original: {originalModel}</span>
              )}
            </div>
            {liveComparison ? (
              <div className="min-h-[400px]">
                <ChatMessage prompt={liveComparison.prompt} response={liveComparison.llmResponse} type="llm" />
              </div>
            ) : (
              <div className="min-h-[400px] flex items-center justify-center text-muted-foreground border border-border rounded-lg bg-background/50">
                <p className="text-lg">LLM response will appear here</p>
              </div>
            )}
          </div>
          <div>
            <div className="flex items-center gap-2 mb-4 flex-wrap">
              <Zap className="size-5 text-accent" />
              <Select value={selectedAgentId} onValueChange={setSelectedAgentId}>
                <SelectTrigger className="bg-transparent border-none shadow-none p-0 h-auto text-lg font-semibold text-foreground hover:bg-transparent focus:ring-0 cursor-pointer w-auto">
                  <SelectValue>
                    {slmLabel || "Tracked Agent (SLM)"}
                  </SelectValue>
                </SelectTrigger>
                <SelectContent className="bg-card border-border text-foreground">
                  {agents.map((agent) => (
                    <SelectItem key={agent.id} value={agent.id}>
                      {agent.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {fallbackNotice && (
                <span className="text-sm text-destructive ml-2">{fallbackNotice}</span>
              )}
            </div>
            {liveComparison ? (
              <div className="min-h-[400px]">
                <ChatMessage prompt={liveComparison.prompt} response={liveComparison.slmResponse} type="slm" />
              </div>
            ) : (
              <div className="min-h-[400px] flex items-center justify-center text-muted-foreground border border-border rounded-lg bg-background/50">
                <p className="text-lg">SLM response will appear here</p>
              </div>
            )}
          </div>
        </div>

        {/* Testing Input Section */}
        <div className="flex flex-col sm:flex-row gap-4 pt-6 border-t border-border">
          <Textarea
            placeholder="Enter your prompt to compare LLM and SLM outputs..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="min-h-24 flex-1 bg-background text-foreground border-border w-full text-lg placeholder:text-lg"
          />
          <Button 
            onClick={handleSubmit} 
            disabled={isLoading || !selectedAgentId} 
            className="self-end sm:self-end h-24 w-12 bg-primary hover:bg-primary/90 text-primary-foreground aspect-square"
          >
            <SendHorizontal className="size-4" />
          </Button>
        </div>
      </Card>

      <Tabs defaultValue="chat" className="w-full">
        <TabsList className="bg-muted">
          <TabsTrigger value="chat">Chat</TabsTrigger>
          <TabsTrigger value="trace">Trace</TabsTrigger>
        </TabsList>

        <TabsContent value="chat" className="mt-6">
          {liveComparison ? (
            <Card className="p-6 bg-card border-border">
              <div className="mb-4">
                <span className="text-xs text-muted-foreground">{liveComparison.timestamp}</span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full">
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <Sparkles className="size-4 text-primary" />
                    <span className="text-sm font-semibold text-foreground">{llmLabel}</span>
                  </div>
                  <ChatMessage prompt={liveComparison.prompt} response={liveComparison.llmResponse} type="llm" />
                </div>
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <Zap className="size-4 text-accent" />
                    <span className="text-sm font-semibold text-foreground">{slmLabel}</span>
                  </div>
                  <ChatMessage prompt={liveComparison.prompt} response={liveComparison.slmResponse} type="slm" />
                </div>
              </div>
            </Card>
          ) : (
            <Card className="p-12 bg-card border-border text-center">
              <p className="text-muted-foreground">Run a query above to see chat responses</p>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="trace" className="mt-6">
          {liveComparison ? (
            <Card className="p-6 bg-card border-border">
              <div className="mb-4">
                <span className="text-xs text-muted-foreground">{liveComparison.timestamp}</span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full">
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <Sparkles className="size-4 text-primary" />
                    <span className="text-sm font-semibold text-foreground">{llmLabel} Trace</span>
                  </div>
                  <TraceMessage prompt={liveComparison.prompt} response={liveComparison.llmResponse} type="llm" />
                </div>
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <Zap className="size-4 text-accent" />
                    <span className="text-sm font-semibold text-foreground">{slmLabel} Trace</span>
                  </div>
                  <TraceMessage prompt={liveComparison.prompt} response={liveComparison.slmResponse} type="slm" />
                </div>
              </div>
            </Card>
          ) : (
            <Card className="p-12 bg-card border-border text-center">
              <p className="text-muted-foreground">Run a query above to see traces</p>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}
