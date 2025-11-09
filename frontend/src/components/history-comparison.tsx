import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { SendHorizontal, Sparkles, Zap } from "lucide-react"
import { ChatMessage } from "@/components/chat-message"
import { TraceMessage } from "@/components/trace-message"
import type { HistoryItem } from "@/types"

export function HistoryComparison() {
  const [prompt, setPrompt] = useState("")
  const [selectedIteration, setSelectedIteration] = useState("all")
  const [isLoading, setIsLoading] = useState(false)
  const [liveComparison, setLiveComparison] = useState<HistoryItem | null>(null)

  // ============= BACKEND CONNECTION NEEDED =============
  // TODO: Replace with actual API call to GET /api/iterations
  // Expected response: Array<{ id: string, label: string, date: string }>
  const iterations = [
    { id: "all", label: "All Iterations", date: "" },
    { id: "iter-3", label: "Iteration 3", date: "2024-01-15" },
    { id: "iter-2", label: "Iteration 2", date: "2024-01-10" },
    { id: "iter-1", label: "Iteration 1", date: "2024-01-05" },
  ]
  // ============= END BACKEND CONNECTION =============

  // ============= BACKEND CONNECTION NEEDED =============
  // TODO: Replace with actual API call to GET /api/history?iteration={selectedIteration}
  // Expected response: Array<{
  //   id: string,
  //   timestamp: string,
  //   iteration: string,
  //   prompt: string,
  //   llmResponse: { content: string, toolCalls?: Array<...> },
  //   slmResponse: { content: string, toolCalls?: Array<...> }
  // }>
  const mockHistory = [
    {
      id: "1",
      timestamp: "2024-01-15 14:23:45",
      iteration: "Iteration 3",
      prompt: "What is the molecular structure of caffeine?",
      llmResponse: {
        content: "Caffeine (C8H10N4O2) is a purine alkaloid with a complex molecular structure...",
        toolCalls: [
          {
            name: "search_database",
            input: { query: "caffeine molecular structure" },
            result: "Found 3 relevant papers",
          },
        ],
      },
      slmResponse: {
        content: "Caffeine has the molecular formula C8H10N4O2. It consists of a fused ring system...",
        toolCalls: [
          {
            name: "search_database",
            input: { query: "caffeine structure" },
            result: "Found 2 papers",
          },
        ],
      },
    },
    {
      id: "2",
      timestamp: "2024-01-14 09:15:22",
      iteration: "Iteration 2",
      prompt: "Explain the process of glycolysis",
      llmResponse: {
        content: "Glycolysis is a metabolic pathway that converts glucose into pyruvate...",
      },
      slmResponse: {
        content: "Glycolysis breaks down glucose to produce ATP and pyruvate...",
      },
    },
  ]
  // ============= END BACKEND CONNECTION =============

  const filteredHistory = Array.isArray(mockHistory)
    ? selectedIteration === "all"
      ? mockHistory
      : mockHistory.filter((item) => item.iteration === selectedIteration)
    : []

  const handleSubmit = async () => {
    if (!prompt.trim()) return

    setIsLoading(true)

    // ============= BACKEND CONNECTION NEEDED =============
    // TODO: Replace with actual API call
    // POST /api/compare
    // Body: { prompt: string, iteration: string }
    // Expected response: {
    //   llmResponse: { content: string, toolCalls?: Array<...> },
    //   slmResponse: { content: string, toolCalls?: Array<...> }
    // }

    // Simulate API call
    setTimeout(() => {
      setLiveComparison({
        id: "live",
        timestamp: new Date().toLocaleString(),
        iteration: selectedIteration,
        prompt: prompt,
        llmResponse: {
          content: "This is a sample LLM response to your prompt...",
          toolCalls: [
            {
              name: "analyze_query",
              input: { text: prompt },
              result: "Analysis complete",
            },
          ],
        },
        slmResponse: {
          content: "This is a sample SLM response to your prompt...",
          toolCalls: [
            {
              name: "analyze_query",
              input: { text: prompt },
              result: "Analysis complete",
            },
          ],
        },
      })
      setIsLoading(false)
    }, 1500)
    // ============= END BACKEND CONNECTION =============
  }

  return (
    <div className="w-full h-full px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-foreground mb-2">Testing</h1>
        <p className="text-muted-foreground">Compare LLM and SLM outputs side-by-side and review past interactions</p>
      </div>

      {/* Unified Testing Component */}
      <Card className="p-6 mb-8 bg-card border-border">
        <h2 className="text-xl font-semibold text-foreground mb-6">Testing</h2>
        
        {/* LLM and SLM Output Panels */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div>
            <div className="flex items-center gap-2 mb-4">
              <Sparkles className="size-5 text-primary" />
              <h3 className="text-lg font-semibold text-foreground">LLM Output</h3>
            </div>
            {liveComparison ? (
              <div className="min-h-[400px]">
                <ChatMessage prompt={liveComparison.prompt} response={liveComparison.llmResponse} type="llm" />
              </div>
            ) : (
              <div className="min-h-[400px] flex items-center justify-center text-muted-foreground border border-border rounded-lg bg-background/50">
                <p>LLM response will appear here</p>
              </div>
            )}
          </div>
          <div>
            <div className="flex items-center gap-2 mb-4">
              <Zap className="size-5 text-accent" />
              <h3 className="text-lg font-semibold text-foreground">SLM Output</h3>
            </div>
            {liveComparison ? (
              <div className="min-h-[400px]">
                <ChatMessage prompt={liveComparison.prompt} response={liveComparison.slmResponse} type="slm" />
              </div>
            ) : (
              <div className="min-h-[400px] flex items-center justify-center text-muted-foreground border border-border rounded-lg bg-background/50">
                <p>SLM response will appear here</p>
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
            className="min-h-24 flex-1 bg-background text-foreground border-border w-full"
          />
          <Button 
            onClick={handleSubmit} 
            disabled={isLoading} 
            className="self-end sm:self-end h-24 bg-primary hover:bg-primary/90 text-primary-foreground"
          >
            <SendHorizontal className="size-4" />
          </Button>
        </div>
      </Card>

      {/* Filter Section */}
      <div className="flex items-center gap-4 mb-6">
        <label className="text-sm font-medium text-foreground">Filter by iteration:</label>
        <Select value={selectedIteration} onValueChange={setSelectedIteration}>
          <SelectTrigger className="w-full sm:w-64 bg-background border-border">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {iterations.map((iteration) => (
              <SelectItem key={iteration.id} value={iteration.id}>
                {iteration.label} {iteration.date && `(${iteration.date})`}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* History Tabs */}
      <Tabs defaultValue="chats" className="w-full">
        <TabsList className="bg-muted">
          <TabsTrigger value="chats">Chat History</TabsTrigger>
          <TabsTrigger value="traces">Traces</TabsTrigger>
        </TabsList>

        <TabsContent value="chats" className="mt-6 space-y-6">
          {filteredHistory.length === 0 ? (
            <Card className="p-12 bg-card border-border text-center">
              <p className="text-muted-foreground">No history available for this iteration</p>
            </Card>
          ) : (
            filteredHistory.map((item) => (
              <Card key={item.id} className="p-6 bg-card border-border">
                <div className="mb-4 flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">{item.timestamp}</span>
                  <span className="text-xs font-medium text-primary">{item.iteration}</span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full">
                  <div>
                    <div className="flex items-center gap-2 mb-3">
                      <Sparkles className="size-4 text-primary" />
                      <span className="text-sm font-semibold text-foreground">LLM</span>
                    </div>
                    <ChatMessage prompt={item.prompt} response={item.llmResponse} type="llm" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2 mb-3">
                      <Zap className="size-4 text-accent" />
                      <span className="text-sm font-semibold text-foreground">SLM</span>
                    </div>
                    <ChatMessage prompt={item.prompt} response={item.slmResponse} type="slm" />
                  </div>
                </div>
              </Card>
            ))
          )}
        </TabsContent>

        <TabsContent value="traces" className="mt-6 space-y-6">
          {filteredHistory.length === 0 ? (
            <Card className="p-12 bg-card border-border text-center">
              <p className="text-muted-foreground">No traces available for this iteration</p>
            </Card>
          ) : (
            filteredHistory.map((item) => (
              <Card key={item.id} className="p-6 bg-card border-border">
                <div className="mb-4 flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">{item.timestamp}</span>
                  <span className="text-xs font-medium text-primary">{item.iteration}</span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full">
                  <div>
                    <div className="flex items-center gap-2 mb-3">
                      <Sparkles className="size-4 text-primary" />
                      <span className="text-sm font-semibold text-foreground">LLM Trace</span>
                    </div>
                    <TraceMessage prompt={item.prompt} response={item.llmResponse} type="llm" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2 mb-3">
                      <Zap className="size-4 text-accent" />
                      <span className="text-sm font-semibold text-foreground">SLM Trace</span>
                    </div>
                    <TraceMessage prompt={item.prompt} response={item.slmResponse} type="slm" />
                  </div>
                </div>
              </Card>
            ))
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}

