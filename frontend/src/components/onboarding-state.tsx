import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Zap, ArrowRight } from "lucide-react"

interface OnboardingStateProps {
  onComplete: () => void
}

export function OnboardingState({ onComplete }: OnboardingStateProps) {
  const [publicKey, setPublicKey] = useState("")
  const [secretKey, setSecretKey] = useState("")
  const [baseUrl, setBaseUrl] = useState("https://cloud.langfuse.com")

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // START: Hard-coded mock - replace with actual API call to LangFuse
    // API endpoint: POST /api/langfuse/connect
    // Payload: { publicKey, secretKey, baseUrl }
    // Expected response: { success: boolean, message: string, data: { projectInfo } }
    fetch("/api/langfuse/connect", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ publicKey, secretKey, baseUrl }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          onComplete()
        } else {
          console.error(data.message)
        }
      })
      .catch((error) => {
        console.error("Error connecting to LangFuse:", error)
      })
    // END: Hard-coded mock
  }

  return (
    <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center p-4 w-full">
      <div className="w-full max-w-2xl space-y-8 px-4">
        <div className="text-center space-y-4">
          <div className="inline-flex items-center justify-center size-16 rounded-2xl bg-primary/10 text-primary">
            <Zap className="size-8" />
          </div>
          <h1 className="text-4xl font-bold tracking-tight text-balance">Train SLMs with LLM-level Performance</h1>
          <p className="text-lg text-muted-foreground text-balance max-w-xl mx-auto">
            {
              "Connect your LangFuse data to automatically fine-tune Small Language Models that match Large Language Model expertise at a fraction of the cost"
            }
          </p>
        </div>

        <Card className="border-border">
          <CardHeader>
            <CardTitle className="text-foreground">Connect LangFuse</CardTitle>
            <CardDescription className="text-muted-foreground">
              Enter your LangFuse credentials to get started with automated model training
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="publicKey" className="text-foreground">
                  Public Key
                </Label>
                <Input
                  id="publicKey"
                  type="text"
                  placeholder="pk-lf-..."
                  value={publicKey}
                  onChange={(e) => setPublicKey(e.target.value)}
                  className="bg-input border-border text-foreground"
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="secretKey" className="text-foreground">
                  Secret Key
                </Label>
                <Input
                  id="secretKey"
                  type="password"
                  placeholder="sk-lf-..."
                  value={secretKey}
                  onChange={(e) => setSecretKey(e.target.value)}
                  className="bg-input border-border text-foreground"
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="baseUrl" className="text-foreground">
                  Base URL
                </Label>
                <Input
                  id="baseUrl"
                  type="url"
                  placeholder="https://cloud.langfuse.com"
                  value={baseUrl}
                  onChange={(e) => setBaseUrl(e.target.value)}
                  className="bg-input border-border text-foreground"
                  required
                />
              </div>
              <Button type="submit" className="w-full bg-primary hover:bg-primary/90 text-primary-foreground" size="lg">
                Connect & Start Training
                <ArrowRight className="ml-2 size-4" />
              </Button>
            </form>
          </CardContent>
        </Card>

        <div className="grid gap-4 md:grid-cols-3">
          <div className="text-center space-y-2">
            <div className="text-2xl font-bold text-primary">10x</div>
            <div className="text-sm text-muted-foreground">Cost Reduction</div>
          </div>
          <div className="text-center space-y-2">
            <div className="text-2xl font-bold text-accent">95%+</div>
            <div className="text-sm text-muted-foreground">Accuracy Retention</div>
          </div>
          <div className="text-center space-y-2">
            <div className="text-2xl font-bold text-chart-3">Auto</div>
            <div className="text-sm text-muted-foreground">Fine-tuning</div>
          </div>
        </div>
      </div>
    </div>
  )
}

