import { useState } from "react"
import { Navigation } from "@/components/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { FlaskConical, Sparkles, Copy, CheckCircle2, AlertCircle } from "lucide-react"

interface ISATabData {
  study_identifier: string
  study_title: string
  study_description: string
  study_submission_date: string
  study_public_release_date: string
  study_file_name: string
  study_factors: Array<{
    name: string
    type: string
    values: string[]
  }>
  study_assays: Array<{
    measurement_type: string
    technology_type: string
    technology_platform: string
  }>
  study_protocols: Array<{
    name: string
    type: string
    description: string
    parameters: string[]
  }>
  study_contacts: Array<{
    name: string
    affiliation: string
    email: string
    role: string
  }>
}

interface ExtractionResult {
  success: boolean
  data?: ISATabData
  error?: string
  raw_notes?: string
  confidence_score?: number
}

export default function LabAutologgerPage() {
  const [labNotes, setLabNotes] = useState("")
  const [extractionResult, setExtractionResult] = useState<ExtractionResult | null>(null)
  const [isExtracting, setIsExtracting] = useState(false)
  const [copied, setCopied] = useState(false)

  const extractData = async () => {
    if (!labNotes.trim()) {
      setExtractionResult({
        success: false,
        error: "Please enter some lab notes to extract data from"
      })
      return
    }

    setIsExtracting(true)
    try {
      const response = await fetch("http://localhost:4000/api/lab-autologger/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ labNotes })
      })

      if (!response.ok) {
        throw new Error(`Extraction failed: ${response.statusText}`)
      }

      const result = await response.json()
      setExtractionResult(result)
    } catch (error) {
      console.error("Extraction error:", error)
      setExtractionResult({
        success: false,
        error: error instanceof Error ? error.message : "Unknown error occurred"
      })
    } finally {
      setIsExtracting(false)
    }
  }

  const copyToClipboard = () => {
    if (extractionResult?.data) {
      navigator.clipboard.writeText(JSON.stringify(extractionResult.data, null, 2))
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  const exampleNotes = `Lab Notebook - Oct 15, 2024
Researcher: Dr. Sarah Chen (sarah.chen@biotech.edu), MIT Biology Dept

Study: Effect of Temperature on E. coli Growth Rates
Running a series of growth curve experiments to measure how different temperatures affect E. coli strain DH5α growth.

Protocol:
- Inoculated 5 flasks with 50mL LB medium each
- Incubated at different temps: 25°C, 30°C, 37°C, 42°C, 45°C
- Measured OD600 every hour for 12 hours
- Used BioTek plate reader

Preliminary observations: 37°C showing optimal growth, 45°C shows significant growth inhibition.
Will collect full dataset and analyze tomorrow.`

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navigation />
      <main className="w-full px-4 py-8 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl space-y-8">
          {/* Header */}
          <header className="space-y-2 border-b border-border pb-6">
            <div className="flex items-center gap-3">
              <FlaskConical className="h-8 w-8 text-primary" />
              <h1 className="text-3xl font-semibold tracking-tight">Lab Autologger</h1>
            </div>
            <p className="text-muted-foreground">
              Extract ISA-Tab compliant experiment metadata from unstructured lab notes using AI
            </p>
          </header>

          <div className="grid gap-8 lg:grid-cols-2">
            {/* Input Section */}
            <Card>
              <CardHeader>
                <CardTitle>Lab Notes Input</CardTitle>
                <CardDescription>
                  Paste your lab notes, email, or any unstructured experiment description
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  value={labNotes}
                  onChange={(e) => setLabNotes(e.target.value)}
                  placeholder="Enter your lab notes here..."
                  className="min-h-[400px] font-mono text-sm"
                />
                <div className="flex items-center gap-3">
                  <Button
                    onClick={extractData}
                    disabled={isExtracting || !labNotes.trim()}
                    className="gap-2"
                  >
                    {isExtracting ? (
                      <>
                        <Sparkles className="h-4 w-4 animate-pulse" />
                        Extracting...
                      </>
                    ) : (
                      <>
                        <Sparkles className="h-4 w-4" />
                        Extract ISA-Tab Data
                      </>
                    )}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => setLabNotes(exampleNotes)}
                  >
                    Load Example
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Output Section */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Extracted ISA-Tab Metadata</CardTitle>
                    <CardDescription>
                      Structured experiment data ready for submission
                    </CardDescription>
                  </div>
                  {extractionResult?.success && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={copyToClipboard}
                      className="gap-2"
                    >
                      {copied ? (
                        <>
                          <CheckCircle2 className="h-4 w-4 text-green-500" />
                          Copied!
                        </>
                      ) : (
                        <>
                          <Copy className="h-4 w-4" />
                          Copy JSON
                        </>
                      )}
                    </Button>
                  )}
                </div>
              </CardHeader>
              <CardContent>
                {!extractionResult && (
                  <div className="flex items-center justify-center min-h-[400px] text-muted-foreground">
                    <div className="text-center space-y-2">
                      <FlaskConical className="h-12 w-12 mx-auto opacity-20" />
                      <p>No data extracted yet</p>
                      <p className="text-sm">Enter lab notes and click "Extract ISA-Tab Data"</p>
                    </div>
                  </div>
                )}

                {extractionResult?.error && (
                  <div className="flex items-center gap-3 p-4 rounded-lg bg-destructive/10 border border-destructive/20">
                    <AlertCircle className="h-5 w-5 text-destructive" />
                    <div>
                      <p className="font-semibold text-destructive">Extraction Failed</p>
                      <p className="text-sm text-muted-foreground">{extractionResult.error}</p>
                    </div>
                  </div>
                )}

                {extractionResult?.success && extractionResult.data && (
                  <Tabs defaultValue="structured" className="w-full">
                    <TabsList className="bg-muted">
                      <TabsTrigger value="structured">Structured View</TabsTrigger>
                      <TabsTrigger value="json">JSON View</TabsTrigger>
                    </TabsList>

                    <TabsContent value="structured" className="space-y-4 mt-4">
                      {extractionResult.confidence_score && (
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-muted-foreground">Confidence:</span>
                          <Badge variant={extractionResult.confidence_score > 0.8 ? "default" : "secondary"}>
                            {(extractionResult.confidence_score * 100).toFixed(0)}%
                          </Badge>
                        </div>
                      )}

                      <div className="space-y-4">
                        {/* Study Information */}
                        <div className="space-y-2">
                          <h3 className="font-semibold text-sm text-muted-foreground uppercase">Study</h3>
                          <div className="space-y-1 text-sm">
                            <p><span className="font-medium">Title:</span> {extractionResult.data.study_title}</p>
                            <p><span className="font-medium">Description:</span> {extractionResult.data.study_description}</p>
                            <p><span className="font-medium">Identifier:</span> {extractionResult.data.study_identifier}</p>
                          </div>
                        </div>

                        {/* Contacts */}
                        {extractionResult.data.study_contacts.length > 0 && (
                          <div className="space-y-2">
                            <h3 className="font-semibold text-sm text-muted-foreground uppercase">Contacts</h3>
                            {extractionResult.data.study_contacts.map((contact, idx) => (
                              <div key={idx} className="rounded-lg border border-border p-3 space-y-1 text-sm">
                                <p className="font-medium">{contact.name}</p>
                                <p className="text-muted-foreground">{contact.affiliation}</p>
                                <p className="text-muted-foreground">{contact.email}</p>
                                <Badge variant="outline" className="text-xs">{contact.role}</Badge>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Assays */}
                        {extractionResult.data.study_assays.length > 0 && (
                          <div className="space-y-2">
                            <h3 className="font-semibold text-sm text-muted-foreground uppercase">Assays</h3>
                            {extractionResult.data.study_assays.map((assay, idx) => (
                              <div key={idx} className="rounded-lg border border-border p-3 space-y-1 text-sm">
                                <p><span className="font-medium">Measurement:</span> {assay.measurement_type}</p>
                                <p><span className="font-medium">Technology:</span> {assay.technology_type}</p>
                                <p><span className="font-medium">Platform:</span> {assay.technology_platform}</p>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Protocols */}
                        {extractionResult.data.study_protocols.length > 0 && (
                          <div className="space-y-2">
                            <h3 className="font-semibold text-sm text-muted-foreground uppercase">Protocols</h3>
                            {extractionResult.data.study_protocols.map((protocol, idx) => (
                              <div key={idx} className="rounded-lg border border-border p-3 space-y-1 text-sm">
                                <p className="font-medium">{protocol.name}</p>
                                <p className="text-muted-foreground">{protocol.description}</p>
                                <Badge variant="outline" className="text-xs">{protocol.type}</Badge>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Factors */}
                        {extractionResult.data.study_factors.length > 0 && (
                          <div className="space-y-2">
                            <h3 className="font-semibold text-sm text-muted-foreground uppercase">Experimental Factors</h3>
                            {extractionResult.data.study_factors.map((factor, idx) => (
                              <div key={idx} className="rounded-lg border border-border p-3 space-y-1 text-sm">
                                <p className="font-medium">{factor.name}</p>
                                <p className="text-muted-foreground">Type: {factor.type}</p>
                                <p className="text-muted-foreground">Values: {factor.values.join(", ")}</p>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </TabsContent>

                    <TabsContent value="json" className="mt-4">
                      <pre className="rounded-lg bg-muted p-4 text-xs overflow-auto max-h-[500px] font-mono">
                        {JSON.stringify(extractionResult.data, null, 2)}
                      </pre>
                    </TabsContent>
                  </Tabs>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Info Section */}
          <Card className="bg-primary/5 border-primary/20">
            <CardHeader>
              <CardTitle className="text-lg">About ISA-Tab Format</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-muted-foreground">
              <p>
                ISA-Tab (Investigation, Study, Assay) is a standardized format for describing experimental metadata
                in life sciences. This tool uses AI to automatically extract structured ISA-Tab data from your
                unstructured lab notes.
              </p>
              <p className="font-medium text-foreground mt-4">Extracted Fields:</p>
              <ul className="list-disc list-inside space-y-1 ml-2">
                <li>Study identification and description</li>
                <li>Researcher contacts and affiliations</li>
                <li>Experimental protocols and procedures</li>
                <li>Assay types and measurement methods</li>
                <li>Experimental factors and variables</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}
