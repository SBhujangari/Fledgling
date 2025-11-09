import { Link } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { ArrowRight, Zap, TrendingDown, TrendingUp, Shield, Code2, Sparkles } from "lucide-react"

function scrollToSection(e: React.MouseEvent<HTMLAnchorElement>, sectionId: string) {
  e.preventDefault()
  const element = document.getElementById(sectionId)
  if (element) {
    element.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }
}

export default function LandingPage() {
  return (
    <div className="min-h-screen w-full bg-background">
      {/* Navigation */}
      <nav className="border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 w-full">
        <div className="w-full px-4 sm:px-6 lg:px-8 flex h-16 items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap className="size-6 text-primary" />
            <span className="text-xl font-semibold text-foreground">Fledgling</span>
          </div>
          <div className="flex items-center gap-4">
            <a
              href="#how-it-works"
              onClick={(e) => scrollToSection(e, 'how-it-works')}
              className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
            >
              How It Works
            </a>
            <Link to="/dashboard">
              <Button size="sm">Get Started</Button>
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="w-full px-4 sm:px-6 lg:px-8 py-24 md:py-32">
        <div className="flex flex-col items-center text-center gap-8 max-w-4xl mx-auto w-full">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20">
            <Sparkles className="size-4 text-primary" />
            <span className="text-sm font-medium text-primary">Save up to 90% on AI costs</span>
          </div>

          <h1 className="text-5xl md:text-7xl font-bold text-balance leading-tight">
            Train Small AI Models to <span className="text-primary">Think Like Giants</span>
          </h1>

          <p className="text-xl text-muted-foreground text-balance max-w-2xl leading-relaxed">
            Stop paying enterprise prices for simple tasks. Fledgling automatically trains efficient small language
            models that match the performance of expensive LLMs—at a fraction of the cost.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 mt-4">
            <Link to="/dashboard">
              <Button size="lg" className="gap-2">
                Start Tuning <ArrowRight className="size-4" />
              </Button>
            </Link>
            <Button size="lg" variant="outline">
              Watch Demo
            </Button>
          </div>

          {/* Visual Cost Comparison */}
          <div className="w-full mt-16 grid md:grid-cols-2 gap-6">
            <Card className="border-destructive/50 bg-destructive/5">
              <CardContent className="pt-6">
                <div className="flex items-center gap-3 mb-4">
                  <TrendingUp className="size-5 text-destructive" />
                  <span className="font-semibold text-foreground">Using Large Models (LLMs)</span>
                </div>
                <div className="text-4xl font-bold text-destructive mb-2">$10,000</div>
                <p className="text-sm text-muted-foreground">Monthly inference costs for 1M requests</p>
                <div className="mt-4 space-y-2 text-sm text-muted-foreground">
                  <div>• High latency (2-5 seconds)</div>
                  <div>• Expensive per-token pricing</div>
                  <div>• Overkill for specific tasks</div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-primary/50 bg-primary/5 relative">
              <div className="absolute -top-3 -right-3 bg-primary text-primary-foreground px-3 py-1 rounded-full text-xs font-semibold">
                90% Savings
              </div>
              <CardContent className="pt-6">
                <div className="flex items-center gap-3 mb-4">
                  <TrendingDown className="size-5 text-primary" />
                  <span className="font-semibold text-foreground">Using Fledgling (SLMs)</span>
                </div>
                <div className="text-4xl font-bold text-primary mb-2">$1,000</div>
                <p className="text-sm text-muted-foreground">Monthly inference costs for 1M requests</p>
                <div className="mt-4 space-y-2 text-sm text-muted-foreground">
                  <div>• Low latency (&lt;500ms)</div>
                  <div>• 10x cheaper per-token</div>
                  <div>• Optimized for your exact task</div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="border-t border-border bg-muted/30 py-24 w-full">
        <div className="w-full px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">How Fledgling Works</h2>
            <p className="text-xl text-muted-foreground text-balance max-w-2xl mx-auto">
              Three simple steps to deploy cost-effective AI models without sacrificing quality
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            <Card>
              <CardContent className="pt-6">
                <div className="size-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                  <Code2 className="size-6 text-primary" />
                </div>
                <div className="text-2xl font-bold mb-2">1. Connect</div>
                <p className="text-muted-foreground leading-relaxed">
                  Link your LangFuse account to import your existing LLM prompts and responses. We analyze your usage
                  patterns automatically.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="size-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                  <Sparkles className="size-6 text-primary" />
                </div>
                <div className="text-2xl font-bold mb-2">2. Train</div>
                <p className="text-muted-foreground leading-relaxed">
                  Our platform automatically fine-tunes a small language model using your LLM's outputs as training
                  data. No ML expertise required.
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="size-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                  <Shield className="size-6 text-primary" />
                </div>
                <div className="text-2xl font-bold mb-2">3. Deploy</div>
                <p className="text-muted-foreground leading-relaxed">
                  Deploy your optimized SLM with 95%+ accuracy retention and 90% cost savings. Monitor performance in
                  real-time.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Use Cases */}
      <section className="py-24 w-full">
        <div className="w-full px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Perfect for Specialized Tasks</h2>
            <p className="text-xl text-muted-foreground text-balance max-w-2xl mx-auto">
              Stop using general-purpose LLMs for specialized workflows
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
            {[
              {
                title: "Customer Support",
                description: "Train SLMs on your support tickets to answer common questions at 1/10th the cost",
              },
              {
                title: "Content Moderation",
                description: "Faster, cheaper content filtering trained on your specific community guidelines",
              },
              {
                title: "Code Generation",
                description: "Generate boilerplate code using models trained on your codebase patterns",
              },
              {
                title: "Data Extraction",
                description: "Extract structured data from documents with domain-specific accuracy",
              },
              {
                title: "Classification",
                description: "Classify text, images, or data with models optimized for your categories",
              },
              {
                title: "Translation",
                description: "Domain-specific translation maintaining terminology consistency",
              },
            ].map((useCase, index) => (
              <Card key={index} className="hover:border-primary/50 transition-colors">
                <CardContent className="pt-6">
                  <h3 className="text-lg font-semibold mb-2">{useCase.title}</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">{useCase.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="border-t border-border bg-muted/30 py-24 w-full">
        <div className="w-full px-4 sm:px-6 lg:px-8">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="text-4xl md:text-5xl font-bold mb-6">Ready to Cut Your AI Costs?</h2>
            <p className="text-xl text-muted-foreground mb-8 text-balance">
              Start training efficient models today. No credit card required.
            </p>
            <Link to="/dashboard">
              <Button size="lg" className="gap-2">
                Get Started Free <ArrowRight className="size-4" />
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-12 w-full">
        <div className="w-full px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <div className="flex items-center gap-2">
              <Zap className="size-5 text-primary" />
              <span className="font-semibold">Fledgling</span>
            </div>
            <div className="text-sm text-muted-foreground">© 2025 Fledgling. All rights reserved.</div>
          </div>
        </div>
      </footer>
    </div>
  )
}
