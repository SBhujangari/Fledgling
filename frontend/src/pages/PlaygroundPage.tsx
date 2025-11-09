import { Navigation } from "@/components/navigation"
import { Playground } from "@/components/playground"

export default function PlaygroundPage() {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <Playground />
    </div>
  )
}

