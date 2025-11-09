import { Navigation } from "@/components/navigation"
import { HistoryComparison } from "@/components/history-comparison"

export default function HistoryPage() {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <HistoryComparison />
    </div>
  )
}

